import os
import sys
import json
__package__ = 'cores'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from transformers import PretrainedConfig, GenerationMixin, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from utils.util import RMSNorm, MOEFFN, FFN, apply_rope_position_encoding, sinusoidal_positional_encoding
from utils.attentions import MultiHeadLatentAttention, MultiHeadAttentionByPspp


class MiniLLMConfig(PretrainedConfig):
    model_type = "minillm"
    DEFAULT_CONFIG = {
          "model_name": "",
          "intermediate_size": None,
          "data_path": "",
          "tokenizer_path": "",
          "save_path": ""
    }

    def __init__(self, config_path=None, **kwargs):
        if config_path is None:
            config = {**self.DEFAULT_CONFIG, **kwargs}
        else:
            with open(config_path, "r") as f:
                config_file = json.load(f)
                config = {**self.DEFAULT_CONFIG, **config_file, **kwargs}
        for k, v in config.items():
            setattr(self, k, v)
        super().__init__(**config)
        self.config_path = config_path

class MiniLLMBlock(nn.Module):
    def __init__(self, layer_id, config:MiniLLMConfig):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.attn = MultiHeadAttentionByPspp(self.config.head_nums, config.hidden_dim, config.hidden_dim)
        self.norm = RMSNorm(config)
        self.after_attn_norm = RMSNorm(config)
        self.mlp = MOEFFN(config) if self.config.use_moe else FFN(config)

    def forward(self, x, position_embeddings=None, attention_mask=None, use_cache=False, past_key_values=None, **kwargs):
        residual = x
        norm = self.norm(x)
        attn, past_key_values = self.attn(norm, position_embeddings=position_embeddings, attention_mask=attention_mask, use_cache=use_cache, past_key_values=past_key_values, **kwargs)
        attn = attn + residual
        after_attn_norm = self.after_attn_norm(attn)
        mlp = self.mlp(after_attn_norm)
        out = mlp + attn
        return out, past_key_values

class MiniLLM(nn.Module):
    def __init__(self, config:MiniLLMConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.attn_layers = nn.ModuleList([MiniLLMBlock(i, self.config) for i in range(self.config.num_hidden_layers)])
        self.layer_norm = RMSNorm(config)
        self.dropout = nn.Dropout(self.config.dropout)
        if self.config.attn_type in ["Attention"]:
            positional_embed_dim = self.config.hidden_dim
        else:
            positional_embed_dim = self.config.hidden_dim // self.config.head_nums
        positional_encoding = sinusoidal_positional_encoding(positional_embed_dim, self.config.max_len, theta_para=self.config.theta_para, dtype=torch.float32)
        self.register_buffer('positional_encoding', positional_encoding, persistent=False)# persistent指是否保存在state_dict中

    def forward(self, input_ids, attention_mask=None, use_cache=False, past_key_values=None, **kwargs):
        batch_size, token_len = input_ids.shape
        hidden_states = self.dropout(self.embed(input_ids))

        if hasattr(past_key_values, "layers"): past_key_values = None
        if past_key_values is not None:
            start_positions = past_key_values[0][0].shape[1]
        else:
            past_key_values = [None] * self.config.num_hidden_layers
            start_positions = 0

        # add position embedding
        position_embeddings = self.positional_encoding[start_positions:start_positions + token_len]

        # 这里可以减少position的长度，根据缓存判断
        attn_outputs = []
        for layer_id, attn in enumerate(self.attn_layers):
            hidden_states, past_key_value = attn(hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask, use_cache=use_cache, past_key_values=past_key_values[layer_id], **kwargs)
            attn_outputs.append(past_key_value)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states, attn_outputs

class MiniLLMForCasualModel(PreTrainedModel, GenerationMixin):
    config_class = MiniLLMConfig
    base_model_prefix = "minillm"
    supports_gradient_checkpointing = True
    def __init__(self, config:MiniLLMConfig):
        super().__init__(config)
        self.config = config
        self.model = MiniLLM(config)
        # lm_head   必须要有的
        self.lm_head = nn.Linear(self.config.hidden_dim, self.config.vocab_size)
        self.lm_head.weight = self.model.embed.weight

    def forward(self, input_ids, attention_mask=None, use_cache=False, past_key_values=None, logits_to_keep=0, return_dict=True, **kwargs):
        hidden_states, past_key_values = self.model(input_ids, attention_mask=attention_mask, use_cache=use_cache, past_key_values=past_key_values, **kwargs)
        #use_cache is True, logits_to_keep will be 1, then just the final token will be predicted.
        slice_index = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # logits.shape is [batch_size, seq_len, vocabulary]
        logits = self.lm_head(hidden_states[:, slice_index, :])

        output = CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)

        # if has other loss such as moe aux loss, you can add loss like this:
        output.aux_loss = 0

        if return_dict:
            return output
        else:
            return (output.logits, past_key_values, hidden_states)
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
          "data_path": "",
          "tokenizer_path": "",
          "save_path": ""
    }

    def __init__(self, config_path=None, **kwargs):
        if config_path is None:
            self.config = {**self.DEFAULT_CONFIG, **kwargs}
        else:
            with open(config_path, "r") as f:
                config_file = json.load(f)
                self.config = { **config_file, **self.DEFAULT_CONFIG, **kwargs}
        super().__init__(**self.config)
        self.config_path = config_path
        self.init()

    def init(self):
        for k, v in self.config.items():
            setattr(self, k, v)


class MiniLLMBlock(nn.Module):
    def __init__(self, layer_id, config:MiniLLMConfig):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.attn = eval(self.config.attn_type)(self.config.head_nums, config.embed_dim, config.embed_dim, self.config.latent_dim)
        self.norm1 = RMSNorm(self.config.embed_dim, self.config.eps)
        self.norm2 = RMSNorm(self.config.embed_dim, self.config.eps)
        self.mlp = MOEFFN(self.config.export_nums, self.config.per_token_exports, self.config.embed_dim, self.config.hidden_dim) if self.config.use_moe else FFN(self.config.embed_dim, self.config.hidden_dim)

    def forward(self, x, position_embeddings=None, attention_mask=None, past_key_values=None, use_cached=False, **kwargs):
        residual = x
        norm1 = self.norm1(x)
        attn = self.attn(norm1, use_cached, position_embeddings, past_key_values,  attention_mask, **kwargs)
        attn = attn + residual
        norm2 = self.norm2(attn)
        mlp = self.mlp(norm2)
        out = mlp + attn
        return out


class MiniLLM(nn.Module):
    def __init__(self, config:MiniLLMConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.attn_layers = nn.ModuleList([MiniLLMBlock(i, self.config) for i in range(self.config.num_layers)])
        self.layer_norm = RMSNorm(self.config.embed_dim)
        self.dropout = nn.Dropout(self.config.dropout)
        if self.config.attn_type in ["Attention"]:
            positional_embed_dim = self.config.embed_dim
        else:
            positional_embed_dim = self.config.embed_dim // self.config.head_nums
        positional_encoding = sinusoidal_positional_encoding(positional_embed_dim, self.config.max_len, theta_para=self.config.theta_para, dtype=torch.float32)
        self.register_buffer('positional_encoding', positional_encoding, persistent=False)# persistent指是否保存在state_dict中

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cached=False, **kwargs):
        batch_size, token_len = input_ids.shape
        hidden_states = self.dropout(self.embed(input_ids))
        if use_cached:
            if past_key_values is None:
                start_pos = 0
            elif isinstance(past_key_values, DynamicCache):
                if past_key_values.layers[0].values is None:
                    start_pos = 0
                else:
                    start_pos = len(past_key_values.layers)
            elif isinstance(past_key_values, torch.Tensor):
                start_pos = past_key_values.shape[0]
            else:
                start_pos = 0
        else:
            start_pos = 0

        # 这里可以减少position的长度，根据缓存判断
        attn_outputs = []
        for layer_id, attn in enumerate(self.attn_layers):
            hidden_states = attn(hidden_states, position_embeddings=self.positional_encoding, attention_mask=attention_mask, past_key_values=past_key_values, use_cached=use_cached, **kwargs)
            attn_outputs.append(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states, attn_outputs

class MiniLLMForCasualModel(PreTrainedModel, GenerationMixin):
    config_class = MiniLLMConfig
    base_model_prefix = "minillm"
    supports_gradient_checkpointing = True
    def __init__(self, config:MiniLLMConfig, loss_fn=nn.CrossEntropyLoss(reduction="none")):
        super().__init__()
        self.config = config
        self.model = MiniLLM(config)
        # lm_head   必须要有的
        self.lm_head = nn.Linear(self.config.embed_dim, self.config.vocab_size)
        self.lm_head.weight = self.model.embed.weight
        self.loss_fn = loss_fn

    # It's import for causal model, and exec before forward
    def prepare_inputs_for_generation(self, input_ids, past_key_value=None, attention_mask=None, **kwargs):
        # 如果有缓存，只取最后一个token输入即可
        if past_key_value is not None:
            input_ids = input_ids[:, -1:]
        return {"input_ids": input_ids, "past_key_value": past_key_value, "attention_mask": attention_mask, "use_cached": kwargs.get("use_cached")}


    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cached=False, labels=None, mask_loss=None, **kwargs):
        hidden_states, attn_outputs = self.model(input_ids, attention_mask, past_key_values, use_cached)
        # logits.shape is [batch_size, seq_len, vocabulary]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.config.vocab_size), labels.view(-1)).view(labels.size())
            loss = (loss * mask_loss).sum() / mask_loss.sum()

        return CausalLMOutputWithPast(loss=loss, logits=logits, hidden_states=hidden_states, **kwargs)


if __name__ == '__main__':
    mini_config = MiniLLMConfig('../configs/minillm_config.json')
    print("----------miniLLM Model-------------")
    model = MiniLLM(mini_config)
    data = torch.randint(0, 6399, size=(1, 13))
    hidden_states, _ = model(data, use_cached=True)
    print("LLM hidden_states.shape:", hidden_states.shape)

    print("----------Test Model--------------")
    layer = nn.Embedding(5, 3)
    print(layer)
    print(layer.weight)

    data = torch.randint(0, 4, size=(1, 2))
    print(data)
    res = layer(data)
    print(res)
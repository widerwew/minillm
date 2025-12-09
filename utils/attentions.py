import torch
import torch.nn as nn
from .util import apply_rope_position_encoding, sinusoidal_positional_encoding

class Attention(nn.Module):
    def __init__(self, din, dout,  bias=False, causal=True, flash_attention=False, context_length=8192):
        super().__init__()
        self.querys = nn.Linear(din, dout, bias=bias)
        self.keys = nn.Linear(din, dout, bias=bias)
        self.values = nn.Linear(din, dout, bias=bias)
        self.causal = causal
        self.flash_attention = flash_attention
        if self.causal:
            # 这里注意mask不能设置和din的维度一致，因为q x k后维度只和token有关系
            self.register_buffer('causal_mask', torch.tril(torch.ones(context_length, context_length), diagonal=0))

    def forward(self, x, position_embedding=None):
        # x.shape is batch, token, din
        batch_size, token_len, dim = x.shape
        query = self.querys(x)
        key = self.keys(x)
        value = self.values(x)
        # add position embedding
        query = apply_rope_position_encoding(query, position_embedding, unsqueeze_dim=None)
        key = apply_rope_position_encoding(key, position_embedding, unsqueeze_dim=None)

        attn_weights = query @ key.transpose(-2, -1)

        if self.causal:
            self.causal_mask.masked_fill(self.causal_mask==0, float('-inf'))
            attn_weights = attn_weights * self.causal_mask[:token_len, :token_len]

        attn_weights = (attn_weights / (attn_weights.shape[-1] ** 0.5)).softmax(dim=-1)
        attn_scores = attn_weights @ value
        return attn_scores

# MHA
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, din, dout, bias=False, causal=True, flash_attention=False, context_length=8192):
        super().__init__()
        self.heads = heads
        self.din = din
        self.dout = dout
        self.causal = causal
        self.flash_attention = flash_attention
        self.bias = bias
        assert dout % heads == 0, "dout must be divisible by heads"
        self.head_dim = dout // heads
        self.querys = nn.Linear(self.din, self.dout, bias=bias)
        self.keys = nn.Linear(self.din, self.dout, bias=bias)
        self.values = nn.Linear(self.din, self.dout, bias=bias)
        self.out_proj = nn.Linear(self.dout, self.dout, bias=bias)

        if self.causal:
            self.register_buffer("causal_mask", torch.tril(torch.ones(context_length, context_length), diagonal=0))


    def forward(self, x, position_embedding=None):
        batch_size, token_len, dim = x.shape
        query = self.querys(x)
        key = self.keys(x)
        value = self.values(x)

        # batch_size, token_len, dim ----> batch_size, heads, token_len, head_dim
        query = query.view(batch_size, token_len, self.heads, self.head_dim)
        key = key.view(batch_size, token_len, self.heads, self.head_dim)

        # add position_embedding
        query = apply_rope_position_encoding(query, position_embedding, unsqueeze_dim=1)
        key = apply_rope_position_encoding(key, position_embedding, unsqueeze_dim=1)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.view(batch_size, token_len, self.heads, self.head_dim).transpose(1, 2)

        attn_weights = query @ key.transpose(3, 2)
        if self.causal:
            self.causal_mask.masked_fill(self.causal_mask==0, float('-inf'))
            attn_weights = attn_weights * self.causal_mask[:token_len, :token_len]

        attn_weights = (attn_weights / (attn_weights.shape[-1] ** 0.5)).softmax(dim=-1)
        attn_scores = attn_weights @ value
        attn_scores = attn_scores.transpose(2, 1).contiguous().view(batch_size, token_len, -1)# Why not weight value of sum？
        attn_scores = self.out_proj(attn_scores)
        return attn_scores

# MHA with pytorch Scale pot product
class MultiHeadAttentionByPspp(nn.Module):
    def __init__(self, heads, din, dout, bias=False, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.din = din
        self.dout = dout
        assert dout % heads == 0, "dout must be divisible by heads"
        self.head_dim = dout // heads
        self.qkv = nn.Linear(self.din, self.dout*3, bias=bias)
        self.dropout = dropout
        self.proj = nn.Linear(self.dout, self.dout, bias=bias)

    def forward(self, x, position_embedding=None):
        batch_size, token_len, dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, token_len, 3, self.heads, self.head_dim)

        # batch_size, token_len, 3, heads, head_dim ---> 3, batch_size, token_len, heads, head_dim
        qkv = qkv.permute(2, 0, 1, 3, 4).contiguous()
        # batch_size, token_len, heads, head_dim
        query, key, value = qkv
        # add position embedding
        query = apply_rope_position_encoding(query, position_embedding, unsqueeze_dim=1)
        key = apply_rope_position_encoding(key, position_embedding, unsqueeze_dim=1)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attention_scores = nn.functional.scaled_dot_product_attention(query, key, value, dropout_p=self.dropout, is_causal=True, attn_mask=None)
        attention_scores = attention_scores.transpose(2, 1).contiguous().view(batch_size, token_len, -1)
        attention_scores = self.proj(attention_scores)
        return attention_scores

#MLA
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, heads, din, dout, latent_dim, bias=False, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.din = din
        self.dout = dout
        assert dout % heads == 0, "dout must be divisible by heads"
        self.head_dim = dout // heads
        self.latent_dim = latent_dim
        self.query = nn.Linear(self.din, self.dout, bias=bias)
        self.latent_layer = nn.Linear(self.din, self.latent_dim, bias=bias)
        self.up_layer = nn.Linear(self.latent_dim, 2 * self.dout, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(self.dout, self.dout, bias=bias)
        self.register_buffer("cache_kv", None)
        self.ptr_index = 0

    def _reset_cache(self):
        self.cache_kv = None
        self.ptr_index = 0

    def forward(self, x, use_cache=False, position_embeddings=None, past_key_value=None, attention_mask=None, **kwargs):
        batch_size, token_len, dim = x.shape
        q_state = self.query(x)
        kv_latent = self.latent_layer(x)
        #--------------------缓存kv-----------------
        if use_cache:
            if self.cache_kv is None:
                self.cache_kv = kv_latent
            else:
                self.cache_kv = torch.cat([self.cache_kv, kv_latent], dim=0)
        else:
            self.cache_kv = kv_latent

        kv_state = self.up_layer(self.cache_kv)
        k_state, v_state = kv_state.view(batch_size, -1, 2, self.dout).permute(2, 0, 1, 3)

        q_state = q_state.view(batch_size, token_len, self.heads, self.head_dim).transpose(1, 2)
        q_state = apply_rope_position_encoding(q_state, position_embeddings[:token_len, :])

        k_state = k_state.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        v_state = v_state.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)

        k_state = apply_rope_position_encoding(k_state, position_embeddings[:k_state.shape[-2], :])
        attn_weights = q_state @ k_state.transpose(-2, -1)

        #-------------------attn mask----------------------
        num_token_q = attn_weights.shape[-2]
        num_token_k = attn_weights.shape[-1]
        if use_cache:
            q_positions = torch.arange(self.ptr_index, self.ptr_index+num_token_q, dtype=torch.long, device=q_state.device)
            self.ptr_index += num_token_q
        else:
            q_positions = torch.arange(num_token_q, dtype=torch.long, device=q_state.device)
            self.ptr_index = num_token_q
        k_positions = torch.arange(num_token_k, dtype=torch.long, device=k_state.device)
        att_mask = q_positions.unsqueeze(1) < k_positions.unsqueeze(0)
        attn_weights = attn_weights.masked_fill(att_mask, float('-inf'))

        attn_weights = (attn_weights / (attn_weights.shape[-1] ** 0.5)).softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        # batch_size, heads, num_token_q, head_dim
        attn_scores = attn_weights @ v_state
        attn_scores = attn_scores.transpose(2, 1).contiguous().view(batch_size, token_len, -1)
        attn_scores = self.proj(attn_scores)
        return attn_scores


#MQA
# multi query heads use a common key and value head.
class MultiQueryAttention(nn.Module):
    def __init__(self, heads, din, dout, bias=False, dropout=0.0, causal=True, context_length=8192):
        super().__init__()
        self.heads = heads
        self.din = din
        self.dout = dout
        assert dout % heads == 0, "dout must be divisible by heads"
        self.head_dim = dout // heads
        self.querys = nn.Linear(self.din, self.dout, bias=bias)
        self.keys = nn.Linear(self.din, self.head_dim, bias=bias)
        self.values = nn.Linear(self.din, self.head_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(self.dout, self.dout, bias=bias)
        self.causal = causal
        if self.causal:
            self.register_buffer("causal_mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x, position_embeddings=None):
        batch_size, token_len, dim = x.shape
        query = self.querys(x)

        # batch_size, token_len, head_dim
        key = self.keys(x)
        value = self.values(x)

        #batch_size, token_len, heads, head_dim ---> batch_size, heads, token_len, head_dim
        query = query.view(batch_size, token_len, self.heads, self.head_dim)
        #add position_embedding
        query = apply_rope_position_encoding(query, position_embeddings, unsqueeze_dim=1)
        key = apply_rope_position_encoding(key, position_embeddings, unsqueeze_dim=None)

        # ---> batch_size, heads, token_len, head_dim
        query = query.transpose(1, 2)
        attention_scores = query @ key.transpose(-2, -1)
        if self.causal:
            attention_scores.masked_fill(self.causal_mask[:token_len, :token_len].bool(), float('-inf'))

        attention_scores = (attention_scores / (attention_scores.shape[-1] ** 0.5)).softmax(dim=-1)
        attention_scores = self.dropout(attention_scores)

        # batch_size, heads, token_len, head_dim
        attention_scores = attention_scores @ value
        attention_scores = attention_scores.transpose(1, 2).contiguous().view(batch_size, token_len, -1)
        attention_scores = self.out_proj(attention_scores)
        return attention_scores


#GQA
# multi group query heads share a common key and value
class GroupQueryAttention(nn.Module):
    def __init__(self, groups, heads, din, dout, bias=False, dropout=0.0, causal=True, context_length=8192):
        super().__init__()
        self.heads = heads
        self.din = din
        self.dout = dout
        self.groups = groups
        assert heads % groups == 0, "heads must be divisible by groups"
        self.head_nums_per_group = heads // groups
        assert dout % heads == 0, "dout must be divisible by heads"
        self.head_dim = dout // heads
        self.querys = nn.Linear(self.din, self.dout, bias=bias)
        self.keys = nn.Linear(self.din, self.groups * self.head_dim, bias=bias)
        self.values = nn.Linear(self.din, self.groups * self.head_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(self.dout, self.dout, bias=bias)
        self.causal = causal
        if self.causal:
            self.register_buffer("causal_mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x, position_embeddings=None):
        batch_size, token_len, dim = x.shape
        query = self.querys(x)
        key = self.keys(x)
        value = self.values(x)
        query = query.view(batch_size, token_len, self.groups, self.head_nums_per_group, self.head_dim)
        key = key.view(batch_size, token_len, self.groups, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, token_len, self.groups, self.head_dim).transpose(1, 2)


        # add position embeddings
        query = query.permute(0, 2, 1, 3, 4).contiguous()
        print("query", query.shape)
        query = apply_rope_position_encoding(query, position_embeddings, unsqueeze_dim=1)
        query = query.transpose(-2, -3)
        key = apply_rope_position_encoding(key, position_embeddings, unsqueeze_dim=None)

        attention_scores = query @ key.transpose(-2, -1).unsqueeze(2)
        if self.causal:
            attention_scores = attention_scores.masked_fill(self.causal_mask[:token_len, :token_len].bool(), float('-inf'))

        attention_scores = (attention_scores / (attention_scores.shape[-1] ** 0.5)).softmax(dim=-1)
        attention_scores = self.dropout(attention_scores)

        attention_scores = attention_scores @ value.unsqueeze(2)
        attention_scores = attention_scores.permute(0, 3, 1, 2, 4).contiguous().view(batch_size, token_len, -1)
        attention_scores = self.out_proj(attention_scores)
        return attention_scores



if __name__ == "__main__":

    import time
    multi_head_attn = Attention(128, 256, causal=False, flash_attention=False, context_length=512)
    data = torch.randn(1, 11, 128)
    position_embedding = sinusoidal_positional_encoding(256, 11)
    time_start = time.time()
    res = multi_head_attn(data, position_embedding)
    end = time.time()
    print(f"Attention cost time: {round((end - time_start)*1000, 2)}ms")
    print(res.shape)

    multi_head_attn = MultiHeadAttentionByPspp(8,128, 256)
    data = torch.randn(1, 11, 128)
    position_embedding = sinusoidal_positional_encoding(256//8, 11)
    time_start = time.time()
    res = multi_head_attn(data, position_embedding)
    end = time.time()
    print(f"MultiHeadAttentionByPspp cost time: {round((end - time_start) * 1000, 2)}ms")
    print(res.shape)

    multi_head_attn = MultiQueryAttention(8, 128, 256)
    data = torch.randn(1, 11, 128)
    position_embedding = sinusoidal_positional_encoding(256//8, 11)
    time_start = time.time()
    res = multi_head_attn(data, position_embedding)
    end = time.time()
    print(f"MultiQueryAttention cost time: {round((end - time_start) * 1000, 2)}ms")
    print(res.shape)

    multi_head_attn = GroupQueryAttention(2, 8, 128, 256)
    data = torch.randn(1, 12, 128)
    position_embedding = sinusoidal_positional_encoding(256 // 8, 12)
    time_start = time.time()
    res = multi_head_attn(data, position_embedding)
    end = time.time()
    print(f"GroupQueryAttention cost time: {round((end - time_start) * 1000, 2)}ms")
    print(res.shape)

    multi_head_attn = MultiHeadLatentAttention(8, 128, 256, 64)
    data = torch.randn(1, 12, 128)
    time_start = time.time()
    res = multi_head_attn(data, use_cache=True)
    end = time.time()
    print(f"MultiHeadLatentAttention cost time: {round((end - time_start) * 1000, 2)}ms")
    print(res.shape)
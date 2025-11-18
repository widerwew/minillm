import torch
import torch.nn as nn

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

    def forward(self, x):
        # x.shape is batch, token, din
        batch_size, token_len, dim = x.shape
        query = self.querys(x)
        key = self.keys(x)
        value = self.values(x)

        attn_weights = query @ key.transpose(-2, -1)

        if self.causal:
            self.causal_mask.masked_fill(self.causal_mask==0, float('-inf'))
            attn_weights = attn_weights * self.causal_mask[:token_len, :token_len]

        attn_weights = (attn_weights / (attn_weights.shape[-1] ** 0.5)).softmax(dim=-1)
        attn_scores = attn_weights @ value
        return attn_scores

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


    def forward(self, x):
        batch_size, token_len, dim = x.shape
        query = self.querys(x)
        key = self.keys(x)
        value = self.values(x)

        # batch_size, token_len, dim ----> batch_size, heads, token_len, head_dim
        query = query.view(batch_size, token_len, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, token_len, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, token_len, self.heads, self.head_dim).transpose(1, 2)

        attn_weights = query @ key.transpose(3, 2)
        if self.causal:
            self.causal_mask.masked_fill(self.causal_mask==0, float('-inf'))
            attn_weights = attn_weights * self.causal_mask[:token_len, :token_len]

        attn_weights = (attn_weights / (attn_weights.shape[-1] ** 0.5)).softmax(dim=-1)
        attn_scores = attn_weights @ value
        attn_scores = attn_scores.transpose(2, 1).contiguous().view(batch_size, token_len, -1)
        attn_scores = self.out_proj(attn_scores)
        return attn_scores



if __name__ == "__main__":
    multi_head_attn = Attention(128, 256, causal=False, flash_attention=False, context_length=512)
    data = torch.randn(1, 11, 128)
    res = multi_head_attn(data)
    print(res.shape)
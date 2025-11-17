import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, din, dout, bias=False, causal=True, flash_attention=False):
        self.querys = nn.Linear(din, dout, bias=bias)
        self.keys = nn.Linear(din, dout, bias=bias)
        self.values = nn.Linear(din, dout, bias=bias)
        self.causal = causal
        self.flash_attention = flash_attention
        if self.causal:
            tril_mask = torch.triu(torch.ones(din, din), diagonal=0)
            self.register_buffer('causal_mask', tril_mask.masked_fill(tril_mask == 0, float('-inf')))

    def forward(self, x):
        # x.shape is batch, token, din
        query = self.querys(x)
        key = self.keys(x)
        value = self.values(x)

        attn_weights = query @ key.T
        if self.causal:
            attn_weights = attn_weights * self.causal_mask
            attn_weights = (attn_weights / (attn_weights.shape[-1] ** 0.5)).softmax(dim=-1)

        attn_scores = attn_weights @ value
        return attn_scores

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, din, dout, bias=False, causal=True, flash_attention=False):
        self.heads = heads
        self.din = din
        self.dout = dout
        self.causal = causal
        self.flash_attention = flash_attention
        self.bias = bias
        assert din % heads == 0
        self.head_dim = din // heads
        self.querys = nn.Linear(self.din, self.head_dim, bias=bias)
        self.keys = nn.Linear(self.din, self.head_dim, bias=bias)
        self.values = nn.Linear(self.din, self.head_dim, bias=bias)


    def forward(self, x):
        pass









if __name__ == "__main__":
    tril_mask = torch.tril(torch.ones(5, 5),diagonal=0)
    print(tril_mask)
    mask = tril_mask.masked_fill(tril_mask == 0, float("-inf"))
    data = torch.ones(5, 5)
    attn = data * mask
    attn_softmak = (attn / attn.shape[-1]**0.5).softmax(dim=-1)
    print("attn:", attn)
    print("attn_softmak:", attn_softmak)

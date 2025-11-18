import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.bias = nn.Parameter(torch.zeros(dim))
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).sum(-1, keepdim=True).sqrt()
        return x / (norm + self.eps) * self.weight
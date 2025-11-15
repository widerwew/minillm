import torch
import torch.nn as nn

class RMSNrom(nn.Module):
    def __init__(self, dim, shift=False, alpha=0.2):
        self.dim = dim
        self.shift = shift
        self.alpha = alpha
        self.paras = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        batch, seq, token, dim = x.shape
        x = x.view(batch, seq, -1)
        x_norm = x.norm(dim=-1, keepdim=True)
        x = x / x_norm
        if self.shift:
            x = self.alpha * x + self.paras
        return x
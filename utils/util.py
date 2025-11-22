import math

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


class FFN(nn.Module):
    def __init__(self, in_dim, hidden_dim, bias=False):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.fc3 = nn.Linear(hidden_dim, in_dim, bias=bias)

    def forward(self, x):
        gate = nn.functional.silu(self.fc1(x))
        return self.fc3(gate * self.fc2(x))

class MOEFFN(nn.Module):
    def __init__(self, expert_nums, experts_per_token, in_dim, hidden_dim, bias=False):
        super().__init__()
        self.expert_nums = expert_nums
        self.experts_per_token = experts_per_token
        self.experts = nn.ModuleList([FFN(in_dim, hidden_dim, bias=bias) for _ in range(self.expert_nums)])
        self.gate = nn.Linear(in_dim, self.expert_nums, bias=bias)

    def forward(self, x):
        batch_size, token_num, in_dim = x.shape
        scores = self.gate(x)
        topk_scores, topk_index= scores.topk(self.experts_per_token)
        out_moe = torch.zeros(batch_size * token_num, in_dim, device=x.device, dtype=x.dtype)
        topk_index = topk_index.reshape(batch_size * token_num, self.experts_per_token)
        topk_scores = topk_scores.reshape(batch_size * token_num, self.experts_per_token)

        #-----------------TODO负载均衡----------------
        #----1. 统计每个专家被选中的频率
        #----2. 统计每个专家的权重和
        #----3. 计算平均每个专家被选择的平均权重
        #-----------------结束-------------------
        x = x.view(batch_size * token_num, in_dim)
        expert_top_index = topk_index.unique()
        for expert_index in expert_top_index:
            mask = (topk_index == int(expert_index.item()))
            mask_scores = topk_scores[mask]
            mask_avaliable = mask.nonzero()
            mask_index = mask_avaliable[:, 0]
            expert_input = x[mask_index, :]
            value = self.experts[expert_index](expert_input)
            result = mask_scores.unsqueeze(1) * value
            out_moe[mask_index, :] += result

        out_moe = out_moe.view(batch_size, token_num, in_dim)
        return out_moe



if __name__=="__main__":
    x = torch.randn(1, 5, 256)
    moeffn = MOEFFN(8, 3, 256, 128)
    ffn = FFN(256, 128)
    res_ffn = ffn(x)
    print("ffn.shape:", res_ffn.shape)
    res = moeffn(x)
    print("moe_ffn.shape:", res.shape)

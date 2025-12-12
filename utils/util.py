import math
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eps = config.eps
        self.bias = nn.Parameter(torch.zeros(config.hidden_dim))
        self.weight = nn.Parameter(torch.ones(config.hidden_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class RMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.ones(config.hidden_dim))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).sqrt()
        return x / (norm + self.eps) * self.weight


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        # handle the config.intermediate
        if config.intermediate_size is not None:
            self.intermediate_size = config.intermediate_size
        else:
            if config.intermediate_ratio is not None:
                if config.intermediate_ratio < 2 or config.intermediate_ratio > 4:
                    raise ValueError("intermediate_ratio must be between 2 and 4")
                intermediate_size = int(config.hidden_dim * config.intermediate_ratio)
                config.intermediate_size = (intermediate_size // 64) * 64
                self.intermediate_size = config.intermediate_size
            else:
                assert config.intermediate_size is not None or config.intermediate_ratio is not None, "You must provide intermediate size or intermediate_ratio!"

        self.gate = nn.Linear(config.hidden_dim, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        gate = nn.functional.silu(self.gate(x))
        return self.dropout(self.down_proj(gate * self.up_proj(x)))

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
        topk_scores = topk_scores.view(batch_size, token_num, self.experts_per_token).softmax(dim=-1)
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

def sinusoidal_positional_encoding(dim, max_len, theta_para=10000, dtype=torch.float32):
    """
    :param dim: input dimension
    :param max_len:  input sequence max length
    :param theta:
    :return:
    positional encoding = [sin(t / (10000**(2i/d))) : cos(t / (10000 ** (2i/d))]
    """
    simu_positional_encoding = torch.zeros(max_len, dim, dtype=dtype)
    theta = 1 / torch.pow(theta_para, torch.arange(0, dim, 2, dtype=dtype) / dim)
    t = torch.arange(max_len, dtype=dtype).to(theta.device)
    theta_table = torch.outer(t, theta)
    simu_positional_encoding[:, 0::2] = torch.sin(theta_table)
    simu_positional_encoding[:, 1::2] = torch.cos(theta_table)
    return simu_positional_encoding

def apply_rope_position_encoding(x, simu_pos_embed, unsqueeze_dim=None):
    # batch, token_num, heads, head_dim
    print(x.shape, simu_pos_embed.shape)
    rope_pos_embeded = torch.zeros_like(x)
    if unsqueeze_dim != None:
        simu_pos_embed = simu_pos_embed.unsqueeze(unsqueeze_dim)
    cos_matrix = simu_pos_embed[..., 1::2]
    sin_matrix = simu_pos_embed[..., 0::2]

    x_odd = x[..., 0::2]
    x_even = x[..., 1::2]

    # 旋转操作
    rope_pos_embeded[..., 0::2] = x_odd * cos_matrix - x_even * sin_matrix
    rope_pos_embeded[..., 1::2] = x_odd * sin_matrix + x_even * cos_matrix

    return rope_pos_embeded


if __name__=="__main__":
    x = torch.randn(1, 5, 256)
    moeffn = MOEFFN(8, 3, 256, 128)
    ffn = FFN(256, 128)
    res_ffn = ffn(x)
    print("ffn.shape:", res_ffn.shape)
    res = moeffn(x)
    print("moe_ffn.shape:", res.shape)

    pos_emb = sinusoidal_positional_encoding(64, 128, dtype=torch.float32)
    print("pos_emb.shape:", pos_emb.shape)

    data = torch.randn(3, 5, 256)
    pos_embed = apply_rope_position_encoding(data, 10000)
    print("apply_rope pos_embed.shape:", pos_embed.shape)


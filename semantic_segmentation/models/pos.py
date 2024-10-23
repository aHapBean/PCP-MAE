import numpy as np 
import torch

def get_pos_embed(embed_dim, ipt_pos):
    """
    embed_dim: output dimension for each position
    ipt_pos: [B, G, 3], where 3 is (x, y, z)
    """
    B, G, _ = ipt_pos.size()
    assert embed_dim % 6 == 0
    omega = torch.arange(embed_dim // 6).float().to(ipt_pos.device) # NOTE
    # print("omega: ", omega.shape, " ", omega) # omega.shape 32 values: 0-31
    omega /= embed_dim / 6.
    # (0-31) / 32
    omega = 1. / 10000**omega  # (D/6,)
    rpe = []
    for i in range(_):
        pos_i = ipt_pos[:, :, i]    # (B, G)
        out = torch.einsum('bg, d->bgd', pos_i, omega)  # (B, G, D/6), outer product
        # 将第一个输入张量 pos_i 的形状为 (b, g) 的二维张量与第二个输入张量 omega 的形状为 (d,) 的一维张量进行乘法，得到一个形状为 (b, g, d) 的三维张量，并将其赋值给变量 out。
        emb_sin = torch.sin(out) # (M, D/6)
        emb_cos = torch.cos(out) # (M, D/6)
        rpe.append(emb_sin)
        rpe.append(emb_cos)
    # 每一个对应 D/6 + D/6
    return torch.cat(rpe, dim=-1)

if __name__ == '__main__':
    x = torch.randn([3, 4, 6])  # batch size: 3, num_group: 4, 
    embed_dim=384
    out = get_pos_embed(embed_dim, x)
    print(out.size(), out)  # 3, 4, embed_dim
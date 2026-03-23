import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath
from einops.einops import rearrange

from model.block.gcn_conv import GCN_module
from model.block.transformer import LimbFormer_module
from model.block.pose_ssm import HipSSM_module
from mamba_ssm import Mamba


class TriadFormer(nn.Module):
    def __init__(self, args, depth=3, embed_dim=160, mlp_hidden_dim=1024, h=8, drop_rate=0.1, length=9):
        super().__init__()

        depth, embed_dim, mlp_hidden_dim, length = args.layers, args.channel, args.d_hid, args.frames
        self.num_joints_in, self.num_joints_out = args.n_joints, args.out_joints

        drop_path_rate = 0.3
        attn_drop_rate = 0.
        qkv_bias = True
        qk_scale = None

        self.patch_embed = nn.Linear(2, embed_dim)

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        dpr = [x.item() for x in torch.linspace(0.1, drop_path_rate, depth)]
        # dpr = [x.item() for x in torch.linspace(0.1, 0.2, depth)]

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=h, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, length=length)
            for i in range(depth)])
        self.Temporal_norm = norm_layer(embed_dim)
        self.fcn = nn.Linear(embed_dim, 3)

    def forward(self, x):
        x = rearrange(x, 'b f j c -> (b f) j c').contiguous()
        x = self.patch_embed(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.Temporal_norm(x)
        x = self.fcn(x)

        x = x.view(x.shape[0], -1, self.num_joints_out, x.shape[2])
        return x

# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from LayoutDiffusion
#   (https://github.com/ZGCTroy/LayoutDiffusion)
# Copyright (c) 2023 LayoutDiffusion authors, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

from abc import abstractmethod
import os
import safetensors
import math
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F


from .nn import (
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from diffusers.utils.constants import SAFETENSORS_WEIGHTS_NAME
from bevdepth.projects.ldm.modules.attention import SpatialTransformer
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version
from .multiscale_fusion import *
from bevdepth.projects.seg_lss_emb import SegBEVLSS

def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()

class SiLU(nn.Module):  # export-friendly version of SiLU()
    @staticmethod
    def forward(x):
        return x * th.sigmoid(x)

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, dino_cond=None, seg_maps=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                if isinstance(layer, SPADEResBlock):
                    x = layer(x, emb, seg_maps)
                elif isinstance(layer, ResBlock):
                    x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, dino_cond)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, out_size=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.out_size = out_size
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            if self.out_size is None:
                x = F.interpolate(
                    x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
                )
            else:
                x = F.interpolate(
                    x, (x.shape[2], self.out_size, self.out_size), mode="nearest"
                )
        else:
            if self.out_size is None:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
            else:
                x = F.interpolate(x, size=self.out_size, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            use_checkpoint=False,
            up=False,
            down=False,
            out_size=None,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims, out_size=out_size)
            self.x_upd = Upsample(channels, False, dims, out_size=out_size)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class FDN(nn.Module):
    def __init__(self, norm_channels, cond_channels):
        super().__init__()
        self.param_free_norm = normalization(norm_channels)
        self.conv_gamma = nn.Conv2d(cond_channels, norm_channels, kernel_size=3, padding=1)
        self.conv_beta  = nn.Conv2d(cond_channels, norm_channels, kernel_size=3, padding=1)

    # def _select_seg_cond(self, seg_bev_dict, target_size):
    #     for ds, seg_feat in seg_bev_dict.items():
    #         if seg_feat.size()[2:] == target_size:
    #             return seg_feat

    def forward(self, x, seg_cond):
        # seg_cond = self._select_seg_cond(seg_bev_dict, x.size()[2:])
        assert seg_cond.size()[2:] == x.size()[2:]
        normalized = self.param_free_norm(x)
        gamma = self.conv_gamma(seg_cond)
        beta = self.conv_beta(seg_cond)
        out = normalized * (1 + gamma) + beta
        return out

class SPADEResBlock(TimestepBlock):
    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=False,
            dims=2,
            use_checkpoint=False,
            up=False,
            down=False,
            out_size=None,
            seg_channels=None
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint

        self.norm_0 = FDN(channels, seg_channels)
        self.norm_1 = FDN(self.out_channels, seg_channels)

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims, out_size=out_size)
            self.x_upd = Upsample(channels, False, dims, out_size=out_size)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.in_layers = nn.Sequential(
            nn.Identity(),  
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                self.out_channels),
        )
        
        self.out_layers = nn.Sequential(
            nn.Identity(),  
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )
        
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb, seg_bev):
        h = self.norm_0(x, seg_bev)
        h = self.in_layers(h)

        # time embedding
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        h = h + emb_out
        h = self.norm_1(h, seg_bev)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            use_checkpoint=False,
            encoder_channels=None,
            return_attention_embeddings=False,
            ds=None,
            resolution=None,
            type=None,
            use_positional_embedding=False
    ):
        super().__init__()
        self.type = type # type: ignore
        self.ds = ds
        self.resolution = resolution
        self.return_attention_embeddings = return_attention_embeddings

        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                    channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.use_positional_embedding = use_positional_embedding
        if self.use_positional_embedding:
            self.positional_embedding = nn.Parameter(th.randn(channels // self.num_heads, resolution ** 2) / channels ** 0.5)  # [C,L1] # type: ignore
        else:
            self.positional_embedding = None

        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)

        self.qkv = conv_nd(1, channels, channels * 3, 1)

        self.attention = QKVAttentionLegacy(self.num_heads)

        self.encoder_channels = encoder_channels
        if encoder_channels is not None:
            self.encoder_kv = conv_nd(1, encoder_channels, channels * 2, 1)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, cond_kwargs=None):
        '''
        :param x: (N, C, H, W)
        :param cond_kwargs['xf_out']: (N, C, L2)
        :return:
            extra_output: N x L2 x 3 x ds x ds
        '''
        extra_output = None
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)  # N x C x (HxW)

        qkv = self.qkv(self.norm(x))  # N x 3C x L1, 其中L1=H*W
        if cond_kwargs is not None and self.encoder_channels is not None:
            kv_for_encoder_out = self.encoder_kv(cond_kwargs['xf_out'])  # xf_out: (N x encoder_channels x L2) -> (N x 2C x L2), 其中L2=max_obj_num
            h = self.attention(qkv, kv_for_encoder_out, positional_embedding=self.positional_embedding)
        else:
            h = self.attention(qkv, positional_embedding=self.positional_embedding)
        h = self.proj_out(h)
        output = (x + h).reshape(b, c, *spatial)

        if self.return_attention_embeddings:
            assert cond_kwargs is not None
            if extra_output is None:
                extra_output = {}
            extra_output.update({
                'type': self.type,
                'ds': self.ds,
                'resolution': self.resolution,
                'num_heads': self.num_heads,
                'num_channels': self.channels,
                'image_query_embeddings': qkv[:, :self.channels, :].detach(),  # N x C x L1
            })
            if cond_kwargs is not None:
                extra_output.update({
                    'layout_key_embeddings': kv_for_encoder_out[:, : self.channels, :].detach()  # N x C x L2
                })

        return output, extra_output


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, encoder_kv=None, positional_embedding=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Q_T, K_T, and V_T.
        :param encoder_kv: an [N x (H * 2 * C) x S] tensor of K_E, and V_E.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)

        if positional_embedding is not None:
            q = q + positional_embedding[None, :, :].to(q.dtype)  # [N, C, T]
            k = k + positional_embedding[None, :, :].to(q.dtype)  # [N, C, T]

        if encoder_kv is not None:
            assert encoder_kv.shape[1] == self.n_heads * ch * 2
            ek, ev = encoder_kv.reshape(bs * self.n_heads, ch * 2, -1).split(ch, dim=1)
            k = th.cat([ek, k], dim=-1)
            v = th.cat([ev, v], dim=-1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class DINOContextAdapter(nn.Module):
    def __init__(self, 
                 c_in=768, 
                 c_emb=1024,
                 num_views=6,
                 dropout=0.0,
                 ln_first=True,
                 ln_after=True,
                 use_cam_embed=True,
                 temperature=1.0):
        super().__init__()
        self.num_views = num_views
        self.use_cam_embed = use_cam_embed
        self.temperature = temperature

        self.ln_first = nn.LayerNorm(c_in) if ln_first else None

        # camera embedding (learned)
        if use_cam_embed:
            self.cam_embed = nn.Embedding(num_views, c_in)
            nn.init.normal_(self.cam_embed.weight, std=0.02)
        else:
            self.register_parameter("cam_embed", None)

        # optional per-view bias (helps learning)
        self.view_bias = nn.Parameter(th.zeros(num_views))

        # projection to emb dim
        self.proj = nn.Sequential(
            nn.Linear(c_in, c_emb),
            nn.GELU(),
            nn.Linear(c_emb, c_emb),
        )
  
        self.ln_after = nn.LayerNorm(c_emb) if ln_after else None

    def forward(self, context, cam_ids=None):
        """
        context: (B, V, C_in)  or (B, C_in) -> treated as V=1
        cam_ids: (B, V) long indices in [0, num_views-1] (optional; if None, 0..V-1)
        """
        if context.ndim == 2:
            context = context.unsqueeze(1)  # (B,1,C)
        elif context.ndim != 3:
            raise ValueError(f"context must be (B,C) or (B,V,C), got {context.shape}")

        B, V, C = context.shape
        x = context
        if self.ln_first is not None:
            x = self.ln_first(x)  # LN over C

        # ----- view weighting -----
        if self.use_cam_embed:
            cam_embed = self.cam_embed(cam_ids)               # (B, V, C)
            # dot-product score with temperature and per-view bias
            logits = (x * cam_embed).sum(dim=-1) / (C ** 0.5) # (B, V)
        else:
            # no cam embedding: fall back to a learned per-view bias only
            logits = th.zeros(B, V, device=x.device)

        # add learned per-view bias 
        bias = self.view_bias[:V].unsqueeze(0)            # (1, V)
        logits = (logits + bias) / max(self.temperature, 1e-6)

        w = F.softmax(logits, dim=1)                      # (B, V)
        w = w.unsqueeze(-1)                               # (B, V, 1)

        # weighted sum over views
        g = (w * x).sum(dim=1)                            # (B, C_in)

        g = self.proj(g)                                  # (B, C_emb)
        if self.ln_after is not None:
            g = self.ln_after(g)
        return g


class DINOBevAligner(nn.Module):
    """
    Self-contained BEV aligner for DINOv2 last_tokens using BEVFormer-style reference generation.

    Inputs:
      - last_tokens: (B, V, N, C_dino)
      - patch_hw:    (Hp, Wp) with Hp*Wp == N
      - img_metas:   list of dicts (len=B), each with:
          * 'lidar2img': (V, 4, 4)
          * 'img_shape' : ((H, W, 3),) or similar; we use [0][0]=H, [0][1]=W

    Returns:
      - bev_feat_ctx: (B, C_ctx, bev_h, bev_w)
    """
    def __init__(
        self,
        bev_h=50,
        bev_w=50,
        cam_view=6,
        pc_range = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        num_points_in_pillar=4,
        input_size=518,             # DINO square resize S
        c_dino=768,                 # DINO feature dim
        c_ctx=None,                  # output channels
        post_ln_affine=True,       # recommended True (stability + capacity)
        eps=1e-6,
        device='cuda'
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        self.num_points_in_pillar = num_points_in_pillar
        self.c_dino = c_dino
        self.c_ctx = c_dino if c_ctx is None else c_ctx
        self.eps = eps

        # Norms are created lazily with correct feature dim
        self.post_ln_affine = post_ln_affine
        # [Improvement 3] Pre-LN enabled for feature normalization before grid sampling
        self.pre_ln = nn.LayerNorm(self.c_dino, elementwise_affine=True).to(device)
        self.post_ln = nn.LayerNorm(self.c_dino, elementwise_affine=self.post_ln_affine).to(device)

        # Per-view weights (initialized lazily with V)
        self._w_view = nn.Parameter(th.zeros(1, cam_view, 1, device=device))

        # [Improvement 4] MLP projection instead of simple linear
        # (B,Q,C_dino) -> (B,Q,C_ctx)
        self.proj = nn.Sequential(
            nn.Linear(self.c_dino, self.c_ctx),
            nn.GELU(),
            nn.Linear(self.c_ctx, self.c_ctx),
        )

        # [Improvement 1] Learnable 2D positional embedding for BEV grid
        self.bev_pos_embed = nn.Parameter(th.zeros(1, self.c_ctx, bev_h, bev_w))
        nn.init.trunc_normal_(self.bev_pos_embed, std=0.02)

    # ---------- BEVFormer-style reference generation ----------
    @staticmethod
    def _get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=th.float32):
        if dim == '3d':
            zs = th.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype, device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = th.linspace(0.5, W - 0.5, W, dtype=dtype, device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = th.linspace(0.5, H - 0.5, H, dtype=dtype, device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = th.stack((xs, ys, zs), -1)                 # (D,H,W,3)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)  # (D, H*W, 3)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)              # (bs, D, H*W, 3)
            return ref_3d
        elif dim == '2d':
            ref_y, ref_x = th.meshgrid(
                th.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                th.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
                indexing='ij'
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = th.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)          # (bs, H*W, 1, 2)
            return ref_2d
        else:
            raise ValueError("dim must be '3d' or '2d'")

    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, img_metas):
        allow_tf32 = th.backends.cuda.matmul.allow_tf32
        allow_tf32_cudnn  = th.backends.cudnn.allow_tf32
        th.backends.cuda.matmul.allow_tf32 = False
        th.backends.cudnn.allow_tf32 = False

        # (B, N, 4, 4)
        lidar2img = np.asarray([m['lidar2img'] for m in img_metas])
        lidar2img = reference_points.new_tensor(lidar2img)

        pc_range = self.pc_range
        ref = reference_points.clone()
        ref[..., 0:1] = ref[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        ref[..., 1:2] = ref[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        ref[..., 2:3] = ref[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        ref = th.cat((ref, th.ones_like(ref[..., :1])), -1)            # (bs, D, Q, 4)

        ref = ref.permute(1, 0, 2, 3)                                  # (D, B, Q, 4)
        D, B, Q = ref.size()[:3]
        num_cam = lidar2img.size(1)

        ref = ref.view(D, B, 1, Q, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)  # (D,B,N,Q,4,1)
        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, Q, 1, 1)

        cam = th.matmul(lidar2img.to(th.float32), ref.to(th.float32)).squeeze(-1)  # (D,B,N,Q,4)
        eps = 1e-5
        depth = cam[..., 2:3]
        bev_mask = (depth > eps)                                                   # (D,B,N,Q,1)

        uv = cam[..., 0:2] / th.maximum(depth, th.ones_like(depth) * eps)          # (D,B,N,Q,2)

        # (V,B,Q,D,2), (V,B,Q,D)
        uv = uv.permute(2, 1, 3, 0, 4).contiguous()           
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1).contiguous()       

        th.backends.cuda.matmul.allow_tf32 = allow_tf32
        th.backends.cudnn.allow_tf32 = allow_tf32_cudnn
        return uv, bev_mask

    def _tokens_to_fmap(self, last_tokens, Hp, Wp):
        B, V, N, C = last_tokens.shape
        fmap = last_tokens.view(B, V, Hp, Wp, C).permute(0,1,4,2,3).contiguous()  # (B,V,C,Hp,Wp)
        if self.pre_ln is not None:
            t = fmap.permute(0,1,3,4,2).reshape(-1, C)  # (B*V*Hp*Wp, C)
            t = self.pre_ln(t)
            fmap = t.view(B, V, Hp, Wp, C).permute(0,1,4,2,3).contiguous()
        return fmap

    def forward(self, last_tokens, patch_hw, img_metas, dino_geom):
        """
        last_tokens: (B,V,N,C_dino)
        patch_hw:    (Hp,Wp)
        img_metas:   list length B (BEVFormer-like metas)
        returns:     (B, C_ctx, bev_h, bev_w)
        """
        Hp, Wp = patch_hw
        B, V, C_dino, _, _ = last_tokens.shape
        last_tokens = last_tokens.reshape(-1, V, Hp*Wp, C_dino).contiguous()  # (B,V,N,C_dino)
        assert last_tokens.ndim == 4
        B, V, N, C_dino = last_tokens.shape
        device = last_tokens.device
        patch_size = dino_geom.get('patch_size', None)

        # (1) DINO fmap
        fmap = self._tokens_to_fmap(last_tokens, Hp, Wp)  # (B, V, C_dino, Hp, Wp)

        # (2) BEV refs and camera projection
        Z_bins = int(round((self.pc_range[5] - self.pc_range[2]) ))  # same spirit as BEVFormer
        ref_3d = self._get_reference_points(self.bev_h, self.bev_w, Z=Z_bins,
                                            num_points_in_pillar=self.num_points_in_pillar,
                                            dim='3d', bs=B, device=device, dtype=fmap.dtype)
        uv, bev_mask = self.point_sampling(ref_3d, img_metas)  # (V, B, Q, D, 2), (V,B,Q,D)

        # (3) uv coords -> patch grid coords
        Q = self.bev_h * self.bev_w
        scale = dino_geom['scale']
        pad_top, pad_left = dino_geom['padding'][0], dino_geom['padding'][1]
        H2, W2 = dino_geom['H2W2'][0], dino_geom['H2W2'][1]
    
        u = uv[..., 0]  # (V,B,Q,D)
        v = uv[..., 1]  # (V,B,Q,D)

        # DINO input pixel coords
        u_d = u * scale + pad_left   #(V,B,Q,D)
        v_d = v * scale + pad_top

        valid_in = (u_d >= 0) & (u_d <= (W2 - 1)) & (v_d >= 0) & (v_d <= (H2 - 1))
        mask_bv = bev_mask & valid_in  # (V,B,Q,D

        # normalization
        u_p = u_d / patch_size  
        v_p = v_d / patch_size  
        gx = 2.0 * (u_p / (W2 - 1.0)) - 1.0  # (V,B,Q,D)
        gy = 2.0 * (v_p / (H2 - 1.0)) - 1.0
        grid = th.stack([gx, gy], dim=-1)  # (V,B,Q,D,2)

        # (4) bilinear sampling
        fmap_v = fmap.view(B*V, C_dino, Hp, Wp)
        grid_v = grid.permute(1,0,2,3,4).contiguous().view(B*V, Q*self.num_points_in_pillar, 1, 2)
        # sampled = F.grid_sample(fmap_v, grid_v, mode='bilinear',
        #                         padding_mode='zeros', align_corners=True)  # (B*V,C,Q*D,1)
        sampled = F.grid_sample(fmap_v, grid_v, mode='bilinear',
                                padding_mode='border', align_corners=False)  # (B*V,C,Q*D,1)
        sampled = sampled.squeeze(-1).permute(0,2,1).contiguous().view(B, V, Q, self.num_points_in_pillar, C_dino)

        # (5) post-norm
        t = sampled.view(-1, C_dino)
        t = self.post_ln(t)
        sampled = t.view(B, V, Q, self.num_points_in_pillar, C_dino)

        # (6) pillar mean + view-weighted mean
        mask = mask_bv.to(device).permute(1,0,2,3).unsqueeze(-1).float()  # (B,V,Q,D,1)
        sampled = sampled * mask
        denom_D = mask.sum(dim=3, keepdim=True).clamp_min(self.eps)        # (B,V,Q,1,1)
        feat_v = sampled.sum(dim=3, keepdim=True) / denom_D                # (B,V,Q,1,C)
        feat_v = feat_v.squeeze(3)                                         # (B,V,Q,C)
        
        # view-weighted mean
        w = F.softplus(self._w_view).expand(B, -1, -1)                         
        w = w.unsqueeze(-1)                                                # (B, V, 1, 1)    
        view_valid = (denom_D.squeeze(3) > 0).float()                                     
        num = (feat_v * w).sum(dim=1)                                      # (B,Q,C)
        den = (w * view_valid).sum(dim=1).clamp_min(self.eps)              # (B,Q,1)
        f_bev = num / den                                                  # (B,Q,C_dino)

        # (7) channel reduction (B,Q,C_dino) -> (B,Q,C_ctx)
        bev_qc = self.proj(f_bev)                                       # (B,Q,C_ctx)

        # reshape to (B,C_ctx,H,W)
        bev_feat_ctx = bev_qc.permute(0,2,1).contiguous().view(B, self.c_ctx, self.bev_h, self.bev_w)
        
        # [Improvement 1] Add BEV positional encoding
        bev_feat_ctx = bev_feat_ctx + self.bev_pos_embed
        
        return bev_feat_ctx

class Mlp(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, act_layer=nn.ReLU):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
    
class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)
    
class CamAwareDINO(nn.Module):
    """
    Apply per-camera (per-view) camera-aware SE to DINO features.
    """
    def __init__(self, in_channels, out_channels, cam_vec_dim=27, mlp_hidden=None):
        super().__init__()
        hidden = in_channels if mlp_hidden is None else mlp_hidden
        self.bn = nn.BatchNorm1d(cam_vec_dim)
        self.mlp = Mlp(cam_vec_dim, hidden, in_channels)
        self.se = SELayer(in_channels)
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, mats_dict):
        B, V, C, H, W = x.shape
        intrins = mats_dict['intrin_mats'][:, 0:1, ..., :3, :3]
        batch_size = intrins.shape[0]
        num_cams = intrins.shape[2]
        ida = mats_dict['ida_mats'][:, 0:1, ...]
        sensor2ego = mats_dict['sensor2ego_mats'][:, 0:1, ..., :3, :]
        bda = mats_dict['bda_mat'].view(batch_size, 1, 1, 4,
                                        4).repeat(1, 1, num_cams, 1, 1)
        cam_vec = torch.cat(
            [
                torch.stack(
                    [
                        intrins[:, 0:1, ..., 0, 0],
                        intrins[:, 0:1, ..., 1, 1],
                        intrins[:, 0:1, ..., 0, 2],
                        intrins[:, 0:1, ..., 1, 2],
                        ida[:, 0:1, ..., 0, 0],
                        ida[:, 0:1, ..., 0, 1],
                        ida[:, 0:1, ..., 0, 3],
                        ida[:, 0:1, ..., 1, 0],
                        ida[:, 0:1, ..., 1, 1],
                        ida[:, 0:1, ..., 1, 3],
                        bda[:, 0:1, ..., 0, 0],
                        bda[:, 0:1, ..., 0, 1],
                        bda[:, 0:1, ..., 1, 0],
                        bda[:, 0:1, ..., 1, 1],
                        bda[:, 0:1, ..., 2, 2],
                    ],
                    dim=-1,
                ),
                sensor2ego.view(batch_size, 1, num_cams, -1),
            ],
            -1,
        )
        # --- BN/MLP per (B*V) ---
        cam_vec_flat = cam_vec.reshape(B * V, -1)                  # (B*V,27)
        cam_vec_flat = self.bn(cam_vec_flat)                       # (B*V,27)
        cam_emb = self.mlp(cam_vec_flat).view(B * V, C, 1, 1)      # (B*V,C,1,1)
        # --- SE per (B*V) ---
        x = x.reshape(B * V, C, H, W)
        x = self.se(x, cam_emb)
        x_out = self.proj(x)                    
        return x_out.view(B, V, -1, H, W)    # (B,V,C_out,H,W)


class DiffusionUNetModel(nn.Module):
    """
    A UNetModel that conditions on layout with an encoding transformer.
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_ds: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.

    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param {
        layout_length: number of layout objects to expect.
        hidden_dim: width of the transformer.
        num_layers: depth of the transformer.
        num_heads: heads in the transformer.
        xf_final_ln: use a LayerNorm after the output layer.
        num_classes_for_layout_object: num of classes for layout object.
        mask_size_for_layout_object: mask size for layout object image.
    }

    """

    def __init__(
            self,
            # layout_encoder,
            in_channels,
            model_channels,
            out_channels,
            seg_channels,
            num_res_blocks,
            attention_ds,
            dino_dim=768,
            context_dim=256, 
            resize_dim=(252, 700),
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_spatial_transformer=True,
            image_size=256,
            num_attention_blocks=1,
            use_key_padding_mask=False,
            norm_first=False,
            num_pre_downsample=0,
            transformer_depth=1,
            return_multiscale=True,
            multiscale_indices='auto',
            legacy=True,
            seg_bev_lss=None,
    ):
        super().__init__()

        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        # self.norm_for_obj_embedding = norm_for_obj_embedding
        # self.channels_scale_for_positional_embedding = channels_scale_for_positional_embedding
        self.norm_first = norm_first
        self.use_key_padding_mask=use_key_padding_mask
        self.num_attention_blocks = num_attention_blocks
        self.image_size = image_size
        # self.use_positional_embedding_for_attention = use_positional_embedding_for_attention

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.resize_dim = resize_dim
        self.num_res_blocks = num_res_blocks
        self.attention_ds = attention_ds
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        # multi-scale features index
        self.return_multiscale = return_multiscale

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.downsample_blocks = nn.ModuleList([])
        self.upsample_blocks = nn.ModuleList([])
        for _ in range(num_pre_downsample):
            self.downsample_blocks.append(Downsample(
                            in_channels, conv_resample, dims=dims, out_channels=in_channels
                        ))
            self.upsample_blocks.append(Upsample(
                            out_channels, conv_resample, dims=dims, out_channels=out_channels
                        ))
            self.image_size = self.image_size // 2  

        ## DINO feature condition
        # self.adapter = DINOContextAdapter(c_in=dino_dim, c_emb=time_embed_dim, pool='mean')
        self.adapter = DINOContextAdapter(c_in=dino_dim, c_emb=time_embed_dim, num_views=6)
        self.cam_se = CamAwareDINO(in_channels=dino_dim, out_channels=dino_dim//2,).cuda()
        self.aligner = DINOBevAligner(bev_h=self.image_size, bev_w=self.image_size, c_dino=dino_dim//2, c_ctx=context_dim)

        ## Grounded SAM segmentation map
        self.seg_bev_aligner = SegBEVLSS(**seg_bev_lss)

        if self.return_multiscale:
            self.multi_concat = MultiScaleConcatWeighted(in_chs=(model_channels, model_channels*2, model_channels*4, model_channels*4), 
                                                            out_dim=out_channels, 
                                                            mid=model_channels)
        
        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_ds:
                    print('encoder attention layer: ds = {}, resolution = {}'.format(ds, self.image_size // ds))
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    for _ in range(self.num_attention_blocks):
                        layers.append(
                            SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim)
                        )
                # self.input_blocks.append(TimestepEmbedSequential(*layers))
                block = TimestepEmbedSequential(*layers)
                block.ctx_ds = ds    
                self.input_blocks.append(block)
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                block = TimestepEmbedSequential(
                            ResBlock(
                                ch,
                                time_embed_dim,
                                dropout,
                                out_channels=out_ch,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                down=True,
                            )
                            # SPADEResBlock(
                            #     ch,
                            #     time_embed_dim,
                            #     dropout,
                            #     out_channels=out_ch,
                            #     dims=dims,
                            #     use_checkpoint=use_checkpoint,
                            #     use_scale_shift_norm=use_scale_shift_norm,
                            #     down=True,
                            # )
                            if resblock_updown
                            else Downsample(
                                ch, conv_resample, dims=dims, out_channels=out_ch
                            )
                        )
                block.ctx_ds = ds  
                self.input_blocks.append(block)
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        print('middle attention layer: ds = {}, resolution = {}'.format(ds, self.image_size // ds))
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            SpatialTransformer(
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
            ),                             
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block.ctx_ds = ds
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    SPADEResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        seg_channels=seg_channels[level],
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_ds:
                    print('decoder attention layer: ds = {}, resolution = {}'.format(ds, self.image_size // ds))
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    for _ in range(self.num_attention_blocks):
                        layers.append(
                            SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                            )
                        )
                      
                current_ds = ds  # SPADEResBlock이 처리하는 현재 해상도 저장
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            out_size=int(self.image_size // (ds // 2))
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch, out_size=int(self.image_size // ds))
                    )
                    ds //= 2
                # self.output_blocks.append(TimestepEmbedSequential(*layers))
                block = TimestepEmbedSequential(*layers)
                block.ctx_ds = current_ds  # upsample 전 ds 값 사용 
                self.output_blocks.append(block)
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )
        self.use_fp16 = use_fp16

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)
        # self.layout_encoder.convert_to_fp16() 

    def forward(self, x, timesteps, mats_dict, dino_cond, segmaps, depth):
        hs, extra_outputs = [], []

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        B, V, _ = dino_cond['last_cls'][:, 0, ...].shape
        cam_ids = th.arange(V, dtype=th.long, device=emb.device).unsqueeze(0).expand(B, V)
        dino_cond_proj = self.adapter(dino_cond['last_cls'][:, 0, ...], cam_ids=cam_ids)

        emb = emb + dino_cond_proj.to(emb)  # emb: (B, 1024)

        # DINOv2 condition
        cam_se_dino = self.cam_se(dino_cond['last_tokens'][:, 0, ...], mats_dict)  # (B, V, C, H, W)
        bev_ctx = self.aligner(cam_se_dino, patch_hw=dino_cond['patch_hw'], 
                               img_metas=dino_cond['img_metas'], dino_geom=dino_cond['geom'])   # (B,256,50,50)
        bev_ctx_transform = bev_ctx.flip(-1).transpose(-1, -2).contiguous()
        
        tokens_by_ds = {}
        for ds_key in self.attention_ds[::-1]:
            target_hw = int(self.image_size // ds_key)  
            tokens_by_ds[ds_key] = self._ctx_tokens_from_bev(bev_ctx_transform, target_hw)  # (B, target_hw*target_hw, 256)

        # Grounded SAM condition
        seg_bev_maps = self.seg_bev_aligner(segmaps, depth[:, 0, ...], mats_dict) # (B, num_class, H, W)

        out_list = []
        h = x.type(self.dtype)  # h: (B, C, H, W)
        for module in self.downsample_blocks:
            h = module(h) 
        # Encoder
        for module in self.input_blocks:
            dino_tokens = self._select_ctx(tokens_by_ds, module)
            seg_bev = self._select_ctx(seg_bev_maps, module)
            h = module(h, emb, dino_tokens, seg_bev) 
            hs.append(h)
        
        # Middle block
        dino_tokens_mid = self._select_ctx(tokens_by_ds, self.middle_block)
        seg_bev_mid = self._select_ctx(seg_bev_maps, self.middle_block)
        h = self.middle_block(h, emb, dino_tokens_mid, seg_bev_mid)
            
        # Decoder
        for i_out, module in enumerate(self.output_blocks):
            h = th.cat([h, hs.pop()], dim=1)
            dino_tokens = self._select_ctx(tokens_by_ds, module)
            seg_bev = self._select_ctx(seg_bev_maps, module)
            h = module(h, emb, dino_tokens, seg_bev)
            # if i_out in [1, 4]:
            #     out_list.append(h)
            
        h = h.type(x.dtype)
        h = self.out(h)

        # out_list.append(final)
        
        for module in self.upsample_blocks:
            h = module(h)
            
        if self.return_multiscale:
            multi_feat = self.multi_concat(out_list[::-1]) 
            return h, multi_feat, out_list
        else:
            return [h]

    def save_pretrained(self, save_directory):
        if os.path.isfile(save_directory):
            print(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)
        weights_name = SAFETENSORS_WEIGHTS_NAME
        safetensors.torch.save_file(self.state_dict(), os.path.join(save_directory, weights_name), metadata={"format": "pt"})
        
    def from_pretrained(self, pretrained_model_name_or_path, subfolder=None):
        weights_name = SAFETENSORS_WEIGHTS_NAME
        if os.path.isfile(pretrained_model_name_or_path):
            checkpoint_file = pretrained_model_name_or_path
        elif os.path.isdir(pretrained_model_name_or_path):
            if os.path.isfile(os.path.join(pretrained_model_name_or_path, weights_name)):
                checkpoint_file = os.path.join(pretrained_model_name_or_path, weights_name)
            elif subfolder is not None and os.path.isfile(
            os.path.join(pretrained_model_name_or_path, subfolder, weights_name)):
                checkpoint_file = os.path.join(pretrained_model_name_or_path, subfolder, weights_name)
        else:
            print(f"Error no file named {weights_name} found in directory {pretrained_model_name_or_path}.")
            return
        state_dict = safetensors.torch.load_file(checkpoint_file, device="cpu")
        try:
            self.load_state_dict(state_dict, strict=True)
            print('successfully load the entire model')
        except:
            print('not successfully load the entire model, try to load part of model')
            self.load_state_dict(state_dict, strict=False)


    def _ctx_tokens_from_bev(self, bev_ctx: th.Tensor, size_hw: int):
        """
        bev_ctx: (B, 256, H, W)
        size_hw: (int) 50, 25, 12
        return:  (B, size_hw*size_hw, 256) -> self.dtype
        """
        if bev_ctx.shape[-1] != size_hw or bev_ctx.shape[-2] != size_hw:
            bev_ctx_s = F.adaptive_avg_pool2d(bev_ctx, (size_hw, size_hw))
        else:
            bev_ctx_s = bev_ctx
        return bev_ctx_s.flatten(2).transpose(1, 2).contiguous().to(self.dtype)


    def _select_ctx(self, tokens_by_ds: dict, module: nn.Module):
        ds = getattr(module, "ctx_ds", 1)
        if ds in tokens_by_ds:
            return tokens_by_ds[ds]
        nearest = min(tokens_by_ds.keys(), key=lambda k: abs(k - ds))
        return tokens_by_ds[nearest]
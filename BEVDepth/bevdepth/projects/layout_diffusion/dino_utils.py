import math
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Utilities (as requested)
# =========================

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)

def avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def update_ema(target_params, source_params, rate=0.99):
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

def scale_module(module, scale):
    for p in module.parameters():
        p.detach().mul_(scale)
    return module

def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def normalization(channels):
    return GroupNorm32(32, channels)


# =========================
# Helper functions
# =========================

def build_bev_xy_grid(B: int, H: int, W: int, pc_range, device):
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    xs = torch.linspace(x_min, x_max, W, device=device)
    ys = torch.linspace(y_min, y_max, H, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")          # (H, W)
    x = xx.unsqueeze(0).expand(B, -1, -1).contiguous()      # (B,H,W)
    y = yy.unsqueeze(0).expand(B, -1, -1).contiguous()      # (B,H,W)
    return x, y

def project_lidar2img(points_xyz: torch.Tensor, lidar2img: torch.Tensor, img_hw: Tuple[int,int]):
    B, V, N, _ = points_xyz.shape
    ones = torch.ones((B, V, N, 1), device=points_xyz.device, dtype=points_xyz.dtype)
    xyz1 = torch.cat([points_xyz, ones], dim=-1)                              # (B,V,N,4)
    cam = torch.einsum('bvij,bvnj->bvni', lidar2img, xyz1)                    # (B,V,N,4)

    z = cam[..., 2].clamp_min(1e-6)
    u = cam[..., 0] / z
    v = cam[..., 1] / z
    H_img, W_img = img_hw
    mask = (z > 0) & (u >= 0) & (u <= W_img - 1) & (v >= 0) & (v <= H_img - 1)
    uv = torch.stack([u, v], dim=-1)
    return uv, mask, z


def uv_to_patch_grid(uv, img_hw, patch_hw):
    """
    uv: (B,V,N,2) pixel coords -> grid_sample grid: (B*V, N, 1, 2) in [-1,1]
    """
    H_img, W_img = img_hw
    Hp, Wp = patch_hw
    up = uv[..., 0] / (W_img / float(Wp))
    vp = uv[..., 1] / (H_img / float(Hp))
    # center aligned
    x = 2.0 * ((up + 0.5) / float(Wp)) - 1.0
    y = 2.0 * ((vp + 0.5) / float(Hp)) - 1.0
    grid = torch.stack([x, y], dim=-1)                                     # (B,V,N,2)
    B, V, N, _ = grid.shape
    return grid.view(B * V, N, 1, 2).contiguous()

def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int, eps: float = 1e-6):
    """
    x: (..., L, C) or (..., L)
    mask: (..., L) boolean
    """
    m = mask.float()
    while m.ndim < x.ndim:
        m = m.unsqueeze(-1)
    num = (x * m).sum(dim=dim)
    den = m.sum(dim=dim).clamp_min(eps)
    return num / den


# ==========================================
# Multi-Head Attention (Self+Cross)
# ==========================================

class DINOBevMixedAttentionMHA(nn.Module):
    """
    ObjectAwareCrossAttention 스타일로
    한 번의 MHA에서 Self-Attn(BEV↔BEV) + Cross-Attn(BEV↔DINO)을 동시에 수행.

    Q = BEV
    K/V = concat([BEV(Self K/V with pos), DINO(Cross K/V with pos)], seq-dim)

    last_tokens: (B, V, Hp*Wp, C_dino)
    patch_hw:    (Hp, Wp)
    img_metas[b]: {'lidar2img': (V,4,4), 'img_shape': (H_img, W_img)}
    """
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,   # e.g., 32
        dino_channels=768,
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        pos_scale_self=1.0,   # BEV pos PE scale for K_self
        pos_scale_dino=1.0,   # DINO pos PE scale for K_dino
        use_ffn=False,         # optional FFN
        ):
        super().__init__()
        
        self.channels = channels
        if num_head_channels == -1:
            # 사용자가 직접 준 num_heads 사용
            assert channels % num_heads == 0, \
                f"q,k,v channels {channels} is not divisible by num_heads {num_heads}"
            self.num_heads = num_heads
            self.d_k = channels // num_heads
        else:
            # 헤드당 채널 수 고정, 헤드 수는 자동 유도
            assert channels % num_head_channels == 0, \
                f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
            self.d_k = num_head_channels

        self.dino_channels = dino_channels
        self.pc_range = pc_range

        # ---- Q from BEV ----
        self.norm_q = normalization(self.channels)
        self.q_proj = conv_nd(1, self.channels, self.num_heads * self.d_k, kernel_size=1)

        # ---- Self K/V from BEV (content + BEV pos) ----
        self.self_pos_raw_dim = 2  # (x_norm, y_norm)
        self.self_pos_dim = max(32, int(self.channels * pos_scale_self))
        self.self_pos_mlp = nn.Sequential(
            conv_nd(1, self.self_pos_raw_dim, self.self_pos_dim, 1),
            nn.GELU(),
            conv_nd(1, self.self_pos_dim, self.self_pos_dim, 1),
        )
        # concat([bev_content(C), bev_pos(self_pos_dim)]) -> project to heads*d_k
        self.W_k_self = conv_nd(1, self.channels + self.self_pos_dim, self.num_heads * self.d_k, 1)
        self.W_v_self = conv_nd(1, self.channels, self.num_heads * self.d_k, 1)

        # ---- Cross K/V from DINO (content + DINO pos) ----
        self.dino_pos_raw_dim = 4  # (u_norm, v_norm, view_ratio, depth_norm)
        self.dino_pos_dim = max(32, int(self.channels * pos_scale_dino))
        self.dino_pos_mlp = nn.Sequential(
            conv_nd(1, self.dino_pos_raw_dim, self.dino_pos_dim, 1),
            nn.GELU(),
            conv_nd(1, self.dino_pos_dim, self.dino_pos_dim, 1),
        )
        # DINO content -> project to C (content embedding), then to heads*d_k
        self.dino_content_to_c = conv_nd(1, self.dino_channels, self.channels, 1)
        self.W_k_dino = conv_nd(1, self.channels + self.dino_pos_dim, self.num_heads * self.d_k, 1)
        self.W_v_dino = conv_nd(1, self.channels, self.num_heads * self.d_k, 1)

        # ---- Output projection (zero-init) ----
        self.proj_out = zero_module(conv_nd(1, self.num_heads * self.d_k, self.channels, 1))

        # ---- Optional FFN ----
        if use_ffn:
            self.ffn = nn.Sequential(
                conv_nd(1, self.channels, 4*self.channels, 1),
                nn.GELU(),
                conv_nd(1, 4*self.channels, self.channels, 1),
            )
        else:
            self.ffn = None

        # ---- Learnable gate for DINO branch (stability) ----
        self.dino_gate = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2) ≈ 0.12

    @torch.no_grad()
    def _sample_dino_tokens(
        self,
        last_tokens,      # (B,V,Hp*Wp,Cd)
        patch_hw,
        uv,               # (B,V,N,2) pixel
        mask,             # (B,V,N) bool
        img_hw,
    ):
        """ bilinear sampling on patch grid → (B,V,N,Cd) """
        B, V, P, Cd = last_tokens.shape
        Hp, Wp = patch_hw
        # (B,V,Hp*Wp,Cd) -> (B,V,Cd,Hp,Wp)
        feat = last_tokens.view(B, V, Hp, Wp, Cd).permute(0,1,4,2,3).contiguous()
        grid = uv_to_patch_grid(uv, img_hw, patch_hw)                                # (B*V, N, 1, 2)
        feat_bv = feat.view(B*V, Cd, Hp, Wp)
        sampled = F.grid_sample(feat_bv, grid, mode="bilinear",
                                align_corners=False, padding_mode="zeros")            # (B*V,Cd,N,1)
        sampled = sampled.squeeze(-1).permute(0,2,1).contiguous()                     # (B*V,N,Cd)
        return sampled.view(B, V, -1, Cd)                                             # (B,V,N,Cd)

    def forward(self,
        bev_feat,                 # (B,C,H,W)
        last_tokens,              # (B,V,Hp*Wp,Cd)
        patch_hw,               # (Hp, Wp)
        img_metas,                  # [{'lidar2img':(V,4,4), 'img_shape':(H_img,W_img)}]*B
        z_samples,                # (S,)
        ):
        
        B, C, H, W = bev_feat.shape
        assert C == self.channels, "bev_feat channels must equal `channels` passed to the module."
        assert last_tokens.shape[-1] == self.dino_channels, \
            "last_tokens feature dim must equal `dino_channels`."
        
        device = bev_feat.device
        L1 = H * W
        S = z_samples.numel()
        Vcams = last_tokens.shape[1]
        Cd = last_tokens.shape[-1]
        Hp, Wp = patch_hw

        # ----- BEV grid & normalized coords (for self pos) -----
        bx, by = build_bev_xy_grid(B, H, W, self.pc_range, device)  # (B,H,W)
        x_min, y_min, z_min, x_max, y_max, z_max = self.pc_range
        x_norm = ((bx - x_min) / max(1e-6, (x_max - x_min))).clamp(0,1).view(B, 1, L1)  # (B,1,L1)
        y_norm = ((by - y_min) / max(1e-6, (y_max - y_min))).clamp(0,1).view(B, 1, L1)  # (B,1,L1)

        # ----- Q from BEV -----
        x = bev_feat.view(B, C, L1)                         # (B,C,L1)
        Q = self.q_proj(self.norm_q(x))                        # (B, H*d_k, L1)

        # ----- Self K/V from BEV -----
        self_pos_raw = torch.cat([x_norm, y_norm], dim=1)         # (B,2,L1)
        self_pos = self.self_pos_mlp(self_pos_raw)                # (B,self_pos_dim,L1)
        K_self_in = torch.cat([x, self_pos], dim=1)               # (B,C+self_pos_dim,L1)
        K_self = self.W_k_self(K_self_in)                         # (B, H*d_k, L1)
        V_self = self.W_v_self(x)                                 # (B, H*d_k, L1)

        # ----- Cross DINO: align & aggregate to one token per BEV cell -----
        # 3D points: (B,V,N=L1*S, 3)
        bx1 = bx.view(B, L1); by1 = by.view(B, L1)
        bz = z_samples.view(1,1,S).expand(B, L1, S).contiguous()
        N = L1 * S
        pts = torch.stack([bx1.unsqueeze(-1).expand(-1,-1,S),
                           by1.unsqueeze(-1).expand(-1,-1,S),
                           bz], dim=-1).view(B, 1, N, 3).expand(-1, Vcams, -1, -1).contiguous()

        lidar2img = torch.stack(
            [torch.as_tensor(m['lidar2img'], device=device, dtype=bev_feat.dtype) for m in img_metas],
            dim=0
        )  # (B,V,4,4)
        H_img, W_img = img_metas[0]['img_shape']
        img_hw = (H_img, W_img)

        uv, mask, depth = project_lidar2img(pts, lidar2img, img_hw)                # (B,V,N,2),(B,V,N),(B,V,N)
        sampled = self._sample_dino_tokens(last_tokens, patch_hw, uv, mask, img_hw) # (B,V,N,Cd)

        # (view,z) 가중 평균 → (B,L1,Cd)
        sampled = sampled.view(B, Vcams, L1, S, Cd)
        mask_vz = mask.view(B, Vcams, L1, S)
        dino_mean_vs = masked_mean(sampled, mask_vz.unsqueeze(-1), dim=1)                # (B,L1,S,Cd)
        dino_mean    = masked_mean(dino_mean_vs, mask_vz.any(dim=1).unsqueeze(-1), dim=2) # (B,L1,Cd)

        # DINO pos raw (u_norm, v_norm, view_ratio, depth_norm) per BEV cell
        uv_ = uv.view(B, Vcams, L1, S, 2)
        uv_mean_vs = masked_mean(uv_, mask_vz.unsqueeze(-1), dim=1)                      # (B,L1,S,2)
        uv_mean    = masked_mean(uv_mean_vs, mask_vz.any(dim=1).unsqueeze(-1), dim=2)    # (B,L1,2)

        visible_views = mask_vz.any(dim=3).float().sum(dim=1)                              # (B,L1)
        view_ratio   = (visible_views / float(Vcams)).clamp(0, 1).view(B, 1, L1)

        u_norm = (uv_mean[..., 0] / max(1, W_img - 1)).clamp(0,1).view(B, 1, L1)
        v_norm = (uv_mean[..., 1] / max(1, H_img - 1)).clamp(0,1).view(B, 1, L1)
        depth_mean = masked_mean(depth.view(B, Vcams, L1, S), mask_vz, dim=1)
        depth_mean = masked_mean(depth_mean, mask_vz.any(dim=1), dim=2).view(B, 1, L1)
        z_min = max(1e-6, self.pc_range[2]); z_max = max(z_min + 1e-3, self.pc_range[5])
        d_norm = ((depth_mean - z_min) / (z_max - z_min)).clamp(0,1)                      # (B,1,L1)

        # DINO content/pos -> heads*d_k
        dino_c = self.dino_content_to_c(dino_mean.transpose(1,2).contiguous())            # (B,C,L1)
        dino_pos_raw = torch.cat([u_norm, v_norm, view_ratio, d_norm], dim=1)             # (B,4,L1)
        dino_pos = self.dino_pos_mlp(dino_pos_raw)                                        # (B,dino_pos_dim,L1)

        K_dino = self.W_k_dino(torch.cat([dino_c, dino_pos], dim=1))                      # (B,H*d_k,L1)
        V_dino = self.W_v_dino(dino_c)                                                    # (B,H*d_k,L1)

        # ---- Mixed memory: concat on sequence dim ----
        g = torch.sigmoid(self.dino_gate)                                                 # scalar in (0,1)
        K_mix = torch.cat([K_self, K_dino], dim=2)                                        # (B,H*d_k, 2*L1)
        V_mix = torch.cat([V_self, g * V_dino], dim=2)                                    # (B,H*d_k, 2*L1)

        # ---- Multi-Head Attention (per-head) ----
        B_, Hd = B, self.num_heads * self.d_k
        Lq, Lk = L1, K_mix.shape[2]
        Qh = Q.view(B_, self.num_heads, self.d_k, Lq).reshape(B_*self.num_heads, self.d_k, Lq)
        Kh = K_mix.view(B_, self.num_heads, self.d_k, Lk).reshape(B_*self.num_heads, self.d_k, Lk)
        Vh = V_mix.view(B_, self.num_heads, self.d_k, Lk).reshape(B_*self.num_heads, self.d_k, Lk)

        # FP16-friendly scaling
        scale = 1.0 / math.sqrt(math.sqrt(float(self.d_k)))
        attn_logits = torch.einsum("bcl, bcs -> bls", Qh*scale, Kh*scale)                 # (B*H, Lq, Lk)
        attn = F.softmax(attn_logits.float(), dim=-1).type_as(attn_logits)
        out_h = torch.einsum("bls, bcs -> bcl", attn, Vh)                                 # (B*H, d_k, Lq)
        out = out_h.view(B_, self.num_heads, self.d_k, Lq).reshape(B_, Hd, Lq)            # (B,H*d_k,L1)

        out = self.proj_out(out)                                                          # (B,C,L1)
        y = (x + out).view(B, C, H, W)                                                    # residual add

        if self.ffn is not None:
            y_ffn = self.ffn(y.view(B, C, L1)).view(B, C, H, W)
            y = y + y_ffn

        return y

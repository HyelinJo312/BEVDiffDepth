# ---- UNet intermediates + LiDAR TOP 1x5 renderer ----
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# nuScenes bits (only needed if you draw the LiDAR pane)
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import BoxVisibility
import numpy as np
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
from PIL import Image
import os

# ---------------------------- utils -----------------------------
def _to_numpy(x):
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)

def _slugify(s: str, maxlen: int = 80) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[|:/\\]+", " ", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9._-]", "", s)
    return s[:maxlen] or "unet_intermediates"

def _ensure_outfile(out_dir, title, ext=".png"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{_slugify(title) if title else 'unet_intermediates'}{ext}")

def _percentile_norm_multi(arrs, p_low=2.0, p_high=98.0, eps=1e-6):
    """Joint percentile scaling of multiple maps to [0,1] for fair comparison."""
    vec = np.concatenate([a.ravel() for a in arrs])
    lo, hi = np.percentile(vec, [p_low, p_high])
    def scale(x):
        return np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0).astype(np.float32)
    return [scale(a) for a in arrs]

def _per_map_zscore_then_clip(maps, eps=1e-6, p_low=1.0, p_high=99.0):
    outs = []
    for m in maps:
        x = (m - m.mean()) / (m.std() + eps)
        lo, hi = np.percentile(x, [p_low, p_high])
        x = np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0)
        outs.append(x.astype(np.float32))
    return outs

def _gaussian_blur_np(img, sigma=0.8):
    if sigma <= 0:
        return img.astype(np.float32)
    try:
        import cv2
        return cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT_101)
    except Exception:
        k = int(max(3, 2*round(3*sigma)+1))
        x = np.arange(k) - (k-1)/2
        g = np.exp(-(x**2)/(2*sigma*sigma)).astype(np.float32); g /= g.sum()
        K = np.outer(g, g); pad = k//2
        gp = np.pad(img.astype(np.float32), ((pad,pad),(pad,pad)), mode="reflect")
        out = np.zeros_like(img, dtype=np.float32)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                out[i, j] = float((gp[i:i+k, j:j+k] * K).sum())
        return out

def bev_extent_from_cfg(bev_cfg):
    # point_cloud_range = [x_min, y_min, z_min, x_max, y_max, z_max]
    x_min, y_min, _, x_max, y_max, _ = bev_cfg.point_cloud_range
    return (x_min, x_max, y_min, y_max)  # (xmin, xmax, ymin, ymax)

def _aggregate_energy(feat_2d, agg="l1", whiten=True, eps=1e-6):
    if whiten:
        m = feat_2d.mean(axis=0, keepdims=True)
        s = feat_2d.std(axis=0, keepdims=True) + eps
        z = (feat_2d - m) / s
    else:
        z = feat_2d

    if agg == "l1":
        e = np.mean(np.abs(z), axis=1)
    elif agg == "rms":
        e = np.sqrt(np.mean(z**2, axis=1))
    elif agg == "max":
        e = np.max(np.abs(z), axis=1)
    elif agg == "l1_pos":          # 양의 활성만
        e = np.mean(np.clip(z, 0, None), 1)
    elif agg == "signed_mean":     # 부호 보존 평균
        e = np.mean(z, 1)
    return e.astype(np.float32)

# def _get_color(nusc: NuScenes, category_name: str):
#     if category_name == 'bicycle':
#         return np.array(nusc.colormap['vehicle.bicycle']) / 255.0
#     if category_name == 'construction_vehicle':
#         return np.array(nusc.colormap['vehicle.construction']) / 255.0
#     if category_name == 'traffic_cone':
#         return np.array(nusc.colormap['movable_object.trafficcone']) / 255.0
#     for key in nusc.colormap.keys():
#         if category_name in key:
#             return np.array(nusc.colormap[key]) / 255.0
#     return np.array([0, 0, 0], dtype=np.float32)

def _get_color(nusc: NuScenes, category_name: str):
    # Dark orange로 모든 bbox 색상 통일
    return np.array([1.0, 0.55, 0.0], dtype=np.float32)

# -------------- LiDAR_TOP pane --------------
def _draw_lidar_top_on_axes(nusc, sample_token, ax,
                            view=np.eye(4),
                            box_vis_level=BoxVisibility.ANY,
                            axes_limit=50.0,
                            show_boxes=True,
                            lidar_render_mode="scatter",   # 'scatter' | 'height'
                            lidar_cmap="viridis",
                            pts_size=0.6,
                            pts_stride=3,
                            pts_alpha=0.9,
                            box_lw=0.6,
                            bev_extent=None):
    sample_record = nusc.get('sample', sample_token)
    assert 'LIDAR_TOP' in sample_record['data'], "No LIDAR_TOP for this sample."
    lidar_token = sample_record['data']['LIDAR_TOP']
    data_path, boxes, _ = nusc.get_sample_data(lidar_token, box_vis_level=box_vis_level)

    if lidar_render_mode == "height":
        curr = plt.get_cmap()
        plt.set_cmap(lidar_cmap)
        LidarPointCloud.from_file(data_path).render_height(ax, view=view)
        plt.set_cmap(curr)
    else:
        pc = LidarPointCloud.from_file(data_path).points  # (4, N)
        if pts_stride > 1:
            pc = pc[:, ::pts_stride]
        P = np.vstack([pc[:3], np.ones(pc.shape[1], dtype=np.float32)])  # (4,N)
        PV = view @ P
        x, y = PV[0], PV[1]
        z = pc[2]
        zmin, zmax = np.percentile(z, [2.0, 98.0])
        z = np.clip((z - zmin) / (zmax - zmin + 1e-6), 0.0, 1.0)
        ax.scatter(x, y, c=z, s=pts_size, alpha=pts_alpha,
                   cmap=lidar_cmap, marker='.', linewidths=0, rasterized=True)

    if show_boxes:
        for box in boxes:
            c = _get_color(nusc, box.name)
            box.render(ax, view=view, colors=(c, c, c), linewidth=box_lw)

    if bev_extent is not None:
        xmin, xmax, ymin, ymax = bev_extent
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    else:
        ax.set_xlim(-axes_limit, axes_limit); ax.set_ylim(-axes_limit, axes_limit)
    ax.axis('off'); ax.set_aspect('equal')

# ----------------------------------------------------------------------------------------


# -------------- MAIN: 4 UNet features + LiDAR TOP --------------
def render_unet_intermediates(
    f1_bchw, f2_bchw, f3_bchw, f4_bchw,   # tensors/ndarrays
    b=0,
    nusc: NuScenes = None,
    sample_token: str = None,
    out_dir: str = None,
    title: str = None,
    labels=("mid 12×12", "out 12×12", "out 25×25", "out 50×50", "LiDAR Top"),
    # BEV drawing params (shared)
    mode="energy",                 # 'energy' | 'pca_pc3'
    agg="l1", whiten=True,         # for energy
    smooth_sigma=0.8,
    joint_clip=(2.0, 98.0), gamma=1.0,
    bev_cmap="viridis", bev_interp="nearest",
    bev_extent=None, bev_origin="lower",   # set from bev_cfg via bev_extent_from_cfg
    # LiDAR params
    lidar_axes_limit=50.0, lidar_view=np.eye(4), lidar_show_boxes=True,
    # canvas
    figsize=(22, 4.5), dpi=260, show=False
    ):
    
    Fs = [ _to_numpy(f) for f in [f1_bchw, f2_bchw, f3_bchw, f4_bchw] ]
    for i, F in enumerate(Fs):
        if F.ndim != 4:
            raise ValueError(f"Feature {i+1} must be [B,C,H,W], got {F.shape}")
        if not (0 <= b < F.shape[0]):
            raise IndexError(f"batch index {b} out of range for feature {i+1} with B={F.shape[0]}")

    # ---- make per-map 2D activation using chosen mode ----
    maps = []
    for F in Fs:
        _, C, H, W = F.shape
        feat_2d = F[b].reshape(C, H*W).T  # [N,C]
        if mode == "energy":
            m = _aggregate_energy(feat_2d, agg=agg, whiten=whiten).reshape(H, W)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        # optional mild smoothing (helps 12×12 / 25×25)
        m = _gaussian_blur_np(m, sigma=smooth_sigma) if smooth_sigma > 0 else m
        maps.append(m)

    # ---- joint normalization across all 4 maps ----
    maps = _percentile_norm_multi(maps, p_low=joint_clip[0], p_high=joint_clip[1])
    if gamma != 1.0:
        maps = [np.power(m, gamma).astype(np.float32) for m in maps]
    scales = [(m, 0.0, 1.0) for m in maps]

    # ---- draw 1x5 ----
    fig, axes = plt.subplots(1, 5, figsize=figsize, dpi=dpi)

    # 4 BEV panes
    for ax, (m, vmin, vmax), lab in zip(axes[:4], scales, labels[:4]):
        ax.imshow(
            m, cmap=bev_cmap, vmin=vmin, vmax=vmax, interpolation=bev_interp,
            extent=bev_extent, origin=bev_origin
        )
        ax.set_aspect('equal')
        if bev_extent is not None:
            ax.set_xlim(bev_extent[0], bev_extent[1])
            ax.set_ylim(bev_extent[2], bev_extent[3])
        ax.set_title(lab)
        ax.axis('off')

    # LiDAR TOP pane
    if (nusc is not None) and (sample_token is not None):
        _draw_lidar_top_on_axes(
            nusc, sample_token, axes[4],
            view=lidar_view,
            box_vis_level=BoxVisibility.ANY,
            axes_limit=lidar_axes_limit,
            show_boxes=lidar_show_boxes,
            lidar_render_mode="scatter",
            lidar_cmap="viridis",
            pts_size=2.0,       
            pts_stride=1,        
            pts_alpha=0.9,
            box_lw=0.6,
            bev_extent=bev_extent
        )
    else:
        axes[4].text(0.5, 0.5, "LiDAR_TOP unavailable",
                     ha="center", va="center", fontsize=10)
        axes[4].axis('off'); axes[4].set_aspect('equal')
    axes[4].set_title(labels[4])

    if title:
        fig.suptitle(title, y=0.99)

    if out_dir is not None:
        out_file = _ensure_outfile(out_dir, title or "unet_intermediates_1x5")
        plt.savefig(out_file, bbox_inches='tight', pad_inches=0.0, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)

    return maps  



# -------------- MAIN: Original feature + 4 UNet features + Concat feature + LiDAR TOP --------------
def render_unet_intermediates_four(
    pre_bchw=None,                 # ← pre-UNet BEV feature (e.g., original_bev)
    f1_bchw=None, f2_bchw=None, f3_bchw=None, f4_bchw=None,   # 기존 intermediate들
    concat_bchw=None,              # ← 마지막 multi-scale concat feature
    b=0,
    nusc: NuScenes = None,
    sample_token: str = None,
    out_dir: str = None,
    title: str = None,
    labels=None,                   # 자동 생성(넘기면 그대로 사용)
    mode="energy",                 # 'energy' | 'pca_pc3'
    agg="l1", whiten=True,         # for energy
    smooth_sigma=0.8,
    joint_clip=(2.0, 98.0), gamma=1.0,
    bev_cmap="viridis", bev_interp="nearest",
    bev_extent=None, bev_origin="lower",
    # LiDAR params
    lidar_axes_limit=50.0, lidar_view=np.eye(4), lidar_show_boxes=True,
    # canvas
    figsize=(26, 4.8), dpi=260, show=False
):
    # -------- 1) 입력 정리: 존재하는 feature만 순서대로 쌓기 --------
    items = []
    # if pre_bchw is not None: items.append(("original", pre_bchw))
    if f1_bchw  is not None: items.append(("mid-block 12×12", f1_bchw))
    if f2_bchw  is not None: items.append(("out-block 12×12", f2_bchw))
    if f3_bchw  is not None: items.append(("out-block 25×25", f3_bchw))
    if f4_bchw  is not None: items.append(("out-block 50×50", f4_bchw))
    if concat_bchw is not None: items.append(("multi-scale concat", concat_bchw))
    if pre_bchw is not None: items.append(("original", pre_bchw))

    if not items:
        raise ValueError("시각화할 feature가 없습니다 (pre_bchw, f1..f4_bchw, concat_bchw 중 하나 이상 필요).")

    # 사용자 labels가 있으면 대체
    if labels is not None:
        # 마지막 LiDAR 제목까지 포함해도 되고, 안 넣어도 상관 없음
        # feature 개수와 맞지 않으면 자동 생성으로 fallback
        if isinstance(labels, (list, tuple)) and len(labels) >= len(items):
            names = list(labels[:len(items)])
        else:
            names = [n for n, _ in items]
    else:
        names = [n for n, _ in items]

    # -------- 2) 모양/인덱스 검사 + BEV map 생성 --------
    maps = []
    for i, (name, F) in enumerate(items):
        F = _to_numpy(F)
        if F.ndim != 4:
            raise ValueError(f"[{name}] must be [B,C,H,W], got {F.shape}")
        if not (0 <= b < F.shape[0]):
            raise IndexError(f"[{name}] batch index {b} out of range (B={F.shape[0]})")

        _, C, H, W = F.shape
        feat_2d = F[b].reshape(C, H*W).T  # [N,C]

        if mode == "energy":
            m = _aggregate_energy(feat_2d, agg=agg, whiten=whiten).reshape(H, W)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        m = _gaussian_blur_np(m, sigma=smooth_sigma) if smooth_sigma > 0 else m
        maps.append(m)

    # -------- 3) joint normalization (모든 맵 공정 비교) --------
    # scaled_maps = _percentile_norm_multi(maps, p_low=joint_clip[0], p_high=joint_clip[1])
    scaled_maps = _per_map_zscore_then_clip(maps, p_low=joint_clip[0], p_high=joint_clip[1])
    if gamma != 1.0:
        scaled_maps = [np.power(m, gamma).astype(np.float32) for m in scaled_maps]
    vmins = [0.0] * len(maps); vmaxs = [1.0] * len(maps)

    # -------- 4) 그리기 (features + LiDAR) --------
    n_feat = len(scaled_maps)
    n_cols = n_feat + 1  # + LiDAR
    fig, axes = plt.subplots(1, n_cols, figsize=figsize, dpi=dpi)

    # feature panes
    for ax, m, vmin, vmax, lab in zip(axes[:n_feat], scaled_maps, vmins, vmaxs, names):
        ax.imshow(
            m, cmap=bev_cmap, vmin=vmin, vmax=vmax, interpolation=bev_interp,
            extent=bev_extent, origin=bev_origin
        )
        ax.set_aspect('equal')
        if bev_extent is not None:
            ax.set_xlim(bev_extent[0], bev_extent[1])
            ax.set_ylim(bev_extent[2], bev_extent[3])
        ax.set_title(lab, fontsize=20); ax.axis('off')

    # LiDAR pane (마지막)
    lidar_ax = axes[-1]
    if (nusc is not None) and (sample_token is not None):
        _draw_lidar_top_on_axes(
            nusc, sample_token, lidar_ax,
            view=lidar_view,
            box_vis_level=BoxVisibility.ANY,
            axes_limit=lidar_axes_limit,
            show_boxes=lidar_show_boxes,
            lidar_render_mode="scatter",
            lidar_cmap="viridis",
            pts_size=2.1,
            pts_stride=1,
            pts_alpha=0.9,
            box_lw=0.8,
            bev_extent=bev_extent
        )
        lidar_ax.set_title("LiDAR Top")
    else:
        lidar_ax.text(0.5, 0.5, "LiDAR_TOP unavailable",
                      ha="center", va="center", fontsize=20)
        lidar_ax.axis('off'); lidar_ax.set_aspect('equal')
        lidar_ax.set_title("LiDAR Top View")

    if title:
        fig.suptitle(title, y=0.99)

    if out_dir is not None:
        out_file = _ensure_outfile(out_dir, title or "unet_intermediates")
        plt.savefig(out_file, bbox_inches='tight', pad_inches=0.0, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)

    # label → map (raw or scaled) 딕셔너리 반환
    out = {}
    for lab, m in zip(names, maps):
        out[lab] = m
    return out


# -------------- MAIN: Original feature + 3 UNet features + Concat feature + LiDAR TOP --------------
def render_unet_intermediates_three(
    pre_bchw=None,                 # ← pre-UNet BEV feature (e.g., original_bev)
    f2_bchw=None, f3_bchw=None, f4_bchw=None,   # 기존 intermediate들
    concat_bchw=None,              # ← 마지막 multi-scale concat feature
    b=0,
    nusc: NuScenes = None,
    sample_token: str = None,
    out_dir: str = None,
    title: str = None,
    labels=None,                   # 자동 생성(넘기면 그대로 사용)
    mode="energy",                 # 'energy' | 'pca_pc3'
    agg="l1", whiten=True,         # for energy
    smooth_sigma=0.8,
    joint_clip=(2.0, 98.0), gamma=1.0,
    bev_cmap="viridis", bev_interp="nearest",
    bev_extent=None, bev_origin="lower",
    # LiDAR params
    lidar_axes_limit=50.0, lidar_view=np.eye(4), lidar_show_boxes=True,
    # canvas
    figsize=(26, 4.8), dpi=260, show=False
):
    # -------- 1) 입력 정리: 존재하는 feature만 순서대로 쌓기 --------
    items = []
    # if pre_bchw is not None: items.append(("original", pre_bchw))
    if f2_bchw  is not None: items.append(("out-block 12×12", f2_bchw))
    if f3_bchw  is not None: items.append(("out-block 25×25", f3_bchw))
    if f4_bchw  is not None: items.append(("out-block 50×50", f4_bchw))
    if concat_bchw is not None: items.append(("concat (multi-scale)", concat_bchw))
    if pre_bchw is not None: items.append(("original", pre_bchw))

    if not items:
        raise ValueError("시각화할 feature가 없습니다 (pre_bchw, f1..f4_bchw, concat_bchw 중 하나 이상 필요).")

    # 사용자 labels가 있으면 대체
    if labels is not None:
        # 마지막 LiDAR 제목까지 포함해도 되고, 안 넣어도 상관 없음
        # feature 개수와 맞지 않으면 자동 생성으로 fallback
        if isinstance(labels, (list, tuple)) and len(labels) >= len(items):
            names = list(labels[:len(items)])
        else:
            names = [n for n, _ in items]
    else:
        names = [n for n, _ in items]

    # -------- 2) 모양/인덱스 검사 + BEV map 생성 --------
    maps = []
    for i, (name, F) in enumerate(items):
        F = _to_numpy(F)
        if F.ndim != 4:
            raise ValueError(f"[{name}] must be [B,C,H,W], got {F.shape}")
        if not (0 <= b < F.shape[0]):
            raise IndexError(f"[{name}] batch index {b} out of range (B={F.shape[0]})")

        _, C, H, W = F.shape
        feat_2d = F[b].reshape(C, H*W).T  # [N,C]

        if mode == "energy":
            m = _aggregate_energy(feat_2d, agg=agg, whiten=whiten).reshape(H, W)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        m = _gaussian_blur_np(m, sigma=smooth_sigma) if smooth_sigma > 0 else m
        maps.append(m)

    # -------- 3) joint normalization (모든 맵 공정 비교) --------
    scaled_maps = _percentile_norm_multi(maps, p_low=joint_clip[0], p_high=joint_clip[1])
    if gamma != 1.0:
        scaled_maps = [np.power(m, gamma).astype(np.float32) for m in scaled_maps]
    vmins = [0.0] * len(maps); vmaxs = [1.0] * len(maps)

    # -------- 4) 그리기 (features + LiDAR) --------
    n_feat = len(scaled_maps)
    n_cols = n_feat + 1  # + LiDAR
    fig, axes = plt.subplots(1, n_cols, figsize=figsize, dpi=dpi)

    # feature panes
    for ax, m, vmin, vmax, lab in zip(axes[:n_feat], scaled_maps, vmins, vmaxs, names):
        ax.imshow(
            m, cmap=bev_cmap, vmin=vmin, vmax=vmax, interpolation=bev_interp,
            extent=bev_extent, origin=bev_origin
        )
        ax.set_aspect('equal')
        if bev_extent is not None:
            ax.set_xlim(bev_extent[0], bev_extent[1])
            ax.set_ylim(bev_extent[2], bev_extent[3])
        ax.set_title(lab); ax.axis('off')

    # LiDAR pane (마지막)
    lidar_ax = axes[-1]
    if (nusc is not None) and (sample_token is not None):
        _draw_lidar_top_on_axes(
            nusc, sample_token, lidar_ax,
            view=lidar_view,
            box_vis_level=BoxVisibility.ANY,
            axes_limit=lidar_axes_limit,
            show_boxes=lidar_show_boxes,
            lidar_render_mode="scatter",
            lidar_cmap="viridis",
            pts_size=2.0,
            pts_stride=1,
            pts_alpha=0.9,
            box_lw=0.6,
            bev_extent=bev_extent
        )
        lidar_ax.set_title("LiDAR Top")
    else:
        lidar_ax.text(0.5, 0.5, "LiDAR_TOP unavailable",
                      ha="center", va="center", fontsize=10)
        lidar_ax.axis('off'); lidar_ax.set_aspect('equal')
        lidar_ax.set_title("LiDAR Top View")

    if title:
        fig.suptitle(title, y=0.99)

    if out_dir is not None:
        out_file = _ensure_outfile(out_dir, title or "unet_intermediates")
        plt.savefig(out_file, bbox_inches='tight', pad_inches=0.0, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)

    # label → map (raw or scaled) 딕셔너리 반환
    out = {}
    for lab, m in zip(names, maps):
        out[lab] = m
    return out


def _safe_imread_rgb(path):
    if (path is None) or (not os.path.exists(path)):
        raise FileNotFoundError(f"Not found: {path}")
    img = Image.open(path).convert('RGB')
    return np.asarray(img)

def _camera_name_from_path(p):
    parts = os.path.normpath(p).split(os.sep)
    try:
        idx = parts.index('samples')
        return parts[idx+1]  # e.g., CAM_FRONT
    except Exception:
        return os.path.basename(p)
    
def render_sixcams_lidar_bev(
    # --- BEV/UNet features (B,C,H,W) ---
    pre_bchw=None, f1_bchw=None, f2_bchw=None, f3_bchw=None, f4_bchw=None, concat_bchw=None,
    b=0,
    # --- NuScenes / tokens ---
    nusc=None, sample_token=None,
    # --- IO / titles ---
    out_dir=None, title=None,
    # --- BEV map params ---
    mode="energy", agg="l1", whiten=True,
    smooth_sigma=0.8, joint_clip=(2.0, 98.0), gamma=1.0,
    bev_cmap="viridis", bev_interp="nearest",
    bev_extent=None, bev_origin="lower",
    # --- LiDAR params ---
    lidar_axes_limit=50.0, lidar_view=np.eye(4), lidar_show_boxes=True,
    # --- Camera images ---
    cam_image_paths=None,  # 6개 경로 권장
    # --- Figure ---
    figsize=(26, 10), dpi=220, show=False
    ):
    """
    레이아웃:
    ----------------------------------------------------------------------
    CAM_FRONT_LEFT | CAM_FRONT | CAM_FRONT_RIGHT |    LiDAR Top (row-span)
    CAM_BACK_LEFT  | CAM_BACK  | CAM_BACK_RIGHT  |    LiDAR Top (row-span)
    ----------------------------------------------------------------------
    [BEV Feature 1] [BEV Feature 2] ... [Feature 6]   (아래 단일 행)
    """

    items = []
    if pre_bchw is not None: items.append(("original", pre_bchw))
    if f1_bchw  is not None: items.append(("mid-block 12×12", f1_bchw))
    if f2_bchw  is not None: items.append(("out-block 12×12", f2_bchw))
    if f3_bchw  is not None: items.append(("out-block 25×25", f3_bchw))
    if f4_bchw  is not None: items.append(("out-block 50×50", f4_bchw))
    if concat_bchw is not None: items.append(("concat (multi-scale)", concat_bchw))
    if not items:
        raise ValueError("시각화할 BEV feature가 없습니다.")

    names = [n for n, _ in items]
    maps = []
    for (name, F) in items:
        F = _to_numpy(F)
        B, C, H, W = F.shape
        if not (0 <= b < B):
            raise IndexError(f"[{name}] batch index {b} out of range (B={B})")
        feat_2d = F[b].reshape(C, H*W).T  # (N,C)
        if mode == "energy":
            m = _aggregate_energy(feat_2d, agg=agg, whiten=whiten).reshape(H, W)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        m = _gaussian_blur_np(m, sigma=smooth_sigma) if smooth_sigma > 0 else m
        maps.append(m)

    # 공정 비교용 정규화
    scaled_maps = _percentile_norm_multi(maps, p_low=joint_clip[0], p_high=joint_clip[1])
    if gamma != 1.0:
        scaled_maps = [np.power(m, gamma).astype(np.float32) for m in scaled_maps]
    vmins = [0.0]*len(maps); vmaxs = [1.0]*len(maps)

    # 아래 행에 최대 6개만 깔끔히 배치 (요청 레이아웃)
    n_feat = min(len(scaled_maps), 6)
    scaled_maps = scaled_maps[:n_feat]
    names = names[:n_feat]
    vmins = vmins[:n_feat]; vmaxs = vmaxs[:n_feat]

    # ---------- 2) 그리드 구성 ----------
    # 상단 2행은 4열 구조: 3열에 카메라, 마지막 1열에 LiDAR(2행 세로 병합)
    # 하단 1행은 BEV feature들을 n_feat개 만큼 좌→우로 배치
    n_cols_top = 4
    n_cols_bottom = max(n_feat, 4)  # 하단이 더 넓을 수 있음
    n_cols = max(n_cols_top, n_cols_bottom)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    # rows: 0,1 (cams + lidar), row:2 (bev features)
    gs = fig.add_gridspec(3, n_cols, height_ratios=[1.0, 1.0, 1.2])

    # ---------- 3) 카메라 6장: 간격 0으로 딱 붙이기 ----------
    cam_paths = list(cam_image_paths or [])
    # 카메라 블록(좌측 3열 × 2행)을 서브그리드로 분리해서 wspace/hspace=0
    cam_gs = gs[0:2, 0:3].subgridspec(2, 3, wspace=0, hspace=0)
    
    if len(cam_paths) >= 6:
        # 앞줄 3장 (row 0)
        for i in range(3):
            ax = fig.add_subplot(cam_gs[0, i])
            try:
                img = _safe_imread_rgb(cam_paths[i])
                ax.imshow(img)
                ax.set_title(_camera_name_from_path(cam_paths[i]), fontsize=10)
            except Exception:
                ax.text(0.5, 0.5, f"Load fail\n{i}", ha='center', va='center', fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(0, img.shape[1] if 'img' in locals() else 1)
            ax.set_ylim(img.shape[0] if 'img' in locals() else 1, 0)
            ax.axis('off')
            for s in ax.spines.values(): s.set_visible(False)

        # 뒷줄 3장 (row 1)
        for i in range(3):
            ax = fig.add_subplot(cam_gs[1, i])
            try:
                img = _safe_imread_rgb(cam_paths[3+i])
                ax.imshow(img)
                ax.set_title(_camera_name_from_path(cam_paths[3+i]), fontsize=10)
            except Exception:
                ax.text(0.5, 0.5, f"Load fail\n{3+i}", ha='center', va='center', fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(0, img.shape[1] if 'img' in locals() else 1)
            ax.set_ylim(img.shape[0] if 'img' in locals() else 1, 0)
            ax.axis('off')
            for s in ax.spines.values(): s.set_visible(False)
    else:
        # 부족하면 빈 칸
        for r in [0,1]:
            for c in range(3):
                ax = fig.add_subplot(cam_gs[r, c])
                ax.axis('off')

    # ---------- 4) LiDAR Top: 우측 열(3번 컬럼) 2행 세로 병합 ----------
    lidar_col = min(3, n_cols-1)
    lidar_ax = fig.add_subplot(gs[0:2, lidar_col])
    if (nusc is not None) and (sample_token is not None):
        _draw_lidar_top_on_axes(
            nusc, sample_token, lidar_ax,
            view=lidar_view, box_vis_level=BoxVisibility.ANY,
            axes_limit=lidar_axes_limit, show_boxes=lidar_show_boxes,
            lidar_render_mode="scatter", lidar_cmap="viridis",
            pts_size=2.0, pts_stride=1, pts_alpha=0.9,
            box_lw=0.6, bev_extent=bev_extent
        )
        lidar_ax.set_title("LiDAR Top")
    else:
        lidar_ax.text(0.5,0.5,"LiDAR_TOP unavailable",ha='center',va='center',fontsize=10)
        lidar_ax.axis('off'); lidar_ax.set_aspect('equal')

    # ---------- 4) 하단 1행: BEV features n_feat개 ----------
    n_feat = min(len(scaled_maps), 6)
    for i in range(n_feat):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(
            scaled_maps[i], cmap=bev_cmap, vmin=vmins[i], vmax=vmaxs[i],
            interpolation=bev_interp, extent=bev_extent, origin=bev_origin
        )
        ax.set_aspect('equal')
        if bev_extent is not None:
            ax.set_xlim(bev_extent[0], bev_extent[1])
            ax.set_ylim(bev_extent[2], bev_extent[3])
        ax.set_title(names[i], fontsize=10)
        ax.axis('off')

    if title:
        fig.suptitle(title, y=0.99)
    
    if out_dir is not None:
        out_file = _ensure_outfile(out_dir, title or "unet_intermediates")
        plt.savefig(out_file, bbox_inches='tight', pad_inches=0.0, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)

# -------------- MAIN: all intermidate features + LiDAR TOP --------------

def render_unet_intermediates_all(
    inter_list,                # 길이 10, 작은 해상도 -> 큰 해상도 [B,C,H,W]
    pre_bchw,                  # original feature
    concat_bchw,               # concat feature
    b=0,
    nusc: NuScenes = None,
    sample_token: str = None,
    out_dir: str = None,
    title: str = None,
    labels=None,   
    mode="energy", agg="l1", whiten=True,
    smooth_sigma=0.8,
    joint_clip=(2.0, 98.0), gamma=1.0,
    bev_cmap="viridis", bev_interp="nearest",
    bev_extent=None, bev_origin="lower",
    # LiDAR
    lidar_axes_limit=50.0, lidar_view=np.eye(4), lidar_show_boxes=True,
    figsize=(22, 12), dpi=300, show=False,
):

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import numpy as np

    # -------------------- 준비/라벨 --------------------
    if not isinstance(inter_list, (list, tuple)) or len(inter_list) != 10:
        raise ValueError("inter_list는 길이 10의 리스트여야 합니다.")

    items, names = [], []
    for i, F in enumerate(inter_list):
        Fnp = _to_numpy(F)
        if Fnp.ndim != 4: raise ValueError(f"[inter{i}] must be [B,C,H,W], got {Fnp.shape}")
        if not (0 <= b < Fnp.shape[0]): raise IndexError(f"[inter{i}] batch {b} out of range")
        _, _, H, W = Fnp.shape
        # lab = f"inter{i} proj" if i == 9 else f"inter{i} {H}x{W}"
        lab = f"mid-block {H}x{W}" if i == 0 else f"inter{i} {H}x{W}"
        items.append((lab, Fnp)); names.append(lab)

    pre_np    = _to_numpy(pre_bchw)
    concat_np = _to_numpy(concat_bchw)
    if pre_np.ndim != 4 or concat_np.ndim != 4:
        raise ValueError("pre_bchw, concat_bchw는 [B,C,H,W]여야 합니다.")
    if not (0 <= b < pre_np.shape[0]):    raise IndexError(f"[original] batch {b} out of range")
    if not (0 <= b < concat_np.shape[0]): raise IndexError(f"[concat]   batch {b} out of range")
    items.append(("original", pre_np));  names.append("original")
    # items.append(("concat",   concat_np)); names.append("concat")
    items.append(("proj output", concat_np)); names.append("proj output")

    # -------------------- 맵 생성 --------------------
    maps = []
    for name, F in items:
        _, C, H, W = F.shape
        feat_2d = F[b].reshape(C, H*W).T
        if mode == "energy":
            m = _aggregate_energy(feat_2d, agg=agg, whiten=whiten).reshape(H, W)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        m = _gaussian_blur_np(m, sigma=smooth_sigma) if smooth_sigma > 0 else m
        maps.append(m)

    # -------------------- joint 정규화 --------------------
    scaled_maps = _percentile_norm_multi(maps, p_low=joint_clip[0], p_high=joint_clip[1])
    # scaled_maps = _per_map_zscore_then_clip(maps, p_low=joint_clip[0], p_high=joint_clip[1])
    if gamma != 1.0:
        scaled_maps = [np.power(m, gamma).astype(np.float32) for m in scaled_maps]
    vmins = [0.0] * len(scaled_maps); vmaxs = [1.0] * len(scaled_maps)

    # -------------------- GridSpec (2x7) --------------------
    # 옵션 A) 마지막 열을 더 넓게: width_ratios에서 마지막만 크게
    LIDAR_COL_RATIO = 2.2  # ← 더 크게 보이게 하고 싶으면 2.5~3.0까지 올려보세요
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(
        2, 7, figure=fig,
        width_ratios=[1, 1, 1, 1, 1, 1, LIDAR_COL_RATIO],  # ← 핵심
        wspace=0.02, hspace=0.08
    )

    # inter 10개: col 0..4, row 0..1
    inter_axes = []
    for idx in range(10):
        r, c = divmod(idx, 5)
        inter_axes.append(fig.add_subplot(gs[r, c]))

    # original/concat
    ax_orig   = fig.add_subplot(gs[0, 5])
    ax_concat = fig.add_subplot(gs[1, 5])

    # LiDAR(세로 2칸 병합, 마지막 열 전체)
    ax_lidar  = fig.add_subplot(gs[:, 6])

    # -------------------- 그리기 --------------------
    for idx, ax in enumerate(inter_axes):
        ax.imshow(scaled_maps[idx], cmap=bev_cmap, vmin=vmins[idx], vmax=vmaxs[idx],
                  interpolation=bev_interp, extent=bev_extent, origin=bev_origin)
        if bev_extent is not None:
            ax.set_xlim(bev_extent[0], bev_extent[1]); ax.set_ylim(bev_extent[2], bev_extent[3])
        ax.set_aspect('equal'); ax.axis('off'); ax.set_title(names[idx], fontsize=8)

    ax_orig.imshow(scaled_maps[10], cmap=bev_cmap, vmin=vmins[10], vmax=vmaxs[10],
                   interpolation=bev_interp, extent=bev_extent, origin=bev_origin)
    ax_orig.set_aspect('equal'); ax_orig.axis('off'); ax_orig.set_title("original", fontsize=8)

    ax_concat.imshow(scaled_maps[11], cmap=bev_cmap, vmin=vmins[11], vmax=vmaxs[11],
                     interpolation=bev_interp, extent=bev_extent, origin=bev_origin)
    # ax_concat.set_aspect('equal'); ax_concat.axis('off'); ax_concat.set_title("concat", fontsize=8)
    ax_concat.set_aspect('equal'); ax_concat.axis('off'); ax_concat.set_title("proj output", fontsize=8)

    if (nusc is not None) and (sample_token is not None):
        _draw_lidar_top_on_axes(
            nusc, sample_token, ax_lidar,
            view=lidar_view, box_vis_level=BoxVisibility.ANY,
            axes_limit=lidar_axes_limit, show_boxes=lidar_show_boxes,
            lidar_render_mode="scatter", lidar_cmap="viridis",
            pts_size=2.0, pts_stride=1, pts_alpha=0.9,
            box_lw=0.6, bev_extent=bev_extent
        )
        ax_lidar.set_title("LiDAR Top View", fontsize=9)
    else:
        ax_lidar.text(0.5, 0.5, "LiDAR_TOP unavailable", ha="center", va="center", fontsize=9)
    ax_lidar.axis('off'); ax_lidar.set_aspect('equal')

    if title: fig.suptitle(title, y=0.98)
    # 더 타이트하게
    # fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.04, wspace=0.06, hspace=0.12)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.05, wspace=0.02, hspace=0.08)

    if out_dir is not None:
        out_file = _ensure_outfile(out_dir, title or "unet_intermediates_all_2x7")
        plt.savefig(out_file, bbox_inches='tight', pad_inches=0.0, dpi=dpi)
    if show: plt.show()
    plt.close(fig)

    return {lab: m for lab, m in zip(names, maps)}



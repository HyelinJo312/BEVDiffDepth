# bev_triplet_nusc.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# nuScenes
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import BoxVisibility
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale


# -------------- utils --------------
def _to_numpy(x):
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)

def _slugify(s: str, maxlen: int = 80) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[|:/\\]+", " ", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9._-]", "", s)
    return s[:maxlen] or "bev_triplet"

def _ensure_outfile(out_dir, title, ext=".png"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{_slugify(title) if title else 'bev_triplet'}{ext}")

def _percentile_norm_joint(a, b, p_low=2.0, p_high=98.0, eps=1e-6):
    """Joint percentile scaling to [0,1] for fair comparison."""
    vec = np.concatenate([a.ravel(), b.ravel()])
    lo, hi = np.percentile(vec, [p_low, p_high])
    def scale(x):
        return np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0).astype(np.float32)
    return scale(a), scale(b)

def bev_extent_from_cfg(bev_cfg):
    # point_cloud_range = [x_min, y_min, z_min, x_max, y_max, z_max]
    x_min, y_min, _, x_max, y_max, _ = bev_cfg.point_cloud_range
    return (x_min, x_max, y_min, y_max)  # (xmin, xmax, ymin, ymax)

def _gaussian_blur_np(img, sigma=0.8):
    if sigma <= 0:
        return img.astype(np.float32)
    try:
        import cv2
        return cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigmaX=sigma)
    except Exception:
        # small numpy fallback (OK for 50x50)
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

# -------------- energy map (no PCA) --------------
def _aggregate_energy(feat_2d, agg="l1", whiten=True, eps=1e-6):
    """
    Args:
        feat_2d: [N, C] flattened features (N=H*W).
        agg: 'l1' | 'rms' | 'max'
        whiten: per-channel z-score across N before aggregation.
    Returns:
        energy: [N] per-location activation.
    """
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

# -------------- color for GT boxes (nuScenes colormap) --------------
def _get_color(nusc: NuScenes, category_name: str):
    """
    Map category name to nuScenes default color. Fallback to black.
    """
    # special cases seen in official helper
    if category_name == 'bicycle':
        return np.array(nusc.colormap['vehicle.bicycle']) / 255.0
    if category_name == 'construction_vehicle':
        return np.array(nusc.colormap['vehicle.construction']) / 255.0
    if category_name == 'traffic_cone':
        return np.array(nusc.colormap['movable_object.trafficcone']) / 255.0

    for key in nusc.colormap.keys():
        if category_name in key:
            return np.array(nusc.colormap[key]) / 255.0
    return np.array([0, 0, 0], dtype=np.float32)

# -------------- LiDAR_TOP pane (official style) --------------

def _draw_lidar_top_on_axes(nusc, sample_token, ax,
                            view=np.eye(4),
                            box_vis_level=BoxVisibility.ANY,
                            axes_limit=50.0,
                            show_boxes=True,
                            # NEW: point / style controls
                            lidar_render_mode="scatter",   # 'scatter' | 'height'
                            lidar_cmap="viridis",
                            pts_size=2.0,                 # smaller = thinner
                            pts_stride=1,                  # take every Nth point
                            pts_alpha=0.9,                 # 0~1
                            box_lw=0.6,                    # GT box line width
                            bev_extent=None):                   
    sample_record = nusc.get('sample', sample_token)
    assert 'LIDAR_TOP' in sample_record['data'], "No LIDAR_TOP for this sample."
    lidar_token = sample_record['data']['LIDAR_TOP']
    data_path, boxes, _ = nusc.get_sample_data(lidar_token, box_vis_level=box_vis_level)

    if lidar_render_mode == "height":
        import matplotlib.pyplot as plt
        curr = plt.get_cmap()
        plt.set_cmap(lidar_cmap)
        LidarPointCloud.from_file(data_path).render_height(ax, view=view)
        plt.set_cmap(curr)
    else:
        # Fine-grained scatter: controllable size/stride/alpha
        pc = LidarPointCloud.from_file(data_path).points  # (4, N)
        if pts_stride > 1:
            pc = pc[:, ::pts_stride]
        # apply view (homogeneous)
        P = np.vstack([pc[:3], np.ones(pc.shape[1], dtype=np.float32)])  # (4,N)
        PV = view @ P
        x, y = PV[0], PV[1]
        z = pc[2]  # color by original height
        # robust normalization for color
        zmin, zmax = np.percentile(z, [2.0, 98.0])
        z = np.clip((z - zmin) / (zmax - zmin + 1e-6), 0.0, 1.0)
        ax.scatter(x, y, c=z, s=pts_size, alpha=pts_alpha,
                   cmap=lidar_cmap, marker='.', linewidths=0, rasterized=True)
    
    if show_boxes:
        for box in boxes:
            c = _get_color(nusc, box.name)
            box.render(ax, view=view, colors=(c, c, c), linewidth=box_lw)

    # ★ apply bounds: prefer BEV extent if provided
    if bev_extent is not None:
        xmin, xmax, ymin, ymax = bev_extent
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_xlim(-axes_limit, axes_limit)
        ax.set_ylim(-axes_limit, axes_limit)

    ax.axis('off')
    ax.set_aspect('equal')


# -------------- main: triplet renderer (Activation Map)--------------
def render_bev_triplet(
    bev_pre_bchw, bev_post_bchw,
    b=0,
    nusc: NuScenes = None,
    sample_token: str = None,
    out_dir: str = None,         # directory only
    title: str = None,
    labels=("pre-UNet", "post-UNet", "LiDAR Top View"),
    # BEV params
    agg="l1", whiten=True,
    smooth_sigma=0.8, joint_clip=(2.0, 98.0), gamma=1.0,
    bev_cmap="gray", bev_interp="nearest",
    bev_extent=None,                 # ★ NEW
    bev_origin="lower",              # ★ NEW
    # LiDAR params
    lidar_axes_limit=50.0, lidar_view=np.eye(4), lidar_show_boxes=True,
    # fig
    figsize=(15, 5), dpi=240, show=False,
    signed=False,               
    signed_clip_pct=98.0          
):
    A = _to_numpy(bev_pre_bchw)
    B = _to_numpy(bev_post_bchw)
    if A.ndim != 4 or B.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W] for both, got {A.shape} and {B.shape}")

    Ba, Ca, H, W = A.shape
    Bb, Cb, H2, W2 = B.shape
    if (H, W) != (H2, W2):
        raise ValueError(f"Spatial mismatch: pre {H}x{W}, post {H2}x{W2}")
    if not (0 <= b < Ba and 0 <= b < Bb):
        raise IndexError(f"batch {b} out of range (pre B={Ba}, post B={Bb})")

    # compute energy maps
    feat_a = A[b].reshape(Ca, H*W).T
    feat_b = B[b].reshape(Cb, H*W).T
    ea = _aggregate_energy(feat_a, agg=agg, whiten=whiten).reshape(H, W)
    eb = _aggregate_energy(feat_b, agg=agg, whiten=whiten).reshape(H, W)

    ea = _gaussian_blur_np(ea, sigma=smooth_sigma)
    eb = _gaussian_blur_np(eb, sigma=smooth_sigma)
    if signed:
        v = np.percentile(np.abs(np.concatenate([ea.ravel(), eb.ravel()])),
                        signed_clip_pct)
        v = float(max(v, 1e-6))

        if gamma != 1.0:
            ea = np.sign(ea) * (np.abs(ea) ** gamma)
            eb = np.sign(eb) * (np.abs(eb) ** gamma)

        # 3) 대칭 클리핑 (부호 유지)
        ea = np.clip(ea, -v, +v)
        eb = np.clip(eb, -v, +v)
        imshow_kwargs = dict(cmap=bev_cmap, vmin=-v, vmax=+v, interpolation=bev_interp,
                            extent=bev_extent, origin=bev_origin)
    else:   
        ea, eb = _percentile_norm_joint(ea, eb, p_low=joint_clip[0], p_high=joint_clip[1])
        if gamma != 1.0:
            ea = np.power(ea, gamma).astype(np.float32)
            eb = np.power(eb, gamma).astype(np.float32)
        imshow_kwargs = dict(cmap=bev_cmap, vmin=0.0, vmax=1.0, interpolation=bev_interp,
                         extent=bev_extent, origin=bev_origin)

    # draw 1x3
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)

    axes[0].imshow(ea, **imshow_kwargs)
    axes[0].set_aspect('equal')
    if bev_extent is not None:
        axes[0].set_xlim(bev_extent[0], bev_extent[1])
        axes[0].set_ylim(bev_extent[2], bev_extent[3])
    axes[0].set_title(labels[0]); axes[0].axis('off')

    # axes[1].imshow(
    #     eb, cmap=bev_cmap, vmin=0.0, vmax=1.0, interpolation=bev_interp,
    #     extent=bev_extent, origin=bev_origin)
    axes[1].imshow(eb, **imshow_kwargs)
    axes[1].set_aspect('equal')
    if bev_extent is not None:
        axes[1].set_xlim(bev_extent[0], bev_extent[1])
        axes[1].set_ylim(bev_extent[2], bev_extent[3])
    axes[1].set_title(labels[1]); axes[1].axis('off')

    if (nusc is not None) and (sample_token is not None):
        _draw_lidar_top_on_axes(
                                nusc, sample_token, axes[2],
                                view=lidar_view,
                                box_vis_level=BoxVisibility.ANY,
                                axes_limit=lidar_axes_limit,
                                show_boxes=lidar_show_boxes,
                                lidar_render_mode="scatter",
                                lidar_cmap="viridis",
                                pts_size=2.0,       # ↓ 더 작게
                                pts_stride=1,        # ↓ 다운샘플
                                pts_alpha=0.9,
                                box_lw=0.6,
                                bev_extent=bev_extent
                            )
    else:
        axes[2].text(0.5, 0.5, "LiDAR_TOP unavailable", ha="center", va="center", fontsize=10)
        axes[2].axis('off'); axes[2].set_aspect('equal')
    axes[2].set_title(labels[2])

    if title:
        fig.suptitle(title, y=0.99)

    if out_dir is not None:
        out_file = _ensure_outfile(out_dir, title or "bev_triplet")
        plt.savefig(out_file, bbox_inches='tight', pad_inches=0.0, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)

    return ea, eb



# ---------- main: RGB PCA triplet ----------
def visualize_bev_rgb_pca_triplet(
    bev_pre_bchw,
    bev_post_bchw,
    b: int = 0,
    bev_extent=None,
    origin: str = "lower",
    whiten: bool = True,
    eps: float = 1e-6,
    stride: int = 1,
    joint_clip: tuple = (0.5, 99.5),
    gamma: float = 1.0,
    # 부드럽게 만들기 옵션
    upsample: int = 3,           # 업샘플 배율
    ssaa: bool = True,           # 초해상 후 다운샘플
    blur_sigma: float = 0.8,     # 가우시안 블러
    edge_preserve: str = "bilateral",  # "none"|"bilateral"|"guided"
    interp: str = "bicubic",     # imshow 보간
    # 출력
    out_dir: str = None,
    title: str = "BEV features: RGB-PCA (pre vs post)",
    dpi: int = 300,
    show: bool = False,
    # LiDAR
    nusc: NuScenes = None,
    sample_token: str = None,
):

    A = _to_numpy(bev_pre_bchw)
    B = _to_numpy(bev_post_bchw)
    _, C, H, W = A.shape
    pre = A[b].reshape(C, H*W).T
    post = B[b].reshape(C, H*W).T

    # 통합 표준화
    if whiten:
        concat = np.vstack([pre, post])
        mean = concat.mean(axis=0, keepdims=True)
        std  = concat.std(axis=0, keepdims=True) + eps
        pre_z, post_z = (pre-mean)/std, (post-mean)/std
    else:
        pre_z, post_z = pre, post

    # PCA
    pca = PCA(n_components=3, random_state=0)
    pca.fit(np.vstack([pre_z, post_z]))
    pre_rgb_flat  = pca.transform(pre_z)
    post_rgb_flat = pca.transform(post_z)

    pre_rgb  = pre_rgb_flat.reshape(H, W, 3)
    post_rgb = post_rgb_flat.reshape(H, W, 3)

    both = np.concatenate([pre_rgb.reshape(-1, 3), post_rgb.reshape(-1, 3)], axis=0)
    # lo, hi = np.percentile(both, joint_clip[0], axis=0), np.percentile(both, joint_clip[1], axis=0)
    lo = np.percentile(both, 0.5, axis=0)
    hi = np.percentile(both, 99.5, axis=0)
    scale = (hi - lo + 1e-6)
    def _scale_img(img):
        x = (img - lo) / scale
        x = np.clip(x, 0.0, 1.0)
        if gamma != 1.0:
            x = np.power(x, gamma)
        return x.astype(np.float32)
    pre_rgb, post_rgb = _scale_img(pre_rgb), _scale_img(post_rgb)

    # ---------- smoothing pipeline ----------
    import cv2
    def _smooth_rgb(img):
        h, w = img.shape[:2]
        if upsample and upsample > 1:
            img = cv2.resize(img, (w*upsample, h*upsample), interpolation=cv2.INTER_CUBIC)
        if edge_preserve == "bilateral":
            for c in range(3):
                img[..., c] = cv2.bilateralFilter((img[..., c]*255).astype(np.uint8),
                                                  d=0, sigmaColor=30, sigmaSpace=3)/255.0
        if blur_sigma and blur_sigma > 0:
            img = cv2.GaussianBlur(img, (0,0), blur_sigma)
        if ssaa and upsample and upsample > 1:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return np.clip(img, 0, 1)

    pre_rgb_s, post_rgb_s = _smooth_rgb(pre_rgb), _smooth_rgb(post_rgb)

    # ---------- Figure ----------
    fig = plt.figure(figsize=(15, 5), dpi=dpi)
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(pre_rgb_s, extent=bev_extent, origin=origin, interpolation=interp)
    ax1.set_title("pre-UNet (RGB-PCA)")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(post_rgb_s, extent=bev_extent, origin=origin, interpolation=interp)
    ax2.set_title("post-UNet (RGB-PCA)")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    if (nusc is not None) and (sample_token is not None):
        _draw_lidar_top_on_axes(nusc, sample_token, ax3, bev_extent=bev_extent)
        ax3.set_title("LiDAR TOP view")
    else:
        ax3.text(0.5, 0.5, "LiDAR_TOP unavailable",
                 ha="center", va="center", fontsize=10)
        ax3.axis("off")

    fig.suptitle(title, y=0.98)

    if out_dir is not None:
        out_file = _ensure_outfile(out_dir, title, ext=".png")
        plt.savefig(out_file, bbox_inches='tight', pad_inches=0.0, dpi=dpi)

    if show:
        plt.show()
    plt.close(fig)

    evr = pca.explained_variance_ratio_
    return pre_rgb_s, post_rgb_s, pca



# ---------- main: RGB PCA triplet (DINOv2 visualization style) ----------
def _fit_common_pca_and_scale(
    bev_o_chw, bev_d_chw,
    n_components=3,
    clip_percentile=(0.5, 99.5),
    gamma=1.0,
    whiten=False,
    eps=1e-6,
):
    """
    두 맵을 합쳐 공통 정규화(mean/std) + 공통 PCA를 fit.
    이후 두 결과를 동일한 클립/스케일로 [0,1] RGB로 변환.
    반환: rgb_o(H,W,3), rgb_d(H,W,3), pca(객체), stats(dict)
    """
    Xo = _to_numpy(bev_o_chw)  # (C,H,W)
    Xd = _to_numpy(bev_d_chw)
    C, H, W = Xo.shape
    Xo = Xo.reshape(C, -1).T   # (N1, C)
    Xd = Xd.reshape(C, -1).T   # (N2, C)

    # --- 공통 표준화(두 맵 합쳐서 mean/std) ---
    Xcat = np.concatenate([Xo, Xd], axis=0)  # (N1+N2, C)
    mean = Xcat.mean(axis=0, keepdims=True)
    std = Xcat.std(axis=0, keepdims=True)
    std = np.where(std < eps, eps, std)

    Xo_n = (Xo - mean) / std
    Xd_n = (Xd - mean) / std
    Xcat_n = np.concatenate([Xo_n, Xd_n], axis=0)

    # --- 공통 PCA ---
    pca = PCA(n_components=n_components, whiten=whiten)
    Ycat = pca.fit_transform(Xcat_n)   # (N1+N2, 3)

    # --- 성분 부호 고정(프레임 간 색 뒤집힘 방지) ---
    # 기준: 각 성분의 로딩 벡터 합이 음수면 플립
    flips = np.sign(np.sum(pca.components_, axis=1))
    flips[flips == 0] = 1.0
    Ycat = Ycat * flips  # (N,3)
    components_fixed = pca.components_ * flips[:, None]

    # --- 강건 스케일링: 두 맵을 합친 공간에서 퍼센타일 클립 후 [0,1] ---
    lo, hi = np.percentile(Ycat, clip_percentile, axis=0)
    Ycat_c = np.clip(Ycat, lo, hi)
    Ycat_s = minmax_scale(Ycat_c)  # 컬럼별 [0,1]

    if gamma != 1.0:
        Ycat_s = np.power(Ycat_s, gamma)

    # --- 다시 분할 ---
    N1 = Xo.shape[0]
    Yo = Ycat_s[:N1].reshape(H, W, 3).astype(np.float32)
    Yd = Ycat_s[N1:].reshape(H, W, 3).astype(np.float32)

    stats = {
        "mean": mean,
        "std": std,
        "components": components_fixed,
        "flips": flips,
        "clip_percentile": clip_percentile,
        "gamma": gamma,
    }
    return Yo, Yd, pca, stats


def visualize_bev_pca_and_lidar(
    bev_orig_bchw,
    bev_denoised_bchw,
    b=0,
    out_dir=None,
    title=None,
    show=False,
    nusc=None,
    sample_token=None,
    bev_extent=None,         # (xmin, xmax, ymin, ymax)  좌표계가 맞으면 LiDAR와 범위 정렬
    origin="lower",   # imshow용
    cmap_lidar="viridis",
    lidar_kwargs=None,
    titles: tuple = ("Original BEV (PCA-RGB)", "Denoised BEV (PCA-RGB)", "LiDAR_TOP"),
    pca_whiten: bool = False,
    pca_clip=(0.5, 99.5),
    pca_gamma=1.0,
    figsize=(15, 5),
    dpi=300,
    interpolation="bicubic"
):
    """
    bev_orig_bchw, bev_denoised_bchw: (B, C, H, W)
    nusc, sample_token: nuScenes 핸들
    bev_extent: 이미지 extent/라이다 범위를 동일하게 맞추고 싶을 때 사용
    lidar_kwargs: _draw_lidar_top_on_axes 에 그대로 전달할 추가 인자 dict
    """
    lidar_kwargs = lidar_kwargs or {}

    # 선택한 배치만 CHW로 꺼냄
    bev_o = _to_numpy(bev_orig_bchw)[b]       # (C,H,W)
    bev_d = _to_numpy(bev_denoised_bchw)[b]   # (C,H,W)
    _, H, W = bev_o.shape

    # PCA-RGB
    rgb_o, rgb_d, pca, stats = _fit_common_pca_and_scale(
        bev_o, bev_d,
        n_components=3,
        clip_percentile=pca_clip,
        gamma=pca_gamma,
        whiten=pca_whiten,
    )

    # ---- Figure ----
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    # Left: Original
    ax = axes[0]
    ax.set_title(titles[0])
    if bev_extent is not None:
        extent = (bev_extent[0], bev_extent[1], bev_extent[2], bev_extent[3])
        ax.imshow(rgb_o, origin=origin, extent=extent, interpolation=interpolation)
        ax.set_xlim(bev_extent[0], bev_extent[1])
        ax.set_ylim(bev_extent[2], bev_extent[3])
        ax.set_aspect('equal')
    else:
        ax.imshow(rgb_o, origin=origin, interpolation=interpolation)
        ax.set_aspect('equal')
    ax.axis('off')

    # Middle: Denoised
    ax = axes[1]
    ax.set_title(titles[1])
    if bev_extent is not None:
        extent = (bev_extent[0], bev_extent[1], bev_extent[2], bev_extent[3])
        ax.imshow(rgb_d, origin=origin, extent=extent, interpolation=interpolation)
        ax.set_xlim(bev_extent[0], bev_extent[1])
        ax.set_ylim(bev_extent[2], bev_extent[3])
        ax.set_aspect('equal')
    else:
        ax.imshow(rgb_d, origin=origin, interpolation=interpolation)
        ax.set_aspect('equal')
    ax.axis('off')
    
    # ---- Right: LiDAR_TOP ----
    ax = axes[2]
    ax.set_title(titles[2])

    default_lidar_kwargs = dict(
        view=np.eye(4),
        box_vis_level=BoxVisibility.ANY,    # BoxVisibility.ANY를 쓰려면 import해서 넘기세요
        axes_limit=50.0,
        show_boxes=True,
        lidar_render_mode="scatter",
        lidar_cmap=cmap_lidar,
        pts_size=2.0,
        pts_stride=1,
        pts_alpha=0.9,
        box_lw=0.6,
        bev_extent=bev_extent
    )

    default_lidar_kwargs.update(lidar_kwargs)
    _draw_lidar_top_on_axes(nusc, sample_token, ax, **default_lidar_kwargs)


    if out_dir is not None:
        out_file = _ensure_outfile(out_dir, title, ext=".png")
        plt.savefig(out_file, bbox_inches='tight', pad_inches=0.0, dpi=dpi)

    if show:
        plt.show()
    plt.close(fig)

    return fig, axes, pca, stats



def _fit_common_pca_and_scale_v2(
    bev_t0, bev_t10, bev_t100, bev_t1000,
    n_components=3,
    clip_percentile=(0.5, 99.5),
    gamma=1.0,
    whiten=False,
    eps=1e-6,
):
    """
    두 맵을 합쳐 공통 정규화(mean/std) + 공통 PCA를 fit.
    이후 두 결과를 동일한 클립/스케일로 [0,1] RGB로 변환.
    반환: rgb_o(H,W,3), rgb_d(H,W,3), pca(객체), stats(dict)
    """
    x1 = _to_numpy(bev_t0)  # (C,H,W)
    x2 = _to_numpy(bev_t10)
    x3 = _to_numpy(bev_t100)
    x4 = _to_numpy(bev_t1000)
    
    C, H, W = x1.shape
    x1 = x1.reshape(C, -1).T   # (N1, C)
    x2 = x2.reshape(C, -1).T   # (N2, C)
    x3 = x3.reshape(C, -1).T   # (N2, C)
    x4 = x4.reshape(C, -1).T   # (N2, C)

    # --- 공통 표준화(두 맵 합쳐서 mean/std) ---
    Xcat = np.concatenate([x1, x2, x3, x4], axis=0)  # (N1+N2, C)
    mean = Xcat.mean(axis=0, keepdims=True)
    std = Xcat.std(axis=0, keepdims=True)
    std = np.where(std < eps, eps, std)

    x1_n = (x1 - mean) / std
    x2_n = (x2 - mean) / std
    x3_n = (x3 - mean) / std
    x4_n = (x4 - mean) / std
    Xcat_n = np.concatenate([x1_n, x2_n, x3_n, x4_n], axis=0)

    # --- 공통 PCA ---
    pca = PCA(n_components=n_components, whiten=whiten)
    Ycat = pca.fit_transform(Xcat_n)   # (N1+N2, 3)

    # --- 성분 부호 고정(프레임 간 색 뒤집힘 방지) ---
    # 기준: 각 성분의 로딩 벡터 합이 음수면 플립
    flips = np.sign(np.sum(pca.components_, axis=1))
    flips[flips == 0] = 1.0
    Ycat = Ycat * flips  # (N,3)
    components_fixed = pca.components_ * flips[:, None]

    # --- 강건 스케일링: 두 맵을 합친 공간에서 퍼센타일 클립 후 [0,1] ---
    lo, hi = np.percentile(Ycat, clip_percentile, axis=0)
    Ycat_c = np.clip(Ycat, lo, hi)
    Ycat_s = minmax_scale(Ycat_c)  # 컬럼별 [0,1]

    if gamma != 1.0:
        Ycat_s = np.power(Ycat_s, gamma)

    # --- 다시 분할 ---
    N1 = x1.shape[0]
    y1 = Ycat_s[:N1].reshape(H, W, 3).astype(np.float32)
    y2 = Ycat_s[N1:N1*2].reshape(H, W, 3).astype(np.float32)
    y3 = Ycat_s[N1*2:N1*3].reshape(H, W, 3).astype(np.float32)
    y4 = Ycat_s[N1*3:].reshape(H, W, 3).astype(np.float32)
    
    stats = {
        "mean": mean,
        "std": std,
        "components": components_fixed,
        "flips": flips,
        "clip_percentile": clip_percentile,
        "gamma": gamma,
    }
    return y1, y2, y3, y4, pca, stats


def visualize_bev_pca_and_lidar_timestep(
    bev_t0,
    bev_t10,
    bev_t100,
    bev_t1000,
    b=0,
    out_dir=None,
    title=None,
    show=False,
    nusc=None,
    sample_token=None,
    bev_extent=None,         # (xmin, xmax, ymin, ymax)  좌표계가 맞으면 LiDAR와 범위 정렬
    origin="lower",   # imshow용
    cmap_lidar="viridis",
    lidar_kwargs=None,
    titles: tuple = ("Original BEV (PCA-RGB)", "Denoised BEV (PCA-RGB)", "LiDAR_TOP"),
    pca_whiten: bool = False,
    pca_clip=(0.5, 99.5),
    pca_gamma=1.0,
    figsize=(15, 5),
    dpi=300,
    interpolation="bicubic"
):
    """
    bev_orig_bchw, bev_denoised_bchw: (B, C, H, W)
    nusc, sample_token: nuScenes 핸들
    bev_extent: 이미지 extent/라이다 범위를 동일하게 맞추고 싶을 때 사용
    lidar_kwargs: _draw_lidar_top_on_axes 에 그대로 전달할 추가 인자 dict
    """
    lidar_kwargs = lidar_kwargs or {}

    # 선택한 배치만 CHW로 꺼냄
    bev_t0 = _to_numpy(bev_t0)[b]       # (C,H,W)
    bev_t101 = _to_numpy(bev_t10)[b]   # (C,H,W)
    bev_t100 = _to_numpy(bev_t100)[b]   # (C,H,W)
    bev_t1000 = _to_numpy(bev_t1000)[b]   # (C,H,W)
    _, H, W = bev_t0.shape

    # PCA-RGB
    rgb_1, rgb_2, rgb_3, rgb_4, pca, stats = _fit_common_pca_and_scale_v2(
        bev_t0, bev_t10, bev_t100, bev_t1000,
        n_components=3,
        clip_percentile=pca_clip,
        gamma=pca_gamma,
        whiten=pca_whiten,
    )

    # ---- Figure ----
    fig, axes = plt.subplots(1, 5, figsize=figsize, constrained_layout=True)

    # T=0
    ax = axes[0]
    ax.set_title(titles[0])
    if bev_extent is not None:
        extent = (bev_extent[0], bev_extent[1], bev_extent[2], bev_extent[3])
        ax.imshow(rgb_1, origin=origin, extent=extent, interpolation=interpolation)
        ax.set_xlim(bev_extent[0], bev_extent[1])
        ax.set_ylim(bev_extent[2], bev_extent[3])
        ax.set_aspect('equal')
    else:
        ax.imshow(rgb_1, origin=origin, interpolation=interpolation)
        ax.set_aspect('equal')
    ax.axis('off')

    # T=10
    ax = axes[1]
    ax.set_title(titles[1])
    if bev_extent is not None:
        extent = (bev_extent[0], bev_extent[1], bev_extent[2], bev_extent[3])
        ax.imshow(rgb_2, origin=origin, extent=extent, interpolation=interpolation)
        ax.set_xlim(bev_extent[0], bev_extent[1])
        ax.set_ylim(bev_extent[2], bev_extent[3])
        ax.set_aspect('equal')
    else:
        ax.imshow(rgb_2, origin=origin, interpolation=interpolation)
        ax.set_aspect('equal')
    ax.axis('off')
    
    # T=100
    ax = axes[2]
    ax.set_title(titles[2])
    if bev_extent is not None:
        extent = (bev_extent[0], bev_extent[1], bev_extent[2], bev_extent[3])
        ax.imshow(rgb_2, origin=origin, extent=extent, interpolation=interpolation)
        ax.set_xlim(bev_extent[0], bev_extent[1])
        ax.set_ylim(bev_extent[2], bev_extent[3])
        ax.set_aspect('equal')
    else:
        ax.imshow(rgb_2, origin=origin, interpolation=interpolation)
        ax.set_aspect('equal')
    ax.axis('off')
    
    # T=1000
    ax = axes[3]
    ax.set_title(titles[3])
    if bev_extent is not None:
        extent = (bev_extent[0], bev_extent[1], bev_extent[2], bev_extent[3])
        ax.imshow(rgb_3, origin=origin, extent=extent, interpolation=interpolation)
        ax.set_xlim(bev_extent[0], bev_extent[1])
        ax.set_ylim(bev_extent[2], bev_extent[3])
        ax.set_aspect('equal')
    else:
        ax.imshow(rgb_3, origin=origin, interpolation=interpolation)
        ax.set_aspect('equal')
    ax.axis('off')
    
    # ---- Right: LiDAR_TOP ----
    ax = axes[4]
    ax.set_title(titles[4])

    default_lidar_kwargs = dict(
        view=np.eye(4),
        box_vis_level=BoxVisibility.ANY,    # BoxVisibility.ANY를 쓰려면 import해서 넘기세요
        axes_limit=50.0,
        show_boxes=True,
        lidar_render_mode="scatter",
        lidar_cmap=cmap_lidar,
        pts_size=2.0,
        pts_stride=1,
        pts_alpha=0.9,
        box_lw=0.6,
        bev_extent=bev_extent
    )

    default_lidar_kwargs.update(lidar_kwargs)
    _draw_lidar_top_on_axes(nusc, sample_token, ax, **default_lidar_kwargs)


    if out_dir is not None:
        out_file = _ensure_outfile(out_dir, title, ext=".png")
        plt.savefig(out_file, bbox_inches='tight', pad_inches=0.0, dpi=dpi)

    if show:
        plt.show()
    plt.close(fig)

    return fig, axes, pca, stats




def _fit_single_pca_and_scale(
    bev_chw,
    n_components=3,
    clip_percentile=(0.5, 99.5),
    gamma=1.0,
    whiten=False,
    eps=1e-6,
):
    """
    단일 BEV (C,H,W)에 대해 표준화 -> PCA(3성분) -> 강건 클립 -> [0,1] 스케일 -> gamma 보정.
    반환: rgb(H,W,3), pca, stats(dict)
    """
    X = _to_numpy(bev_chw)           # (C,H,W)
    C, H, W = X.shape
    Xf = X.reshape(C, -1).T          # (N, C)

    # 표준화
    mean = Xf.mean(axis=0, keepdims=True)
    std  = Xf.std(axis=0, keepdims=True)
    std  = np.where(std < eps, eps, std)
    Xn   = (Xf - mean) / std

    # PCA
    pca = PCA(n_components=n_components, whiten=whiten)
    Y   = pca.fit_transform(Xn)      # (N, 3)

    # 성분 부호 고정 (색 뒤집힘 방지용)
    flips = np.sign(np.sum(pca.components_, axis=1))
    flips[flips == 0] = 1.0
    Y = Y * flips
    components_fixed = pca.components_ * flips[:, None]

    # 강건 클리핑 + [0,1] 스케일
    lo, hi = np.percentile(Y, clip_percentile, axis=0)
    Yc = np.clip(Y, lo, hi)
    Ys = minmax_scale(Yc)            # 각 컬럼별 [0,1]

    if gamma != 1.0:
        Ys = np.power(Ys, gamma)

    rgb = Ys.reshape(H, W, 3).astype(np.float32)

    stats = {
        "mean": mean,
        "std": std,
        "components": components_fixed,
        "flips": flips,
        "clip_percentile": clip_percentile,
        "gamma": gamma,
    }
    return rgb, pca, stats


def visualize_single_bev_pca_and_lidar(
    bev_bchw,                 # (B,C,H,W)
    b=0,
    out_dir=None,
    title=None,
    show=False,
    nusc=None,
    sample_token=None,
    bev_extent=None,          # (xmin, xmax, ymin, ymax)
    origin="lower",
    cmap_lidar="viridis",
    lidar_kwargs=None,
    titles: tuple = ("BEV (PCA-RGB)", "LiDAR_TOP"),
    pca_whiten: bool = False,
    pca_clip=(0.5, 99.5),
    pca_gamma=1.2,
    figsize=(10, 5),
    dpi=300,
    interpolation="bicubic"
):
    """
    단일 BEV feature + LiDAR TOP 시각화 (2-패널).
    - 왼쪽: BEV를 PCA-RGB로 시각화
    - 오른쪽: LiDAR_TOP (nuScenes 핸들 필요)
    """
    lidar_kwargs = lidar_kwargs or {}

    # 배치 선택 후 CHW
    bev_chw = _to_numpy(bev_bchw)[b]    # (C,H,W)

    # BEV → PCA-RGB
    rgb, pca, stats = _fit_single_pca_and_scale(
        bev_chw,
        n_components=3,
        clip_percentile=pca_clip,
        gamma=pca_gamma,
        whiten=pca_whiten,
    )

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # Left: BEV PCA-RGB
    ax = axes[0]
    ax.set_title(titles[0])
    if bev_extent is not None:
        extent = (bev_extent[0], bev_extent[1], bev_extent[2], bev_extent[3])
        ax.imshow(rgb, origin=origin, extent=extent, interpolation=interpolation)
        ax.set_xlim(bev_extent[0], bev_extent[1])
        ax.set_ylim(bev_extent[2], bev_extent[3])
        ax.set_aspect('equal')
    else:
        ax.imshow(rgb, origin=origin, interpolation=interpolation)
        ax.set_aspect('equal')
    ax.axis('off')

    # Right: LiDAR_TOP
    ax = axes[1]
    ax.set_title(titles[1])

    default_lidar_kwargs = dict(
        view=np.eye(4),
        box_vis_level=BoxVisibility.ANY,  # 필요시 import하여 사용
        axes_limit=50.0,
        show_boxes=True,
        lidar_render_mode="scatter",
        lidar_cmap=cmap_lidar,
        pts_size=2.0,
        pts_stride=1,
        pts_alpha=0.9,
        box_lw=0.6,
        bev_extent=bev_extent
    )
    default_lidar_kwargs.update(lidar_kwargs)

    _draw_lidar_top_on_axes(nusc, sample_token, ax, **default_lidar_kwargs)

    # 파일 저장
    if out_dir is not None:
        out_file = _ensure_outfile(out_dir, title, ext=".png")
        plt.savefig(out_file, bbox_inches='tight', pad_inches=0.0, dpi=dpi)

    if show:
        plt.show()
    plt.close(fig)


    return fig, axes, pca, stats
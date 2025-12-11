import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import BoxVisibility

# ---------- utils ----------
def _to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().float().numpy()
    except Exception:
        pass
    return np.asarray(x, dtype=np.float32)

def _slugify(s: str, maxlen: int = 80) -> str:
    import re
    s = (s or "").strip().lower()
    s = re.sub(r"[|:/\\]+", " ", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9._-]", "", s)
    return s[:maxlen] or "bev_rgb_pca"

def _ensure_outfile(out_dir, title, ext=".png"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{_slugify(title) if title else 'bev_rgb_pca'}{ext}")

def bev_extent_from_cfg(bev_cfg):
    # point_cloud_range = [x_min, y_min, z_min, x_max, y_max, z_max]
    x_min, y_min, _, x_max, y_max, _ = bev_cfg.point_cloud_range
    return (x_min, x_max, y_min, y_max)

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
    """
    좌: pre RGB-PCA, 중: post RGB-PCA, 우: LiDAR TOP (extent 일치).
    RGB-PCA를 부드럽게 시각화하기 위한 업샘플링/스무딩 옵션 포함.
    """
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

    # 공동 스케일링
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
        print(f"[Saved] {out_file}")

    if show:
        plt.show()
    plt.close(fig)

    evr = pca.explained_variance_ratio_
    return pre_rgb_s, post_rgb_s, pca

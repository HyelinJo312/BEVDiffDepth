    """
BEV (original / denoised) + nuScenes LIDAR_TOP triplet visualization (activation maps)

- BEV: activation/energy map from [B,C,H,W] feature tensor
- LiDAR: LIDAR_TOP point cloud rendered in *flat ego* frame (yaw-only), matching BEV grid
- Axes: use point_cloud_range -> extent (xmin,xmax,ymin,ymax)
- Saves a 1x3 figure: original BEV / denoised BEV / LiDAR_TOP

Requirements:
  pip install nuscenes-devkit pyquaternion opencv-python matplotlib

Usage example:
  extent = bev_extent_from_point_cloud_range([-51.2,-51.2,-5.0, 51.2,51.2,3.0])
  render_bev_triplet_activation(
      bev_pre_bchw=original_bev, bev_post_bchw=denoised_bev,
      sample_token=sample_token, nusc=nusc,
      out_path="out/triplet.png",
      bev_extent=extent,
      bev_orient="T",           # try: "T", "T_flip_y", "none", ...
      signed=False,
      agg="l1", whiten=True
  )
"""

from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Optional, Tuple, Union, Literal, Dict

import numpy as np
import matplotlib.pyplot as plt

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud


# ---------------------------
# Basic utilities
# ---------------------------
def to_numpy(x):
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def slugify(s: str, maxlen: int = 120) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[|:/\\]+", " ", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9._-]", "", s)
    return (s[:maxlen] or "bev_triplet")


def ensure_parent(path: Union[str, Path]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def bev_extent_from_point_cloud_range(point_cloud_range) -> Tuple[float, float, float, float]:
    # [x_min, y_min, z_min, x_max, y_max, z_max]
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    return (float(x_min), float(x_max), float(y_min), float(y_max))


# ---------------------------
# BEV activation map
# ---------------------------
def gaussian_blur_np(img: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    img = img.astype(np.float32)
    if sigma <= 0:
        return img
    if _HAS_CV2:
        return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    # tiny fallback (slow but OK for 128x128)
    k = int(max(3, 2 * round(3 * sigma) + 1))
    x = np.arange(k) - (k - 1) / 2
    g = np.exp(-(x**2) / (2 * sigma * sigma)).astype(np.float32)
    g /= g.sum()
    K = np.outer(g, g)
    pad = k // 2
    gp = np.pad(img, ((pad, pad), (pad, pad)), mode="reflect")
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = float((gp[i:i+k, j:j+k] * K).sum())
    return out


def aggregate_energy(feat_2d: np.ndarray, agg: str = "l1", whiten: bool = True, eps: float = 1e-6) -> np.ndarray:
    """
    feat_2d: [N, C] (N=H*W)
    returns: [N]
    """
    z = feat_2d
    if whiten:
        m = z.mean(axis=0, keepdims=True)
        s = z.std(axis=0, keepdims=True) + eps
        z = (z - m) / s

    if agg == "l1":
        e = np.mean(np.abs(z), axis=1)
    elif agg == "rms":
        e = np.sqrt(np.mean(z**2, axis=1))
    elif agg == "max":
        e = np.max(np.abs(z), axis=1)
    elif agg == "signed_mean":
        e = np.mean(z, axis=1)
    else:
        raise ValueError(f"Unknown agg={agg}")
    return e.astype(np.float32)


def percentile_norm_joint(a: np.ndarray, b: np.ndarray, p_low=2.0, p_high=98.0, eps=1e-6) -> Tuple[np.ndarray, np.ndarray]:
    vec = np.concatenate([a.ravel(), b.ravel()])
    lo, hi = np.percentile(vec, [p_low, p_high])
    def scale(x):
        return np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0).astype(np.float32)
    return scale(a), scale(b)


BEVOrient = Literal[
    "none",
    "T",
    "flip_x",     # reverse columns
    "flip_y",     # reverse rows
    "T_flip_x",
    "T_flip_y",
    "rot90",
    "rot180",
    "rot270",
]


def fix_bev_orientation(e: np.ndarray, mode: BEVOrient) -> np.ndarray:
    if mode == "none":
        return e
    if mode == "T":
        return e.T
    if mode == "flip_x":
        return e[:, ::-1]
    if mode == "flip_y":
        return e[::-1, :]
    if mode == "T_flip_x":
        return e.T[:, ::-1]
    if mode == "T_flip_y":
        return e.T[::-1, :]
    if mode == "rot90":
        return np.rot90(e, 1)
    if mode == "rot180":
        return np.rot90(e, 2)
    if mode == "rot270":
        return np.rot90(e, 3)
    raise ValueError(mode)


def bev_activation_map(
    bev_bchw: Union[np.ndarray, "torch.Tensor"],
    b: int = 0,
    agg: str = "l1",
    whiten: bool = True,
    smooth_sigma: float = 0.8,
    signed: bool = False,
    signed_clip_pct: float = 98.0,
    gamma: float = 1.0,
    joint_clip: Tuple[float, float] = (2.0, 98.0),
    orient: BEVOrient = "none",
    ref_for_joint: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Returns: 2D map (H,W) either [0,1] (unsigned) or [-v,+v] (signed clipped)
    If ref_for_joint is provided (another map), joint percentile normalization is applied outside.
    """
    A = to_numpy(bev_bchw)
    assert A.ndim == 4, f"Expected [B,C,H,W], got {A.shape}"
    B, C, H, W = A.shape
    assert 0 <= b < B

    feat = A[b].reshape(C, H * W).T  # [N,C]
    e = aggregate_energy(feat, agg=agg, whiten=whiten).reshape(H, W)
    e = fix_bev_orientation(e, orient)
    e = gaussian_blur_np(e, sigma=smooth_sigma)

    if signed:
        v = np.percentile(np.abs(e.ravel()), signed_clip_pct)
        v = float(max(v, 1e-6))
        if gamma != 1.0:
            e = np.sign(e) * (np.abs(e) ** gamma)
        e = np.clip(e, -v, +v).astype(np.float32)
        return e
    else:
        # normalization done later jointly if desired
        if gamma != 1.0:
            e = np.power(np.clip(e, 0, None), gamma).astype(np.float32)
        return e.astype(np.float32)


# ---------------------------
# LiDAR (flat ego) rendering
# ---------------------------
def lidar_points_flat_ego(nusc: NuScenes, lidar_sd_token: str) -> np.ndarray:
    """
    Returns points in flat ego frame (yaw-only), shape (3,N) for x,y,z.
    """
    sd = nusc.get("sample_data", lidar_sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])

    # load in sensor frame
    path = nusc.get_sample_data_path(lidar_sd_token)
    pc = LidarPointCloud.from_file(path)  # points: (4,N), [x,y,z,1] in sensor

    # sensor -> ego (full)
    pc.rotate(Quaternion(cs["rotation"]).rotation_matrix)
    pc.translate(np.array(cs["translation"]))

    # ego -> global (full)
    pc.rotate(Quaternion(pose["rotation"]).rotation_matrix)
    pc.translate(np.array(pose["translation"]))

    # global -> flat ego (yaw only) at that pose
    q = Quaternion(pose["rotation"])
    yaw = q.yaw_pitch_roll[0]
    R_yaw = Quaternion(axis=[0, 0, 1], angle=yaw).rotation_matrix
    t = np.array(pose["translation"]).reshape(3, 1)

    pts = pc.points[:3]  # (3,N) in global
    pts = pts - t
    pts = R_yaw.T @ pts  # back to flat ego

    return pts.astype(np.float32)


def draw_lidar_top(
    nusc: NuScenes,
    sample_token: str,
    ax: plt.Axes,
    bev_extent: Optional[Tuple[float, float, float, float]] = None,
    axes_limit: float = 50.0,
    stride: int = 1,
    s: float = 1.0,
    alpha: float = 0.9,
    cmap: str = "viridis",
):
    sample = nusc.get("sample", sample_token)
    lidar_sd_token = sample["data"]["LIDAR_TOP"]

    pts = lidar_points_flat_ego(nusc, lidar_sd_token)  # (3,N)
    if stride > 1:
        pts = pts[:, ::stride]
    x, y, z = pts[0], pts[1], pts[2]

    # color normalize by height
    zmin, zmax = np.percentile(z, [2.0, 98.0])
    zn = np.clip((z - zmin) / (zmax - zmin + 1e-6), 0.0, 1.0)

    ax.scatter(x, y, c=zn, s=s, alpha=alpha, cmap=cmap,
               marker=".", linewidths=0, rasterized=True)

    ax.set_aspect("equal")
    if bev_extent is not None:
        xmin, xmax, ymin, ymax = bev_extent
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_xlim(-axes_limit, axes_limit)
        ax.set_ylim(-axes_limit, axes_limit)
    ax.axis("off")


# ---------------------------
# Main: Triplet renderer
# ---------------------------
def render_bev_triplet_activation(
    bev_pre_bchw: Union[np.ndarray, "torch.Tensor"],
    bev_post_bchw: Union[np.ndarray, "torch.Tensor"],
    sample_token: str,
    nusc: NuScenes,
    out_path: Optional[str] = None,
    *,
    b: int = 0,
    bev_extent: Optional[Tuple[float, float, float, float]] = None,
    bev_origin: str = "lower",
    bev_interp: str = "bilinear",
    bev_cmap: str = "gray",
    lidar_cmap: str = "viridis",
    title: Optional[str] = None,
    labels: Tuple[str, str, str] = ("original BEV", "denoised BEV", "LiDAR_TOP"),
    # BEV activation params
    agg: str = "l1",
    whiten: bool = True,
    smooth_sigma: float = 0.8,
    joint_clip: Tuple[float, float] = (2.0, 98.0),
    gamma: float = 1.0,
    signed: bool = False,
    signed_clip_pct: float = 98.0,
    bev_orient: BEVOrient = "T",
    # LiDAR render params
    lidar_axes_limit: float = 50.0,
    lidar_stride: int = 1,
    lidar_pt_size: float = 1.0,
    lidar_alpha: float = 0.9,
    # fig
    figsize: Tuple[float, float] = (15, 5),
    dpi: int = 240,
    show: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Returns dict with 'ea', 'eb' (final displayed maps).
    """
    A = to_numpy(bev_pre_bchw)
    B = to_numpy(bev_post_bchw)
    assert A.ndim == 4 and B.ndim == 4, f"Need [B,C,H,W], got {A.shape}, {B.shape}"
    assert A.shape[0] > b and B.shape[0] > b, f"batch b={b} out of range"

    ea = bev_activation_map(
        A, b=b, agg=agg, whiten=whiten, smooth_sigma=smooth_sigma,
        signed=signed, signed_clip_pct=signed_clip_pct,
        gamma=gamma, joint_clip=joint_clip, orient=bev_orient
    )
    eb = bev_activation_map(
        B, b=b, agg=agg, whiten=whiten, smooth_sigma=smooth_sigma,
        signed=signed, signed_clip_pct=signed_clip_pct,
        gamma=gamma, joint_clip=joint_clip, orient=bev_orient
    )

    if signed:
        # symmetric clip per-map already applied; unify range for visualization
        v = np.percentile(np.abs(np.concatenate([ea.ravel(), eb.ravel()])), signed_clip_pct)
        v = float(max(v, 1e-6))
        imshow_kwargs = dict(cmap=bev_cmap, vmin=-v, vmax=+v, interpolation=bev_interp,
                             extent=bev_extent, origin=bev_origin)
    else:
        ea, eb = percentile_norm_joint(ea, eb, p_low=joint_clip[0], p_high=joint_clip[1])
        imshow_kwargs = dict(cmap=bev_cmap, vmin=0.0, vmax=1.0, interpolation=bev_interp,
                             extent=bev_extent, origin=bev_origin)

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)

    axes[0].imshow(ea, **imshow_kwargs)
    axes[0].set_aspect("equal")
    if bev_extent is not None:
        axes[0].set_xlim(bev_extent[0], bev_extent[1])
        axes[0].set_ylim(bev_extent[2], bev_extent[3])
    axes[0].set_title(labels[0])
    axes[0].axis("off")

    axes[1].imshow(eb, **imshow_kwargs)
    axes[1].set_aspect("equal")
    if bev_extent is not None:
        axes[1].set_xlim(bev_extent[0], bev_extent[1])
        axes[1].set_ylim(bev_extent[2], bev_extent[3])
    axes[1].set_title(labels[1])
    axes[1].axis("off")

    draw_lidar_top(
        nusc=nusc,
        sample_token=sample_token,
        ax=axes[2],
        bev_extent=bev_extent,
        axes_limit=lidar_axes_limit,
        stride=lidar_stride,
        s=lidar_pt_size,
        alpha=lidar_alpha,
        cmap=lidar_cmap,
    )
    axes[2].set_title(labels[2])

    if title:
        fig.suptitle(title, y=0.99)

    if out_path is not None:
        out_path = ensure_parent(out_path)
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0.0, dpi=dpi)

    if show:
        plt.show()
    plt.close(fig)

    return {"ea": ea, "eb": eb}


# ---------------------------
# Optional: quick helper to try orientations and save a grid
# ---------------------------
def sweep_bev_orientations(
    bev_pre_bchw,
    bev_post_bchw,
    sample_token: str,
    nusc: NuScenes,
    out_dir: str,
    bev_extent: Tuple[float, float, float, float],
    *,
    b: int = 0,
    signed: bool = False,
):
    """
    Saves multiple triplets with different BEV orientation modes.
    Great for quickly finding the correct alignment once.
    """
    modes = ["none", "T", "T_flip_y", "T_flip_x", "flip_y", "flip_x", "rot90", "rot270", "rot180"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in modes:
        render_bev_triplet_activation(
            bev_pre_bchw, bev_post_bchw,
            sample_token=sample_token, nusc=nusc,
            out_path=str(out_dir / f"triplet_orient_{m}.png"),
            b=b,
            bev_extent=bev_extent,
            bev_orient=m,  # type: ignore
            signed=signed,
            title=f"BEV orient = {m}",
        )

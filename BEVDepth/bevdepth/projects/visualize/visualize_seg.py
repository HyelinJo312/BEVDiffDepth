from pathlib import Path
import os
import numpy as np
import cv2
from tqdm import tqdm


# -------------------------
# IO / dtype
# -------------------------
def infer_dtype_from_file_size(bin_path: Path, H: int, W: int):
    """Infer dtype by file size assuming shape is (H, W)."""
    size = os.path.getsize(bin_path)
    if size == H * W:
        return np.uint8
    if size == 2 * H * W:
        return np.uint16
    raise ValueError(
        f"[{bin_path.name}] cannot infer dtype: file_size={size} bytes, "
        f"expected {H*W} (uint8) or {2*H*W} (uint16)."
    )


def load_label_map(bin_path: Path, H: int, W: int, dt) -> np.ndarray:
    """Load raw label map (.bin) into int32 (H,W)."""
    data = np.fromfile(str(bin_path), dtype=dt)
    if data.size != H * W:
        raise ValueError(f"[{bin_path.name}] size mismatch: {data.size} != {H}*{W}")
    return data.reshape(H, W).astype(np.int32)


# -------------------------
# Visualization
# -------------------------
def make_distinct_palette(num_classes: int, background_bgr=(0, 0, 0), ignore_bgr=(0, 0, 0)):
    """
    Deterministic 'distinct' palette with good separation using HSV sweep.
    Returns BGR palette of shape (num_classes,3).
    """
    palette = np.zeros((num_classes, 3), dtype=np.uint8)

    # class 0: background
    palette[0] = np.array(background_bgr, dtype=np.uint8)

    # other classes: evenly-spaced hues
    for c in range(1, num_classes):
        h = int((180 * (c - 1) / max(1, num_classes - 1)))  # OpenCV hue: [0,179]
        s, v = 200, 255
        bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0] # type: ignore
        palette[c] = bgr

    return palette


def make_bit_palette(num_classes, background_bgr=(0, 0, 0)):
    """
    Pascal VOC 스타일 bit-wise 팔레트 (deterministic, 매우 다양한 색)
    Returns palette in BGR for OpenCV.
    """
    palette = np.zeros((num_classes, 3), dtype=np.uint8)

    def bitget(byteval, idx):
        return (byteval >> idx) & 1

    for i in range(num_classes):
        r = g = b = 0
        cid = i
        for j in range(8):
            r |= bitget(cid, 0) << (7 - j)
            g |= bitget(cid, 1) << (7 - j)
            b |= bitget(cid, 2) << (7 - j)
            cid >>= 3
        palette[i] = np.array([b, g, r], dtype=np.uint8)  # BGR

    palette[0] = np.array(background_bgr, dtype=np.uint8)
    return palette


def colorize_semantic_palette(lbl, palette, ignore_label=255, ignore_bgr=(0, 0, 0)):
    lbl = lbl.astype(np.int32)

    ignore_mask = None
    if ignore_label is not None:
        ignore_mask = (lbl == ignore_label)

    x = lbl.copy()
    if ignore_mask is not None:
        x[ignore_mask] = 0  # 임시로 background로

    x = np.clip(x, 0, palette.shape[0] - 1)
    sem_bgr = palette[x]

    if ignore_mask is not None:
        sem_bgr[ignore_mask] = np.array(ignore_bgr, dtype=np.uint8)

    return sem_bgr

def overlay_with_ignore(rgb_bgr, sem_bgr, lbl, alpha, ignore_label=255):
    alpha = float(alpha)
    alpha = max(0.0, min(1.0, alpha))
    blended = cv2.addWeighted(rgb_bgr, 1-alpha, sem_bgr, alpha, 0)

    if ignore_label is None:
        return blended

    m = (lbl != ignore_label)
    out = rgb_bgr.copy()
    out[m] = blended[m]
    return out

def draw_label_boundaries(lbl, ignore_label, thickness):
    """
    Compute boundary mask where neighboring pixels have different labels.
    Ignore-label borders can be excluded if ignore_label is given.
    Returns uint8 mask (H,W) with 255 on boundary.
    """
    lbl = lbl.astype(np.int32)
    H, W = lbl.shape
    edge = np.zeros((H, W), dtype=np.uint8)

    # differences
    edge[:-1, :] |= (lbl[:-1, :] != lbl[1:, :])
    edge[:, :-1] |= (lbl[:, :-1] != lbl[:, 1:])

    if ignore_label is not None:
        # remove edges that are purely due to ignore transitions (optional, makes cleaner)
        ign = (lbl == ignore_label)
        # if either side is ignore, drop that edge
        drop = np.zeros_like(edge, dtype=bool)
        drop[:-1, :] |= (ign[:-1, :] | ign[1:, :])
        drop[:, :-1] |= (ign[:, :-1] | ign[:, 1:])
        edge[drop] = 0

    edge = (edge.astype(np.uint8) * 255)

    if thickness > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))
        edge = cv2.dilate(edge, k, iterations=1)

    return edge


def overlay_boundaries(rgb_bgr: np.ndarray, boundary_mask: np.ndarray, boundary_bgr=(255, 255, 255)) -> np.ndarray:
    out = rgb_bgr.copy()
    out[boundary_mask > 0] = boundary_bgr
    return out


# -------------------------
# Batch visualize
# -------------------------
def visualize_dir(
    bin_dir: Path,
    out_dir: Path,
    H: int,
    W: int,
    num_classes: int,
    overlay_flag: bool,
    img_dir,
    img_ext: str,
    alpha: float,
    save_sem_only: bool,
    auto_dtype: bool,
    dtype,
    ignore_label,
    color_mode: str = "palette",         # "palette" 추천
    save_boundary: bool = True,          # 경계-only overlay도 저장 (추천)
    boundary_thickness: int = 2,
):
    sem_dir = out_dir / "semantic"
    ovl_dir = out_dir / "overlay"
    sem_dir.mkdir(parents=True, exist_ok=True)
    if overlay_flag:
        ovl_dir.mkdir(parents=True, exist_ok=True)
        
    bin_files = sorted(bin_dir.glob("*.bin"))
    if not bin_files:
        raise RuntimeError(f"No .bin files in {bin_dir}")

    palette = make_distinct_palette(num_classes) if color_mode == "palette" else None

    for bin_path in tqdm(bin_files, desc="Visualizing"):
        stem = bin_path.stem

        # dtype
        dt = infer_dtype_from_file_size(bin_path, H, W) if auto_dtype else dtype

        # load label
        try:
            lbl = load_label_map(bin_path, H, W, dt)
        except Exception as e:
            print(f"[WARN] skip {bin_path.name}: {e}")
            continue

        # colorize (ignore 처리 포함)
        palette = make_bit_palette(num_classes)
        sem_bgr = colorize_semantic_palette(lbl, palette, ignore_label=255)

        # save semantic-only
        if save_sem_only or not overlay_flag:
            cv2.imwrite(str(sem_dir / f"{stem}.png"), sem_bgr)

        # overlay
        if overlay_flag:
            if img_dir is None:
                raise ValueError("overlay_flag=True but img_dir is None")

            # image stem: remove trailing _mask
            img_stem = stem[:-5] if stem.endswith("_mask") else stem
            img_path = img_dir / f"{img_stem}{img_ext}"
            if not img_path.exists():
                print(f"[WARN] missing RGB image: {img_path}")
                continue

            rgb_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if rgb_bgr is None:
                print(f"[WARN] failed to read: {img_path}")
                continue

            # resize semantic and label if needed
            if rgb_bgr.shape[:2] != sem_bgr.shape[:2]:
                sem_for_overlay = cv2.resize(sem_bgr, (rgb_bgr.shape[1], rgb_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
                lbl_for_overlay = cv2.resize(lbl, (rgb_bgr.shape[1], rgb_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                sem_for_overlay = sem_bgr
                lbl_for_overlay = lbl

            over = overlay_with_ignore(rgb_bgr, sem_for_overlay, lbl_for_overlay, alpha, ignore_label=255)
            cv2.imwrite(str(ovl_dir / f"{stem}_overlay.png"), over)

    print(f"[OK] saved to: {out_dir}")


# =========================================================
# Hyperparameters
# =========================================================
if __name__ == "__main__":
    BIN_DIR = Path("../../data/nuscenes_semantic/samples/CAM_FRONT")
    IMG_DIR = Path("../../data/nuScenes/samples/CAM_FRONT")
    OUT_DIR = Path("../../semantic_maps/CAM_FRONT_bestviz")

    H, W = 900, 1600
    AUTO_DTYPE = True
    DTYPE = np.uint8

    NUM_CLASSES = 256
    IGNORE_LABEL = 255  # ★ 핵심

    OVERLAY = True
    IMG_EXT = ".jpg"
    ALPHA = 0.45

    SAVE_SEM_ONLY = True       # semantic-only도 같이 저장 추천
    COLOR_MODE = "palette"     # "palette" 추천 (turbo도 가능)
    SAVE_BOUNDARY = False       # boundary-only overlay 저장 추천

    visualize_dir(
        bin_dir=BIN_DIR,
        out_dir=OUT_DIR,
        H=H, W=W,
        num_classes=NUM_CLASSES,
        overlay_flag=OVERLAY,
        img_dir=IMG_DIR if OVERLAY else None,
        img_ext=IMG_EXT,
        alpha=ALPHA,
        save_sem_only=SAVE_SEM_ONLY,
        auto_dtype=AUTO_DTYPE,
        dtype=DTYPE,
        ignore_label=IGNORE_LABEL,
        color_mode=COLOR_MODE,
        save_boundary=SAVE_BOUNDARY,
        boundary_thickness=2,
    )

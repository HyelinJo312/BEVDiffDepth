from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm


# ============ Your mapping (for reference) ============
dic = {
    'sedan': 1,
    'highway': 2,
    'bus': 3,
    'truck': 4,
    'terrain': 5,
    'tree': 6,
    'sidewalk': 7,
    'bicycle': 8,
    'bicyclist': 8,
    'barrier': 9,
    'barricade': 9,
    'person': 10,
    'pedestrian': 10,
    'building': 11,
    'bridge': 11,
    'pole': 11,
    'billboard': 11,
    'light': 11,
    'ashbin': 11,
    'motorcycle': 12,
    'motorcyclist': 12,
    'crane': 13,
    'trailer': 14,
    'cone': 15,
    'sky': 16,
}

# colors[class_id] = [R,G,B,A]  (note: index 0 is background)
# colors = np.array(
#     [
#         [0, 0, 0, 153],              # 0 background (we will map -1 -> 0)
#         [255, 120, 50, 153],         # 1 sedan
#         [255, 192, 203, 153],        # 2 highway
#         [255, 255, 0, 153],          # 3 bus
#         [200, 180, 0, 153],          # 4 truck
#         [0, 255, 255, 153],          # 5 terrain
#         [0, 175, 0, 153],            # 6 tree
#         [255, 0, 0, 153],            # 7 sidewalk
#         [255, 240, 150, 153],        # 8 bicycle/bicyclist
#         [135, 60, 0, 153],           # 9 barrier/barricade
#         [160, 32, 240, 153],         # 10 person/pedestrian
#         [255, 0, 255, 153],          # 11 manmade
#         [139, 137, 137, 153],        # 12 motorcycle
#         [75, 0, 75, 153],            # 13 crane
#         [150, 240, 80, 153],         # 14 trailer
#         [230, 230, 250, 153],        # 15 cone
#         [135, 206, 235, 153],         # 16 sky
#         [0, 255, 127, 153],          # 17 (unused)
#         [255, 99, 71, 153],          # 18 (unused)
#         [111, 111, 222, 153],        # 19 (unused)
#     ],
#     dtype=np.uint8
# )

colors = np.array(
    [
        [0, 0, 0, 153],              # 0 background
        [255, 120, 50, 153],         # 1 sedan
        [255, 192, 203, 153],        # 2 highway
        [255, 255, 0, 153],          # 3 bus
        [200, 180, 0, 153],          # 4 truck
        [0, 255, 255, 153],          # 5 terrain
        [0, 175, 0, 153],            # 6 tree
        [255, 0, 0, 153],            # 7 sidewalk
        [255, 240, 150, 153],        # 8 bicycle/bicyclist
        [135, 60, 0, 153],           # 9 barrier/barricade
        [160, 32, 240, 153],         # 10 person/pedestrian

        [255, 0, 255, 153],          # 11 building (기존 manmade)
        [139, 137, 137, 153],        # 12 motorcycle
        [75, 0, 75, 153],            # 13 crane
        [150, 240, 80, 153],         # 14 trailer
        [230, 230, 250, 153],        # 15 cone
        [135, 206, 235, 153],        # 16 sky

        # 새로 추가 (17~21)
        [0, 255, 127, 153],          # 17 bridge
        [255, 99, 71, 153],          # 18 pole
        [111, 111, 222, 153],        # 19 billboard
        [255, 215, 0, 153],          # 20 light
        [0, 191, 255, 153],          # 21 ashbin
    ],
    dtype=np.uint8
)

# ---------------------------
# Core: load + colorize + overlay
# ---------------------------
def load_mask_int8(bin_path: Path, H: int, W: int) -> np.ndarray:
    """
    GroundedSAM 생성 코드에서 mask.astype(np.int8) 로 저장했으므로 dtype=int8로 읽는다.
    """
    x = np.fromfile(str(bin_path), dtype=np.int8)
    if x.size != H * W:
        raise ValueError(f"[{bin_path.name}] size mismatch: {x.size} != {H}*{W}")
    return x.reshape(H, W).astype(np.int16)  # 계산 안전하게 int16로 올림


def colorize_mask_to_rgba(mask_hw: np.ndarray, colors_rgba: np.ndarray,
                          unknown_to_bg: bool = True,
                          unknown_rgba=(0, 0, 0, 0)) -> np.ndarray:
    """
    mask_hw 값:
      -1 : unknown/uncertain (생성 코드 dic.get(x,-1))
      1..16 : class id
    시각화 규칙:
      - 기본: -1을 background(0)으로 보이게 (unknown_to_bg=True)
      - 만약 unknown을 따로 강조하고 싶으면 unknown_to_bg=False로 두고 unknown_rgba를 쓰면 됨.
    """
    m = mask_hw.copy()

    # unknown handling
    if unknown_to_bg:
        m[m < 0] = 0
    else:
        # keep -1 as special; we will paint it with unknown_rgba later
        pass

    # clip to palette range
    max_id = colors_rgba.shape[0] - 1
    valid = (m >= 0) & (m <= max_id)

    out = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.uint8)

    if not unknown_to_bg:
        # initialize unknown as unknown_rgba
        out[:] = np.array(unknown_rgba, dtype=np.uint8)

    # fill valid labels
    out[valid] = colors_rgba[m[valid].astype(np.int32)]

    # if unknown_to_bg=True, unknown already became 0 so it is painted with colors[0]
    return out

def colorize_semantic_rgb(mask_hw, colors_rgb):
    """
    mask_hw: (H,W), int8 / int16
      -1 : unknown
      1..16 : class id
    """
    m = mask_hw.copy()
    m[m < 0] = 0  # unknown -> background

    max_id = colors_rgb.shape[0] - 1
    m = np.clip(m, 0, max_id)

    sem_rgb = colors_rgb[m.astype(np.int32)]
    return sem_rgb  # (H,W,3) uint8


def overlay_rgba_on_bgr(rgb_bgr: np.ndarray, mask_rgba: np.ndarray) -> np.ndarray:
    """
    mask_rgba: RGBA uint8
    returns: BGR uint8
    """
    rgb = rgb_bgr.astype(np.float32)
    mask_rgb = mask_rgba[..., :3].astype(np.float32)      # RGB
    mask_bgr = mask_rgb[..., ::-1]                        # BGR
    alpha = (mask_rgba[..., 3:4].astype(np.float32) / 255.0)

    out = rgb * (1.0 - alpha) + mask_bgr * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------
# Batch visualization
# ---------------------------
def visualize_dir(
    bin_dir: Path,
    out_dir: Path,
    H: int,
    W: int,
    colors_rgba: np.ndarray,
    overlay_flag: bool = False,
    img_dir: Path = None,
    img_ext: str = ".jpg",
    strip_suffix: str = "_mask",
    save_sem_only: bool = True,
    unknown_to_bg: bool = True,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    sem_dir = out_dir / "semantic_rgb"
    ovl_dir = out_dir / "overlay"
    if save_sem_only:
        sem_dir.mkdir(parents=True, exist_ok=True)
    if overlay_flag:
        if img_dir is None:
            raise ValueError("overlay_flag=True but img_dir is None")
        ovl_dir.mkdir(parents=True, exist_ok=True)

    bin_files = sorted(bin_dir.glob("*.bin"))
    if not bin_files:
        raise RuntimeError(f"No .bin files found in {bin_dir}")

    colors_rgb = colors_rgba[:, :3].copy()

    for bin_path in tqdm(bin_files, desc="Visualizing GroundedSAM masks"):
        stem = bin_path.stem  # e.g., xxx_mask

        # 1) load int8 map
        try:
            mask = load_mask_int8(bin_path, H, W)
        except Exception as e:
            print(f"[WARN] skip {bin_path.name}: {e}")
            continue

        # 2) colorize RGBA
        mask_rgba = colorize_mask_to_rgba(mask, colors_rgba, unknown_to_bg=unknown_to_bg)
        sem_rgb = colorize_semantic_rgb(mask, colors_rgb)

        # 3) save semantic-only (as BGRA for cv2)
        if save_sem_only:
            # mask_bgra = mask_rgba[..., [2, 1, 0, 3]]
            # cv2.imwrite(str(sem_dir / f"{stem}.png"), mask_bgra)
            cv2.imwrite(str(sem_dir / f"{stem}.png"), sem_rgb[:, :, ::-1])  # RGB → BGR

        # 4) overlay
        if overlay_flag:
            img_stem = stem
            if strip_suffix and img_stem.endswith(strip_suffix):
                img_stem = img_stem[: -len(strip_suffix)]

            img_path = img_dir / f"{img_stem}{img_ext}"
            if not img_path.exists():
                print(f"[WARN] missing RGB image: {img_path}")
                continue

            rgb_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if rgb_bgr is None:
                print(f"[WARN] failed to read: {img_path}")
                continue

            # resize semantic if needed
            if rgb_bgr.shape[0] != mask_rgba.shape[0] or rgb_bgr.shape[1] != mask_rgba.shape[1]:
                mask_rgba_rs = cv2.resize(mask_rgba, (rgb_bgr.shape[1], rgb_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                mask_rgba_rs = mask_rgba

            over = overlay_rgba_on_bgr(rgb_bgr, mask_rgba_rs)
            cv2.imwrite(str(ovl_dir / f"{stem}_overlay.png"), over)

    print(f"[OK] semantic-only -> {sem_dir if save_sem_only else '(disabled)'}")
    print(f"[OK] overlay       -> {ovl_dir if overlay_flag else '(disabled)'}")


# ---------------------------
# Hyperparameters
# ---------------------------
if __name__ == "__main__":
    BIN_DIR = Path("../../data/nuscenes_semantic/samples/CAM_FRONT")   # where *_mask.bin are
    IMG_DIR = Path("../../data/nuscenes/samples/CAM_FRONT")            # RGB images
    OUT_DIR = Path("../../semantic_maps/grounded_sam_vis")

    H, W = 900, 1600

    OVERLAY = True
    IMG_EXT = ".jpg"

    SAVE_SEM_ONLY = True

    # 생성 코드가 -1을 unknown으로 쓰므로, 기본은 background로 보이게 (-1->0)
    UNKNOWN_TO_BG = True   # False로 두면 unknown을 따로 표시(투명/다른색) 가능

    visualize_dir(
        bin_dir=BIN_DIR,
        out_dir=OUT_DIR,
        H=H, W=W,
        colors_rgba=colors,
        overlay_flag=OVERLAY,
        img_dir=IMG_DIR if OVERLAY else None,
        img_ext=IMG_EXT,
        strip_suffix="_mask",
        save_sem_only=SAVE_SEM_ONLY,
        unknown_to_bg=UNKNOWN_TO_BG,
    )

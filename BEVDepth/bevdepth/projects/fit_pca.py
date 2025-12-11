# offline_fit_pca.py
import torch
import numpy as np
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from bevdepth.datasets.nusc_det_dataset import NuscDetDataset, collate_fn
from functools import partial

# fit_pca_offline.py
import os
from functools import partial

import mmcv
import numpy as np
import torch
from torch.utils.data import DataLoader
from bevdepth.datasets.nusc_det_dataset import NuscDetDataset, collate_fn
from bevdepth.projects.fm_feature import GetDINOV2Feat

# =======================================================================
# 0. Config 
# =======================================================================

H = 900
W = 1600
final_dim = (480, 900)
img_conf = dict(
    img_mean=[123.675, 116.28, 103.53],
    img_std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

backbone_conf = {
    'x_bound': [-51.2, 51.2, 0.8],
    'y_bound': [-51.2, 51.2, 0.8],
    'z_bound': [-5, 3, 8],
    'd_bound': [2.0, 58.0, 0.5],
    'final_dim': final_dim,
    'output_channels': 80,
    'downsample_factor': 14,
}

ida_aug_conf = {
    'resize_lim': (0.386, 0.55),
    'final_dim': final_dim,
    'rot_lim': (-5.4, 5.4),
    'H': H,
    'W': W,
    'rand_flip': True,
    'bot_pct_lim': (0.0, 0.0),
    'cams': [
        'CAM_FRONT_LEFT',
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',
        'CAM_BACK',
        'CAM_BACK_RIGHT',
    ],
    'Ncams': 6,
}

bda_aug_conf = {
    'rot_lim': (-22.5, 22.5),
    'scale_lim': (0.95, 1.05),
    'flip_dx_ratio': 0.5,
    'flip_dy_ratio': 0.5,
}

CLASSES = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
]

# =======================================================================
# 1. PCA setup
# =======================================================================

C_IN = 768          
C_REDUCED = 128    
MAX_SAMPLES = 1_000_000   

DATA_ROOT = 'data/nuScenes'  # nuScenes root
TRAIN_INFO_PATH = os.path.join(DATA_ROOT, 'nuscenes_infos_train.pkl')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =======================================================================
# 2. NuScenes train dataloader 
# =======================================================================

def build_train_dataloader(
    batch_size_per_device: int = 4,
    num_workers: int = 4,
    use_cbgs: bool = False,
    num_sweeps: int = 1,
    sweep_idxes=None,
    key_idxes=None,
    data_return_depth: bool = False,
    use_fusion: bool = False,
    ):

    if sweep_idxes is None:
        sweep_idxes = []
    if key_idxes is None:
        key_idxes = []

    train_dataset = NuscDetDataset(
        ida_aug_conf=ida_aug_conf,
        bda_aug_conf=bda_aug_conf,
        classes=CLASSES,
        data_root=DATA_ROOT,
        info_paths=TRAIN_INFO_PATH,
        is_train=True,
        use_cbgs=use_cbgs,
        img_conf=img_conf,
        num_sweeps=num_sweeps,
        sweep_idxes=sweep_idxes,
        key_idxes=key_idxes,
        return_depth=data_return_depth,
        use_fusion=use_fusion,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_device,
        num_workers=num_workers,
        drop_last=True,
        shuffle=False,  # PCA용이므로 shuffle=False여도 상관 없음
        collate_fn=partial(
            collate_fn,
            is_return_depth=data_return_depth or use_fusion,
        ),
        sampler=None,
    )
    return train_loader


# =======================================================================
# 3. Feature extractor 
# =======================================================================
@torch.no_grad()
def feature_extractor(batch) -> torch.Tensor:
    '''
    batch: (sweep_imgs, mats, _, img_metas, gt_boxes, gt_labels, depth_labels)
    Return:
        - last_tokens: 
    '''
    dino_model = GetDINOV2Feat()
    imgs = batch[0]
    img_metas = batch[3]
    dino_outputs = dino_model(imgs, img_metas)
    
    img_feats = dino_outputs['last_tokens'] # [B, V, N, C] N=35x58
    # patch_hw = dino_outputs['patch_hw'] # (H2/14, W2/14)  (35, 58)
    
    # img_feats = img_feats.reshape(-1, img_feats.shape[1], img_feats.shape[-1], patch_hw[0], patch_hw[1])  # [B*N, V, C, H2/14, W2/14]
    assert img_feats.shape[-1] == C_IN, f"Expected last dim = {C_IN}, got {img_feats.shape[-1]}"
    return img_feats  # (B, V, N, C) 


# =======================================================================
# 4. Feature sample + PCA fit + save
# =======================================================================
def collect_pca_stats(
    dataloader,
    max_samples: int = MAX_SAMPLES,
    ):
    S = torch.zeros(C_IN, C_IN, dtype=torch.float64, device=DEVICE)
    sum_x = torch.zeros(C_IN, dtype=torch.float64, device=DEVICE)
    N = 0

    for batch_idx, batch in enumerate(dataloader):
        feats = feature_extractor(batch).to(DEVICE)  # (B,V,N,C)
        feats = feats.reshape(-1, C_IN)  # (N_batch, C_IN)
        n_batch = feats.size(0)

        # 최대 샘플 수 초과하지 않도록 잘라냄
        if N + n_batch > max_samples:
            n_batch = max_samples - N
            feats = feats[:n_batch]

        if n_batch <= 0:
            break

        # Accumulate statistics
        # S += Xᵀ X
        S += feats.T @ feats          # (C_IN, C_IN)
        # sum_x += Σ x
        sum_x += feats.sum(dim=0)     # (C_IN,)
        N += n_batch

        print(
            f"[collect_pca_stats] batch {batch_idx}, "
            f"collected = {N}/{max_samples}"
        )

        if N >= max_samples:
            break

    print("[collect_pca_stats] Done. N =", N)
    return S.cpu(), sum_x.cpu(), N


def fit_and_save_pca_from_stats(
    S: torch.Tensor,
    sum_x: torch.Tensor,
    N: int,
    n_components: int = C_REDUCED,
    out_dir: str = "./pca_ckpts",
):
    """
    - S: (C_IN, C_IN) = Σ x xᵀ
    - sum_x: (C_IN,)  = Σ x
    - N: int          = # of samples
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) mean
    mu = (sum_x / N).numpy().astype(np.float32)   # (C_IN,)

    # 2) Cov = E[xxᵀ] - μμᵀ -> E[xxᵀ] ≈ S / N
    S_np = S.numpy()
    Cov = S_np / float(N) - np.outer(mu, mu)      # (C_IN, C_IN)

    print("[fit_and_save_pca_from_stats] Computing eigen decomposition...")
    # eigen-decomposition: Cov = Q Λ Qᵀ
    eigvals, eigvecs = np.linalg.eigh(Cov)        # eigvecs: (C_IN, C_IN)

    # 3) 고유값 내림차순 정렬 후 상위 n_components 선택
    idx = np.argsort(eigvals)[::-1]              # 큰 값부터
    idx = idx[:n_components]
    # eigvecs[:, idx]: (C_IN, n_components)
    P = eigvecs[:, idx].astype(np.float32)       # projection matrix

    print("[fit_and_save_pca_from_stats] eigvals (top 5):", eigvals[idx[:5]])
    print("[fit_and_save_pca_from_stats] P shape:", P.shape)
    print("[fit_and_save_pca_from_stats] mu shape:", mu.shape)

    # 4) Save
    #   - mu: (C_IN,)
    #   - P : (C_IN, C_REDUCED)
    out_path = os.path.join(out_dir, f"pca_{C_IN}_to_{n_components}.npz")
    np.savez(out_path, mu=mu, P=P)

    print(f"[fit_and_save_pca_from_stats] Saved PCA mean and projection to {out_path}")


# =======================================================================
# 5. main
# =======================================================================

def main():
    mmcv.mkdir_or_exist("./outputs")

    # 1) dataloader 
    train_loader = build_train_dataloader(
        batch_size_per_device=1,
        num_workers=4,
        use_cbgs=False,
        num_sweeps=1,
        sweep_idxes=[],
        key_idxes=[],
        data_return_depth=True,
        use_fusion=True,
    )

    # 2) PCA statistics (S, sum_x, N)
    S, sum_x, N = collect_pca_stats(
        dataloader=train_loader,
        max_samples=MAX_SAMPLES,
    )

    # 3) PCA fit + save (mu, P)
    fit_and_save_pca_from_stats(
        S=S,
        sum_x=sum_x,
        N=N,
        n_components=C_REDUCED,
        out_dir="./pca_ckpts",
    )


if __name__ == "__main__":
    main()
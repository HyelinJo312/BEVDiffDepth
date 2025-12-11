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
from sklearn.decomposition import PCA
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
C_REDUCED = 256    
MAX_SAMPLES = 3_000_000   

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
        shuffle=False,  
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
def collect_feature_samples(
    dataloader,
    dino_model,
    max_samples: int = MAX_SAMPLES,
    batch_size_per_device=4,
    ):

    feature_list = []
    N_collected = 0

    for batch_idx, batch in enumerate(dataloader):
        feats = feature_extractor(batch)  # (B, V, N, C_IN)
        # (B, V, N, C) -> (B*V*N, C)
        feats = feats.reshape(-1, C_IN)               # (N_batch, C_IN)

        feats_np = feats.detach().cpu().numpy()       # float32 or float16 -> numpy
        N_batch = feats_np.shape[0]

        if N_collected + N_batch > max_samples:
            remain = max_samples - N_collected
            if remain <= 0:
                break
            
            idx = np.random.choice(N_batch, remain, replace=False)
            feats_np = feats_np[idx]
            N_batch = remain

        feature_list.append(feats_np)
        N_collected += N_batch

        print(
            f"[collect_feature_samples] {batch_idx+1}-th batch, "
            f"[collect_feature_samples] {(batch_idx+1)*batch_size_per_device} frames "
            f"collected = {N_collected}/{max_samples}"
        )

        if N_collected >= max_samples:
            break

    feature_bank = np.concatenate(feature_list, axis=0)  # (N_samples, C_IN)
    print("[collect_feature_samples] Final feature bank shape:", feature_bank.shape)
    return feature_bank


def fit_and_save_pca(
    feature_bank,
    n_components,
    out_dir,
    tag):
    """
    feature_bank: (N_samples, C_IN)
    sklearn.PCA로 fit 후, projection P와 mean mu를 .npz로 저장.
    """
    os.makedirs(out_dir, exist_ok=True)

    print("[fit_and_save_pca] Fitting PCA...")
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",  
        whiten=False,
    )
    pca.fit(feature_bank)

    # components_: (n_components, C_IN)
    P = pca.components_.astype(np.float32)  # (C_REDUCED, C_IN)
    mu = pca.mean_.astype(np.float32)       # (C_IN,)

    print("[fit_and_save_pca] PCA components shape (P):", P.shape)
    print("[fit_and_save_pca] PCA mean shape (mu):", mu.shape)

    out_path = os.path.join(
        out_dir,
        f"pca_{tag}_{C_IN}_to_{n_components}.npz"
    )
    np.savez(out_path, mu=mu, P=P)

    print(f"[fit_and_save_pca] Saved PCA params to {out_path}")
    return out_path 


# =======================================================================
# 5. main
# =======================================================================

def main():
    mmcv.mkdir_or_exist("./outputs")

    # 1) dataloader 
    train_loader = build_train_dataloader(
        batch_size_per_device=4,
        num_workers=4,
        use_cbgs=False,
        num_sweeps=1,
        sweep_idxes=[],
        key_idxes=[],
        data_return_depth=True,
        use_fusion=True,
    )

    print("[main] Loading DINOv2 model...")
    dino_model = GetDINOV2Feat(device=DEVICE)
    dino_model.to(DEVICE)
    dino_model.eval()

    feature_bank = collect_feature_samples(
        dataloader=train_loader,
        dino_model=dino_model,
        max_samples=MAX_SAMPLES,
        batch_size_per_device=4,
    )

    pca_path = fit_and_save_pca(
        feature_bank=feature_bank,
        n_components=C_REDUCED,
        out_dir="./pca_ckpts",
        tag="sckit",
    )

    print("[main] Done. PCA saved at:", pca_path)


if __name__ == "__main__":
    main()
import torch

import os
import numpy as np
import torch


class LoadDepth:
    """
    Load depth maps saved as .npy using camera filenames from img_metas.
    
    Depth file location example:
    root_dir/CAM_FRONT/npy/<image_name>_dpt_depth.npy
    """
    def __init__(self, root_dir="../data/nuscenes/depth"):
        self.root_dir = root_dir

    def _get_cam_name(self, img_path: str):
        return os.path.basename(os.path.dirname(img_path))

    def _get_depth_path(self, cam_name: str, image_fname: str):

        base = os.path.splitext(os.path.basename(image_fname))[0]
        depth_name = f"{base}_dpt_depth.npy"
        depth_path = os.path.join(self.root_dir, cam_name, "npy", depth_name)
        return depth_path

    def load_one_sample(self, img_meta: dict):
        filenames = img_meta["filename"]
        depth_list = []

        for img_path in filenames:
            cam_name = self._get_cam_name(img_path)
            depth_path = self._get_depth_path(cam_name, img_path)

            if not os.path.exists(depth_path):
                raise FileNotFoundError(f"[LoadDepth] depth not found: {depth_path}")

            depth_np = np.load(depth_path).astype(np.float32)   # (H, W)
            depth_tensor = torch.from_numpy(depth_np)
            depth_list.append(depth_tensor)

        depth_stack = torch.stack(depth_list, dim=0)  # (V, H, W)
        return depth_stack

    def __call__(self, img_metas):
        batch_depth = []

        for meta in img_metas:
            depth_tensor = self.load_one_sample(meta)   # (V, H, W)
            batch_depth.append(depth_tensor)

        batch_depth = torch.stack(batch_depth, dim=0)    # (B, V, H, W)
        return batch_depth
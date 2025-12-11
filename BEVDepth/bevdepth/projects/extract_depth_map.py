# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from diffusers
#   (https://github.com/huggingface/diffusers)
# Copyright (c) 2022 diffusers authors, licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

'''
Following code is adapted from 
https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
'''

import argparse
import os, sys
import time

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image

from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer
# from diffusers import DDPMScheduler, DDIMScheduler, UNet2DConditionModel

import mmcv
from mmcv import Config
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, wrap_fp16_model)
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/..")
from bevdepth.datasets.nusc_det_dataset import NuscDetDataset, collate_fn
from mmdet.apis import set_random_seed
from bevdepth.projects.fm_feature import GetDPTDepth, GetDPTDepthV2

logger = get_logger(__name__, log_level="INFO")

def parse_args():
     # put all arg parse here
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    parser.add_argument('--bev_config', 
                        default="",
                        help='test config file path')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def load_raw_images_from_filenames(img_filenames):
    """
    img_filenames: list[str], 길이 V (예: 6개 카메라)
    Returns:
      imgs: torch.Tensor, shape (V, C, H, W), raw RGB, 0~255 float
    """
    imgs = []
    for fname in img_filenames:
        img = Image.open(fname).convert("RGB")
        img_np = np.array(img).astype(np.float32)  # (H, W, 3)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (3, H, W)
        imgs.append(img_tensor)
    imgs = torch.stack(imgs, dim=0)  # (V, C, H, W)
    return imgs


def test():
    args = parse_args()

    bev_cfg = Config.fromfile(args.bev_config)
    
    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=False)
        
    if args.launcher != 'none':
        init_dist(args.launcher, **bev_cfg.dist_params)

    get_depth = GetDPTDepthV2()
    
    train_cfg = dict(
        point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
        grid_size=[512, 512, 1],
        voxel_size=[0.2, 0.2, 8],
        out_size_factor=4,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
    )

    test_cfg = dict(
        post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        max_per_img=500,
        max_pool_nms=False,
        min_radius=[4, 12, 10, 1, 0.85, 0.175],
        score_threshold=0.1,
        out_size_factor=4,
        voxel_size=[0.2, 0.2, 8],
        nms_type='circle',
        pre_max_size=1000,
        post_max_size=83,
        nms_thr=0.2,
    )
    dataset = build_dataset(bev_cfg.data.train,
                            default_args={
                                        'pc_range': bev_cfg.point_cloud_range,
                                        'use_3d_bbox': bev_cfg.use_3d_bbox,
                                        'num_classes': bev_cfg.num_classes,
                                        'num_bboxes': bev_cfg.num_bboxes,
                                    })
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=bev_cfg.data.samples_per_gpu,
        workers_per_gpu=bev_cfg.data.workers_per_gpu,
        dist=(args.launcher != 'none'),
        shuffle=False,
        nonshuffler_sampler=bev_cfg.data.nonshuffler_sampler,
    )

    
    save_path = os.path.join('../../data/nuscenes/depth')

    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    
    for step, batch in enumerate(dataloader):
        img = batch['img'].data[0]       # (B, len_queue, V, C, H, W)
        len_queue = img.size(1)
        img = img[:, -1, ...]
        B = img.size(0)
        
        img_metas_all = batch['img_metas'].data[0]  # list[ list[dict] ], len = B
        curr_img_metas = [meta_seq[len_queue-1] for meta_seq in img_metas_all]  # list[dict], len = B

        all_raw_imgs = []
        for b in range(B):
            img_filenames = curr_img_metas[b]['filename']   # list[str], (CAM_FRONT, ...)
            raw_imgs_b = load_raw_images_from_filenames(img_filenames)  # (V, C, H, W)
            all_raw_imgs.append(raw_imgs_b)

        raw_imgs = torch.stack(all_raw_imgs, dim=0)         # (B, V, C, H, W)

        get_depth(raw_imgs, curr_img_metas, save_dir=save_path)

        if rank == 0:
            prog_bar.update(B*world_size)


if __name__ == "__main__":
    test()






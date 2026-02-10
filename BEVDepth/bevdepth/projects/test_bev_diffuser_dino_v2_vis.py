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
from functools import partial
import mmcv
from mmcv import Config
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, wrap_fp16_model)
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/..")
# from bevdepth.datasets.nusc_det_dataset_v2 import NuscDetDataset, collate_fn
from bevdepth.projects.data_utils import CustomNuScenesDiffusionDataset, collate_fn, DistributedGroupSampler
# from bevdepth.projects.mmdet3d_plugin.datasets.builder import build_dataloader
# from bevdepth.projects.mmdet3d_plugin.bevformer.apis.test import custom_encode_mask_results, collect_results_cpu
from mmdet.apis import set_random_seed

from bevdepth.projects.scheduler_utils import DDIMGuidedScheduler
from bevdepth.projects.model_utils import get_bev_model, build_unet, instantiate_from_config, get_bevdepth_model
from bevdepth.projects.layout_diffusion.layout_diffusion_unet import LayoutDiffusionUNetModel
from bevdepth.projects.fm_feature import GetDINOV2Feat
from bevdepth.utils.torch_dist import all_gather_object, get_rank, synchronize
from bevdepth.evaluators.det_evaluators import DetNuscEvaluator
from torch.utils.data.distributed import DistributedSampler
from bevdepth.projects.visualize.bev_visualize import render_bev_triplet, bev_extent_from_cfg
# from bevdepth.projects.visualize.bev_visualize_v2 import render_bev_triplet_activation, bev_extent_from_point_cloud_range

logger = get_logger(__name__, log_level="INFO")

def parse_args():
     # put all arg parse here
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    parser.add_argument('--bev_config', 
                        default="",
                        help='test config file path')
    
    parser.add_argument('--bev_checkpoint', 
                        default="",
                        help='checkpoint file')
    
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher')
    
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
        choices=[
            "CompVis/stable-diffusion-v1-4",
            "stabilityai/stable-diffusion-2-1"
        ],
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="",
        help="The checkpoint directory of unet.",
    )


    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'sample' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    
    parser.add_argument(
        "--use_classifier_guidence",
        action='store_true',
        help="whether to use classifier guidence",
    )
    
    parser.add_argument(
        '--noise_timesteps', 
        type=int, 
        default=0, 
        help='The number of timesteps to add noise.')
    
    parser.add_argument(
        '--denoise_timesteps', 
        type=int, 
        default=5, 
        help='The number of timesteps to denoise.')
    
    parser.add_argument(
        '--num_inference_steps', 
        type=int, 
        default=5, 
        help='The number of diffusion steps to run the unet.')
    
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')

    parser.add_argument(
        "--depth_dir",
        type=str,
        default=None
    )

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args



def test():
    args = parse_args()

    bev_cfg = Config.fromfile(args.bev_config)
    
    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=False)
        
    if args.launcher != 'none':
        init_dist(args.launcher, **bev_cfg.dist_params)
        
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMGuidedScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    if args.prediction_type is not None:
        noise_scheduler.register_to_config(prediction_type=args.prediction_type)
    
    bev_model = get_bevdepth_model(bev_cfg, args).to(device)
    bev_model.requires_grad_(False)
    bev_model.eval()

    # unet = instantiate_from_config(bev_cfg.unet)
    # # unet = build_unet(bev_cfg.unet)
    # unet.from_pretrained(args.checkpoint_dir, subfolder="unet")
    # unet.to(device, dtype=torch.float32)
    # unet.requires_grad_(False) 
    # unet.eval()
    unet = None
    
    get_dino = GetDINOV2Feat()
    
    # dataset = NuscDetDataset(bev_cfg.data.val)
    dataset = CustomNuScenesDiffusionDataset(bev_cfg.data.val)

    rank, world_size = get_dist_info()
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=4,
        shuffle=False,
        persistent_workers=True,
        collate_fn=partial(collate_fn, is_return_depth=bev_cfg.data_return_depth, 
                           has_depth_any=bev_cfg.use_da3, use_layout_info=bev_cfg.use_layout, use_semantics=bev_cfg.use_semantics),
        # collate_fn=partial(collate_fn, is_return_depth=bev_cfg.use_fusion, has_depth_any=True),
        sampler=sampler,
    )
    
    save_path = "/home/user/data/processed_dataset/hyelin/bevdiffdepth_vis"
    # save_path = os.path.join('../../../results', args.bev_config.split('/')[-1].split('.')[-2], args.checkpoint_dir.split('/')[-2], args.checkpoint_dir.split('/')[-1])
    # save_path = os.path.join('../../../results/stage1', args.checkpoint_dir.split('/')[-2], args.checkpoint_dir.split('/')[-1])

    evaluate(unet=unet,
             bev_model=bev_model,
             get_dino=get_dino,
             noise_scheduler=noise_scheduler,
             dataset=dataset,
             dataloader=dataloader,
             bev_cfg=bev_cfg,
             device=device,
             eval=args.eval,
             save_path=save_path,
             noise_timesteps=args.noise_timesteps,
             denoise_timesteps=args.denoise_timesteps,
             num_inference_steps=args.num_inference_steps,
             use_classifier_guidence=args.use_classifier_guidence)


def evaluate(unet,
             bev_model,
             get_dino,
             noise_scheduler,
             dataset,
             dataloader,
             bev_cfg,
             device,
             eval='bbox',
             save_path='',
             noise_timesteps=0,
             denoise_timesteps=0,
             num_inference_steps=0,
             use_classifier_guidence=False):
             
    def get_classifier_gradient(x, **kwargs):
        x_ = x.detach().requires_grad_(True)
        x_ = x_.permute(0, 2, 3, 1)
        x_ = x_.reshape(-1, bev_cfg.bev_h_*bev_cfg.bev_w_, bev_cfg._dim_)
        loss = bev_model(return_loss=False, only_bev=False, given_bev=x_, return_eval_loss=True, **kwargs)
        gradient = torch.autograd.grad(loss, x_)[0]
        gradient = gradient.reshape(-1, bev_cfg.bev_h_, bev_cfg.bev_w_, bev_cfg._dim_)
        gradient = gradient.permute(0, 3, 1, 2)
        return gradient    
    
    ds = getattr(dataloader, "dataset", dataset)
    nusc = getattr(ds, "nusc", None)
    if nusc is None:
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(
            version=getattr(ds, "version", "v1.0-trainval"),
            dataroot=getattr(ds, "data_root", "./data/nuscenes"),
            verbose=False
        )
    
    rank, world_size = get_dist_info()

    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    
    for step, batch in enumerate(dataloader):
        # (imgs, mats, _, img_metas, gt_boxes, gt_labels, depth_anything) = batch
        (imgs, mats, _, img_metas, gt_boxes, gt_labels, depth_maps, segmaps) = batch
        
        sample_token = img_metas[0]['token']
        
        # depth = depth_anything.to(device=device)
        depth = depth_maps.to(device)
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.to(device)
            imgs = imgs.to(device); key_img = imgs[:, 0, ...]
            gt_boxes = [gt_box.to(device) for gt_box in gt_boxes]
            gt_labels = [gt_label.to(device) for gt_label in gt_labels]
    
        def get_dino_uncond(cond):
            uncond = {k: v.clone() if isinstance(v, torch.Tensor) else v
                     for k, v in cond.items()}
            last_cls_u = torch.zeros_like(cond['last_cls'])  # (B,V,C_in)
            last_tokens_u = torch.zeros_like(cond['last_tokens'])  # (B,V,N,C_in)
            uncond['last_cls'] = last_cls_u
            uncond['last_tokens'] = last_tokens_u
            return uncond
        
        dino_cond = get_dino(imgs, img_metas)
        dino_uncond = get_dino_uncond(dino_cond) 
        latents = bev_model(imgs, depth, mats, img_metas, only_bev=True, dino_out=dino_cond).detach()
        original_bev = latents.clone().cpu()
        transform_bev = original_bev.transpose(-1, -2).flip(-1)
        # if noise_timesteps > 0:
        #     if noise_timesteps > 1000:
        #         latents = torch.randn_like(latents)
        #         latents = latents * noise_scheduler.init_noise_sigma
        #     else:   
        #         noise = torch.randn_like(latents)
        #         noise_timesteps = torch.as_tensor(noise_timesteps).long()   
        #         latents = noise_scheduler.add_noise(latents, noise, noise_timesteps)
        
        # if denoise_timesteps > 0:    
        #     # # DDIM
        #     # layout_cond, layout_uncond = get_condition(gt_layout, use_cond=True), get_condition(gt_layout, use_cond=False)
        #     noise_scheduler.config.num_train_timesteps=denoise_timesteps
        #     noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)
            
        #     for _, t in enumerate(noise_scheduler.timesteps): 
        #         t_batch = torch.tensor([t] * latents.shape[0], device=latents.device)
        #         noise_pred_uncond, noise_pred_cond = unet(latents, t_batch, mats, dino_uncond)[0], unet(latents, t_batch, mats, dino_cond)[0]
        #         # noise_pred_uncond, noise_pred_cond = unet(latents, t_batch, mats, dino_uncond, **layout_uncond)[0], unet(latents, t_batch, mats, dino_cond, **layout_cond)[0]
        #         noise_pred = noise_pred_uncond + 3 * (noise_pred_cond - noise_pred_uncond)
        #         classifier_gradient = get_classifier_gradient(latents, **batch) if use_classifier_guidence else None
        #         latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False, classifier_gradient=classifier_gradient)[0]
        
        # denoised_bev = latents.detach().clone().cpu().transpose(-1, -2).flip(-1)

        # -------------------------------- PCA -------------------------------- ##
        
        #--------- version 1 : gaussian blur ---------
        # pre_rgb, post_rgb, pca = visualize_bev_rgb_pca_triplet(
        #     original_bev, multi_feat, # (B,C,H,W)
        #     b=0,
        #     bev_extent=extent, origin="lower",
        #     upsample=3, ssaa=True,
        #     blur_sigma=0.8, edge_preserve="bilateral",
        #     interp="bicubic",
        #     out_dir=f"{save_path}/visualize/rgb_pca_triplet_bicubic_multi_scale",
            #     title=f"step {step} | BEV feature (RGB-PCA smooth)",
        #     dpi=300,
        #     nusc=nusc, sample_token=sample_token,
        #     show=False
        # )
        
        #--------- version 2 (DINOV2 style) ---------
        # fig, axes, pca, stats = visualize_bev_pca_and_lidar(
        #     bev_orig_bchw=output_feats1[4],           # (B,C,H,W) torch.Tensor or np.ndarray
        #     bev_denoised_bchw=output_feats3[4],   # (B,C,H,W)
        #     b=0,
        #     out_dir=f"{save_path}/visualize/rgb_pca_bicubic_inter4_t0_t999",
        #     title=f"step {step} | BEV feature inter4 (PCA)",
        #     titles=("inter4 T=0", "inter4 T=100", "LiDAR_TOP"),
        #     show=False,
        #     nusc=nusc,
        #     sample_token=sample_token,
        #     bev_extent=extent,      # LiDAR와 동일한 범위로 맞추고 싶을 때
        #     pca_whiten=False,
        #     pca_clip=(2, 98),
        #     pca_gamma=1.2,
        #     lidar_kwargs=dict(pts_stride=1, lidar_render_mode="scatter"),
        #     dpi=300,
        #     interpolation="bicubic"
        # )
        
        #--------- version 2 (DINOV2 style): compare timesteps ---------
        # fig, axes, pca, stats = visualize_bev_pca_and_lidar_timestep(
        #     bev_t0=out_list[0],
        #     bev_t10=out_list[1],
        #     bev_t100=out_list[2],
        #     bev_t1000=out_list[3],
        #     b=0,
        #     out_dir=f"{save_path}/visualize/rgb_pca_bicubic_inter_feats",
        #     title=f"step {step} | BEV feature inter4 (PCA)",
        #     # titles=("T=0", "T=10", " T=100", "T=1000", "LiDAR_TOP"),
        #     titles=("mid 12×12", "out 12×12", "out 25×25", "out 50×50", "LiDAR Top"),
        #     show=False,
        #     nusc=nusc,
        #     sample_token=sample_token,
        #     bev_extent=extent,      # LiDAR와 동일한 범위로 맞추고 싶을 때
        #     pca_whiten=False,
        #     pca_clip=(2, 98),
        #     pca_gamma=1.2,
        #     lidar_kwargs=dict(pts_stride=1, lidar_render_mode="scatter"),
        #     dpi=300,
        #     interpolation="bicubic"
        # )
        
        #--------- version 2 (DINOV2 style): visualize only 1 feature ---------
        # fig, axes, pca, stats = visualize_single_bev_pca_and_lidar(
        #     bev_bchw=multi_feat,              # (B,C,H,W)
        #     b=0,
        #     nusc=nusc,
        #     sample_token=sample_token,
        #     bev_extent=extent,
        #     out_dir=f"{save_path}/visualize/rgb_pca_inter/rgb_pca_bicubic_multi-concat",
        #     title=f"step {step} | BEV feature multi-concat (PCA)",
        #     show=False,
        #     pca_whiten=False,
        #     pca_clip=(2, 98),
        #     pca_gamma=1.2
        #     )
        
        # xmin, xmax, dx = bev_cfg.backbone_conf['x_bound']
        # ymin, ymax, dy = bev_cfg.backbone_conf['y_bound']
        # H, W = original_bev.shape[-2:]
        # extent = (xmin, xmin + W*dx, ymin, ymin + H*dy)   
        # extent = (xmin, xmin, ymin, ymax)  
         
        xmin, xmax, dx = bev_cfg.backbone_conf['x_bound']
        ymin, ymax, dy = bev_cfg.backbone_conf['y_bound']
        extent = (xmin, xmax, ymin, ymax)
                
        ## ----------------------------- Activation Map ----------------------------- ##
        render_bev_triplet(
            original_bev, transform_bev, b=0,
            nusc=nusc, sample_token=sample_token,
            out_dir=f"{save_path}/visualize/activation_map_lidar_gt_box",
            title=f"step {step} | BEV feature",
            labels=("original", "transformed", "LiDAR Top View"),
            agg="l1", whiten=True, smooth_sigma=0.8,
            joint_clip=(2.0, 98.0), gamma=1.0,   # joint_clip=(1,99)
            bev_cmap="viridis", bev_interp="bilinear",
            bev_extent=extent,           
            bev_origin="lower",          
            lidar_axes_limit=50.0,
                figsize=(15,5), dpi=300, show=False,
            signed=False, signed_clip_pct=98.0,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            class_names=bev_cfg.CLASSES,
            draw_gt_on_bev=True,
            gt_color="r",
        )
        
        if rank == 0:
            prog_bar.update()
        
        # debug_bev_orientation_side_by_side(
        #         ea, nusc, sample_token,
        #         bev_extent=extent,
        #         out_file=f"{save_path}/dbg_orient/orient_grid_step{step}.png",
        #         lidar_pts_size=3.0,     # 더 두껍게
        #         lidar_pts_stride=1,
        #         lidar_pts_alpha=1.0,
        #         show_boxes=True,
        #     )
        

   


  

if __name__ == "__main__":
    test()






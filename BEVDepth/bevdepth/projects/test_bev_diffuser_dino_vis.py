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
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.bevformer.apis.test import custom_encode_mask_results, collect_results_cpu
from mmdet.apis import set_random_seed

from scheduler_utils import DDIMGuidedScheduler
from model_utils import get_bev_model, build_unet, instantiate_from_config
from layout_diffusion.layout_dino_diffusion_unet import LayoutDiffusionUNetModel
from projects.bevdiffuser.fm_feature import GetDINOv2Cond
from projects.bevdiffuser.visualize.bev_visualize import *
from projects.bevdiffuser.visualize.bev_visualize_multi_scale import *

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
        
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMGuidedScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    if args.prediction_type is not None:
        noise_scheduler.register_to_config(prediction_type=args.prediction_type)
    
    bev_model = get_bev_model(args)
    if not args.use_classifier_guidence:
        bev_model.requires_grad_(False)
    bev_model.eval()
    
    unet = build_unet(bev_cfg.unet)
    unet.from_pretrained(args.checkpoint_dir, subfolder="unet")
    unet.to(bev_model.device, dtype=torch.float32)
    unet.requires_grad_(False) 
    unet.eval()

    get_dino = GetDINOv2Cond()
    
    bev_cfg.data.test.test_mode = True
    bev_cfg.data.test.load_annos = True
    dataset = build_dataset(bev_cfg.data.test,
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
  
    # save_path = os.path.join('../../test', args.bev_config.split('/')[-1].split('.')[-2], args.checkpoint_dir.split('/')[-2], args.checkpoint_dir.split('/')[-1])
    save_path = os.path.join('../../../results/stage1', args.checkpoint_dir.split('/')[-2], args.checkpoint_dir.split('/')[-1])       
    
    evaluate(unet=unet,
             bev_model=bev_model,
             get_dino=get_dino,
             noise_scheduler=noise_scheduler,
             dataset=dataset,
             dataloader=dataloader,
             bev_cfg=bev_cfg,
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
    
    def get_condition(batch, use_cond=True):
        cond = {}
        if 'layout_obj_classes' in batch:
            cond['obj_class'] = torch.stack(batch['layout_obj_classes'].data[0])
        if 'layout_obj_bboxes' in batch:
            cond['obj_bbox'] = torch.stack(batch['layout_obj_bboxes'].data[0])
        if 'layout_obj_is_valid' in batch:
            cond['is_valid_obj'] = torch.stack(batch['layout_obj_is_valid'].data[0]) 
        if 'layout_obj_names' in batch:
            cond['obj_name'] = torch.stack(batch['layout_obj_names'].data[0])
        
        if not use_cond:
            if isinstance(unet, LayoutDiffusionUNetModel):
                if 'obj_class' in unet.layout_encoder.used_condition_types:
                    cond['obj_class'] = torch.ones_like(cond['obj_class']).fill_(unet.layout_encoder.num_classes_for_layout_object - 1)
                    cond['obj_class'][:, 0] = unet.layout_encoder.num_classes_for_layout_object - 2
                if 'obj_name' in unet.layout_encoder.used_condition_types:
                    cond['obj_name'] = torch.stack(batch['default_obj_names'].data[0])
                if 'obj_bbox' in unet.layout_encoder.used_condition_types:
                    cond['obj_bbox'] = torch.zeros_like(cond['obj_bbox'])
                    if unet.layout_encoder.use_3d_bbox:
                        cond['obj_bbox'][:, 0] = torch.FloatTensor([0, 0, 0, 1, 1, 1, 0, 0, 0])
                    else:
                        cond['obj_bbox'][:, 0] = torch.FloatTensor([0, 0, 1, 1])
                cond['is_valid_obj'] = torch.zeros_like(cond['is_valid_obj'])
                cond['is_valid_obj'][:, 0] = 1.0 
        for key, value in cond.items():
            if isinstance(value, torch.Tensor):
                cond[key] = value.to(latents.device)            
        return cond
    
    det_res_path = f"{noise_timesteps}_{denoise_timesteps}_{num_inference_steps}"
    bbox_results = []
    mask_results = []
    have_mask = False
    
    ds = getattr(dataloader, "dataset", dataset)
    nusc = getattr(ds, "nusc", None)
    if nusc is None:
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(
            version=getattr(ds, "version", "v1.0-trainval"),
            dataroot=getattr(ds, "data_root", "./data/nuscenes"),
            verbose=False
        )
    
    def rearrange_cam_paths(paths):
        import os
        order = [
            "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT"
        ]
        def cam_key(p):
            parts = os.path.normpath(p).split(os.sep)
            cam = None
            try:
                i = parts.index('samples')
                cam = parts[i+1]
            except Exception:
                pass
            return order.index(cam) if cam in order else len(order)
        paths = list(paths)[:6]
        return sorted(paths, key=cam_key)
    
    det_res_path = f"{noise_timesteps}_{denoise_timesteps}_{num_inference_steps}"
    bbox_results = []
    mask_results = []
    have_mask = False
    
    extent = bev_extent_from_cfg(bev_cfg) 
    
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    
    for step, batch in enumerate(dataloader):
        
        latents = bev_model(return_loss=False, only_bev=True, **batch).detach()
        
        latents = latents.reshape(-1, bev_cfg.bev_h_, bev_cfg.bev_w_, bev_cfg._dim_)
        
        latents = latents.permute(0, 3, 1, 2)
        
        original_bev = latents.detach().clone()  # (B,C,H,W)

        img = batch['img'][0].data[0]
        img_metas = batch['img_metas'][0].data[0]
        sample_token = img_metas[0]['sample_idx']
        img_filenames = img_metas[0]['filename']
        cam_paths = rearrange_cam_paths(img_filenames)

        def get_dino_uncond(cond):
            uncond = {k: v.clone() if isinstance(v, torch.Tensor) else v
                     for k, v in cond.items()}
            last_cls_u = torch.zeros_like(cond['last_cls'])  # (B,V,C_in)
            last_tokens_u = torch.zeros_like(cond['last_tokens'])  # (B,V,N,C_in)
            uncond['last_cls'] = last_cls_u
            uncond['last_tokens'] = last_tokens_u
            return uncond
        
        if noise_timesteps > 0:
            if noise_timesteps > 1000:
                latents = torch.randn_like(latents)
                latents = latents * noise_scheduler.init_noise_sigma
            else:   
                noise = torch.randn_like(latents)
                noise_timesteps = torch.tensor(noise_timesteps).long()   
                latents = noise_scheduler.add_noise(latents, noise, noise_timesteps)
        
        if denoise_timesteps > 0:        
            cond, uncond = get_condition(batch, use_cond=True), get_condition(batch, use_cond=False)
            dino_cond = get_dino(img, img_metas)
            dino_uncond = get_dino_uncond(dino_cond)
            
            # # DDIM
            noise_scheduler.config.num_train_timesteps=denoise_timesteps
            noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        
            for _, t in enumerate(noise_scheduler.timesteps):
                t_batch = torch.tensor([t] * latents.shape[0], device=latents.device)
                noise_pred_uncond, noise_pred_cond = unet(latents, t_batch, dino_uncond, **uncond)[0], unet(latents, t_batch, dino_cond, **cond)[0]
                noise_pred = noise_pred_uncond + 2 * (noise_pred_cond - noise_pred_uncond)
                classifier_gradient = get_classifier_gradient(latents, **batch) if use_classifier_guidence else None
                latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False, classifier_gradient=classifier_gradient)[0]
    
        denoised_bev = latents.detach().clone()
        
        # get detection results
        # latents = latents.permute(0, 2, 3, 1)            
        # latents = latents.reshape(-1, bev_cfg.bev_h_*bev_cfg.bev_w_, bev_cfg._dim_)
        # det_result = bev_model(return_loss=False, only_bev=False, given_bev=latents, rescale=True, **batch)

        # extract multi-scale features
        cond = get_condition(batch, use_cond=True)
        dino_cond = get_dino(img, img_metas)    
        t_test = torch.tensor([0] * latents.shape[0], device=latents.device)  
        # output, multi_feat, out_list = unet(latents, t_test, dino_cond, **cond)
        output, out_list = unet(latents, t_test, dino_cond, **cond)


        ## -------------------------------- PCA -------------------------------- ##
        
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
        #     bev_bchw=inter_feats[8],              # (B,C,H,W)
        #     b=0,
        #     nusc=nusc,
        #     sample_token=sample_token,
        #     bev_extent=extent,
        #     out_dir=f"{save_path}/visualize/rgb_pca_feat/rgb_pca_bicubic_out50_t100",
        #     title=f"step {step} | BEV feature out-block 50x50 (PCA)",
        #     show=False,
        #     pca_whiten=False,
        #     pca_clip=(2, 98),
        #     pca_gamma=1.2
        #     )
        
        
     
        ## ----------------------------- Activation Map ----------------------------- ##
        
        # render_bev_triplet(
        #     original_bev, denoised_bev, b=0,
        #     nusc=nusc, sample_token=sample_token,
        #     out_dir=f"{save_path}/visualize/pca_bev_signed",
        #     title=f"step {step} | BEV feature",
        #     labels=("original", "denoised", "LiDAR Top View"),
        #     agg="l1", whiten=True, smooth_sigma=0.8,
        #     joint_clip=None, gamma=1.0,   # joint_clip=(1,99)
        #     bev_cmap="bwr", bev_interp="bilinear",
        #     bev_extent=extent,           
        #     bev_origin="lower",          
        #     lidar_axes_limit=50.0,
        #     figsize=(15,5), dpi=300, show=False,
        #     signed=True, signed_clip_pct=98.0
        # )
        
        #------- only four multi-scale features --------
        # f1, f2, f3, f4 = output_feats1[2], output_feats2[2], output_feats3[2], output_feats4[2] 
        # render_unet_intermediates(
        #     f1, f2, f3, f4, b=0,
        #     nusc=nusc, sample_token=sample_token,
        #     out_dir=f"{save_path}/visualize/unet_feats_inter4_timestep0-999",
        #     title=f"step {step} | UNet features",
        #     # labels=("mid 12×12", "out 12×12", "out 25×25", "out 50×50", "LiDAR Top"),
        #     labels=("T=0", "T=10", " T=100", "T=999", "LiDAR_TOP"),
        #     # --- choose one ---
        #     mode="energy",          # 채널-집계 에너지 (권장: 비교용)
        #     agg="l1", whiten=True, smooth_sigma=0.8,
        #     joint_clip=(2.0, 98.0), gamma=1.0,
        #     bev_cmap="viridis", bev_interp="bilinear",
        #     bev_extent=extent, bev_origin="lower",
        #     lidar_axes_limit=50.0, lidar_view=np.eye(4), lidar_show_boxes=True,
        #     figsize=(22,4.5), dpi=300, show=False,
        # )
        
        #------- original feature & multi-scale features & concat feature -------
        # f1, f2, f3, f4 = out_list
        # render_unet_intermediates_four(
        #     pre_bchw=original_bev,          # ← pre-UNet
        #     f1_bchw=f1, f2_bchw=f2, f3_bchw=f3, f4_bchw=f4,  # 중간/출력들
        #     concat_bchw=output,       # ← multi-scale concat
        #     b=0,
        #     nusc=nusc, sample_token=sample_token,
        #     out_dir=f"{save_path}/visualize/unet_intermediates",
        #     title=f"step {step} | UNet features",
        #     mode="energy", agg="l1", whiten=True,
        #     smooth_sigma=0.8,
        #     joint_clip=(2,98), gamma=1.0,
        #     bev_cmap="viridis", bev_interp="bilinear",
        #     bev_extent=extent, bev_origin="lower",
        #     lidar_axes_limit=50.0,
        #     figsize=(30, 5), dpi=300, show=False
        # )
        
        # f2, f3, f4 = out_list
        # render_unet_intermediates_three(
        #     pre_bchw=original_bev,          # ← pre-UNet
        #     f2_bchw=f2, f3_bchw=f3, f4_bchw=f4,  # 중간/출력들
        #     concat_bchw=multi_feat,       # ← multi-scale concat
        #     b=0,
        #     nusc=nusc, sample_token=sample_token,
        #     out_dir=f"{save_path}/visualize/unet_intermediates_no-mid_v2",
        #     title=f"step {step} | UNet features",
        #     mode="energy", agg="l1", whiten=True,
        #     smooth_sigma=0.8,
        #     joint_clip=(2,98), gamma=1.3,
        #     bev_cmap="viridis", bev_interp="bilinear",
        #     bev_extent=extent, bev_origin="lower",
        #     lidar_axes_limit=50.0,
        #     figsize=(30, 5), dpi=300, show=False
        # )
        
        # render_sixcams_lidar_bev(
        #     pre_bchw=original_bev,          # ← pre-UNet
        #     f1_bchw=f1, f2_bchw=f2, f3_bchw=f3, f4_bchw=f4,  # 중간/출력들
        #     concat_bchw=multi_feat,       # ← multi-scale concat
        #     b=0,
        #     nusc=nusc, sample_token=sample_token,
        #     out_dir=f"{save_path}/visualize/unet_intermediates_img",
        #     title=f"step {step} | UNet features",
        #     mode="energy", agg="l1", whiten=True,
        #     smooth_sigma=0.8,
        #     joint_clip=(2,98), gamma=1.0,
        #     bev_cmap="viridis", bev_interp="bilinear",
        #     bev_extent=extent, bev_origin="lower",
        #     lidar_axes_limit=50.0,
        #     cam_image_paths=cam_paths,
        #     figsize=(26, 10), dpi=300, show=False
        # )
        
        #------- original feature & all inter features & concat feature -------
        inter_list = out_list
        # inter_list.append(output)
        res = render_unet_intermediates_all(
                inter_list=inter_list,    # 작은 해상도 -> 큰 해상도
                pre_bchw=original_bev,
                concat_bchw=output,
                b=0,
                nusc=nusc, sample_token=sample_token,
                out_dir=f"{save_path}/visualize/unet_intermediates_all_t0",
                title=f"step {step} | UNet Intermediates (Energy)",
                mode="energy", agg="l1", whiten=True,
                smooth_sigma=0.8,
                joint_clip=(2,98), gamma=1.0,
                bev_cmap="viridis", bev_interp="bilinear",
                bev_extent=extent, bev_origin="lower",
                lidar_axes_limit=50.0,
                figsize=(22, 8), dpi=300, show=False,
            )
  

if __name__ == "__main__":
    test()






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
from bevdepth.projects.utils.data_utils import CustomNuScenesDiffusionDataset, collate_fn, DistributedGroupSampler
# from bevdepth.projects.mmdet3d_plugin.datasets.builder import build_dataloader
# from bevdepth.projects.mmdet3d_plugin.bevformer.apis.test import custom_encode_mask_results, collect_results_cpu
from mmdet.apis import set_random_seed

from bevdepth.projects.utils.scheduler_utils import DDIMGuidedScheduler
from bevdepth.projects.utils.model_utils import get_bev_model, build_unet, instantiate_from_config, get_bevdepth_model
from bevdepth.projects.layout_diffusion.layout_diffusion_unet import LayoutDiffusionUNetModel
from bevdepth.projects.fm_feature import GetDINOV2Feat
from bevdepth.utils.torch_dist import all_gather_object, get_rank, synchronize
from bevdepth.evaluators.det_evaluators import DetNuscEvaluator
from torch.utils.data.distributed import DistributedSampler

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

    # unet = build_unet(bev_cfg.unet)
    unet = instantiate_from_config(bev_cfg.unet)
    unet.from_pretrained(args.checkpoint_dir, subfolder="unet")
    unet.to(device, dtype=torch.float32)
    unet.requires_grad_(False) 
    unet.eval()
    
    get_dino = GetDINOV2Feat()

    # dataset = NuscDetDataset(bev_cfg.data.val)
    dataset = CustomNuScenesDiffusionDataset(bev_cfg.data.val)
    
    rank, world_size = get_dist_info()
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bev_cfg.batch_size_per_device,
        shuffle=False,
        collate_fn=partial(collate_fn, is_return_depth=bev_cfg.data_return_depth, 
                    has_depth_any=bev_cfg.use_da3, use_layout_info=bev_cfg.use_layout, use_semantics=bev_cfg.use_semantics),
        num_workers=4,
        sampler=sampler,
    )
    save_path = os.path.join('../../../results/test', args.bev_config.split('/')[-1].split('.')[-2], args.checkpoint_dir.split('/')[-2], args.checkpoint_dir.split('/')[-1])

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
             use_classifier_guidence=False,
             max_eval_steps=None):
             
    def get_classifier_gradient(x, **kwargs):
        x_ = x.detach().requires_grad_(True)
        x_ = x_.permute(0, 2, 3, 1)
        x_ = x_.reshape(-1, bev_cfg.bev_h_*bev_cfg.bev_w_, bev_cfg._dim_)
        loss = bev_model(return_loss=False, only_bev=False, given_bev=x_, return_eval_loss=True, **kwargs)
        gradient = torch.autograd.grad(loss, x_)[0]
        gradient = gradient.reshape(-1, bev_cfg.bev_h_, bev_cfg.bev_w_, bev_cfg._dim_)
        gradient = gradient.permute(0, 3, 1, 2)
        return gradient    
    
    evaluator = DetNuscEvaluator(class_names=bev_cfg.CLASSES, data_root=bev_cfg.data_root, output_dir=save_path)

    rank, world_size = get_dist_info()

    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    
    all_pred_results = list()
    all_img_metas = list()
    
    for step, batch in enumerate(dataloader):
        (imgs, mats, _, img_metas, gt_boxes, gt_labels, depth_maps, segmaps) = batch
        
        depth = depth_maps.to(device=device)
        segmaps = segmaps.to(device=device)
        
        # if len(depth_labels.shape) == 5:
        #     lidar_depth = depth_labels[:, 0, ...].to(device)
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.to(device)
            imgs = imgs.to(device)
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

        def get_segmaps_uncond(segmaps):
            return torch.zeros_like(segmaps)
        
        dino_cond = get_dino(imgs, img_metas)
        dino_uncond = get_dino_uncond(dino_cond) 
        segmaps_uncond = get_segmaps_uncond(segmaps)
        
        latents = bev_model(imgs, depth, mats, img_metas, only_bev=True, dino_out=dino_cond).detach()
    
        if noise_timesteps > 0:
            if noise_timesteps > 1000:
                latents = torch.randn_like(latents)
                latents = latents * noise_scheduler.init_noise_sigma
            else:   
                noise = torch.randn_like(latents)
                noise_timesteps = torch.as_tensor(noise_timesteps).long()   
                latents = noise_scheduler.add_noise(latents, noise, noise_timesteps)
        
        if denoise_timesteps > 0:    
            # # DDIM
            # layout_cond, layout_uncond = get_condition(gt_layout, use_cond=True), get_condition(gt_layout, use_cond=False)
            noise_scheduler.config.num_train_timesteps=denoise_timesteps
            noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)
            
            for _, t in enumerate(noise_scheduler.timesteps): 
                t_batch = torch.tensor([t] * latents.shape[0], device=latents.device)
                noise_pred_uncond, noise_pred_cond = unet(latents, t_batch, mats, dino_uncond, segmaps_uncond, depth)[0], unet(latents, t_batch, mats, dino_cond, segmaps, depth)[0]
                # noise_pred_uncond, noise_pred_cond = unet(latents, t_batch, mats, dino_uncond, **layout_uncond)[0], unet(latents, t_batch, mats, dino_cond, **layout_cond)[0]
                noise_pred = noise_pred_uncond + 2 * (noise_pred_cond - noise_pred_uncond)
                classifier_gradient = get_classifier_gradient(latents, **batch) if use_classifier_guidence else None
                latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False, classifier_gradient=classifier_gradient)[0]
        
        # get detection results
        preds = bev_model(imgs, depth, mats, img_metas, given_bev=latents) 

        if isinstance(bev_model, torch.nn.parallel.DistributedDataParallel):
            results = bev_model.module.get_bboxes(preds, img_metas)
        else:
            results = bev_model.get_bboxes(preds, img_metas)
        for i in range(len(results)):
            results[i][0] = results[i][0].detach().cpu().numpy()
            results[i][1] = results[i][1].detach().cpu().numpy()
            results[i][2] = results[i][2].detach().cpu().numpy()
            results[i].append(img_metas[i])
    
        for predict_step_output in results:
            all_pred_results.append(predict_step_output[:3])
            all_img_metas.append(predict_step_output[3])
        
        batch_size = len(results)
        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()
    
    synchronize()
    len_dataset = len(dataset)
    all_pred_results = sum(map(list, zip(*all_gather_object(all_pred_results))),[])[:len_dataset]
    all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),[])[:len_dataset]
    
    # if get_rank() == 0:
    #     evaluator.evaluate(all_pred_results, all_img_metas)
    metrics = {}
    if get_rank() == 0:
        metrics = evaluator.evaluate(all_pred_results, all_img_metas)
        if metrics is None:
            metrics = {}
    synchronize()
    return metrics

if __name__ == "__main__":
    test()






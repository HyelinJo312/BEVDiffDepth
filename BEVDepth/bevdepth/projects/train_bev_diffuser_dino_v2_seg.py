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
import logging
import math
import os, sys
import shutil
import random
import itertools
import wandb
import warnings
import accelerate
import datasets
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn.functional as F
import torch.utils.data
import torch.utils.checkpoint
import transformers
import diffusers
import importlib
from functools import partial
from torch.nn.parallel import DistributedDataParallel as DDP, DataParallel
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
import logging
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from packaging import version
from torchvision import transforms
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
# from diffusers.training_utils import EMAModel
from mmcv import Config, DictAction
from torch.utils.data.distributed import DistributedSampler
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,save_checkpoint, wrap_fp16_model)
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/..")
# from projects.mmdet3d_plugin.datasets.builder import build_dataloader
# from bevdepth.datasets.nusc_det_dataset_v2 import NuscDetDataset, collate_fn
from bevdepth.projects.utils.data_utils import CustomNuScenesDiffusionDataset, collate_fn, DistributedGroupSampler
from mmdet.apis import set_random_seed
from bevdepth.projects.utils.scheduler_utils import DDIMGuidedScheduler
from bevdepth.projects.utils.model_utils import get_bev_model, build_unet, instantiate_from_config, get_bevdepth_model
from bevdepth.projects.test_bev_diffuser_dino_v2_seg import evaluate
from torch.utils.tensorboard import SummaryWriter
from bevdepth.projects.fm_feature import GetDINOV2Feat
# from bevdepth.projects.layout_diffusion.diffusion_unet_v2_seg import SPADEResBlock, ResBlock
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# logger = get_logger(__name__, log_level="INFO")

def train():
    args = parse_args()

    bev_cfg = Config.fromfile(args.bev_config)
    if args.cfg_options is not None:
        bev_cfg.merge_from_dict(args.cfg_options)
    
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        force=True,
    )
    logger = get_logger(__name__, log_level="INFO")
    logger.info(accelerator.state, main_process_only=False)

    # change output dir first
    if args.resume_from_checkpoint:
        # change the output dir manually
        resume_ckpt_number = args.resume_from_checkpoint.split("-")[-1]
        args.output_dir = f"{args.output_dir}-resume-{resume_ckpt_number}"
        logger.info(f"change output dir to {args.output_dir}")

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
        
    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)
        
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    
    DDIM_scheduler = DDIMGuidedScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    
    if args.prediction_type is not None:
        noise_scheduler.register_to_config(prediction_type=args.prediction_type)
        DDIM_scheduler.register_to_config(prediction_type=args.prediction_type)
        
    # bev_model = get_bev_model(args)
    bev_model = get_bevdepth_model(bev_cfg, args)
    bev_model.requires_grad_(False)
    if args.task_loss_scale != 0:
        bev_model.head.requires_grad_(True)
    
    def get_task_loss(x, sweep_imgs, sweep_depth, mats, img_metas, gt_boxes, gt_labels):
        preds = bev_model(sweep_imgs, sweep_depth, mats, img_metas, given_bev=x)
        if isinstance(bev_model, torch.nn.parallel.DistributedDataParallel):
            targets = bev_model.module.get_targets(gt_boxes, gt_labels)
            detection_loss = bev_model.module.loss(targets, preds)
        else:
            targets = bev_model.get_targets(gt_boxes, gt_labels)
            detection_loss = bev_model.loss(targets, preds)
        return detection_loss

    unet = instantiate_from_config(bev_cfg.unet)
    # unet = build_unet(bev_cfg.unet)
    if args.pretrained_unet_checkpoint is not None and (os.path.isfile(args.pretrained_unet_checkpoint) or os.path.isdir(args.pretrained_unet_checkpoint)):
        unet.from_pretrained(args.pretrained_unet_checkpoint, subfolder="unet")
        if accelerator.is_main_process:
            print(f"Successfully loaded pretrained unet from {args.pretrained_unet_checkpoint}")
        unet.requires_grad_(True)
        # Freeze specific backbone components
        unet.time_embed.requires_grad_(False)
        unet.input_blocks[0].requires_grad_(False)  # Freeze First Input Convolution
        unet.downsample_blocks.requires_grad_(False)
        unet.upsample_blocks.requires_grad_(False)
        
        frozen_count = 0
        for name, module in unet.named_modules():
            # Use class name string comparison (isinstance fails across different module imports)
            class_name = module.__class__.__name__
            if class_name == 'ResBlock':  # Freeze ResBlock only 
                module.requires_grad_(False)
                frozen_count += 1
        # Print trainable status
        trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in unet.parameters())
        if accelerator.is_main_process:
            print(f"Frozen {frozen_count} backbone blocks (ResBlock).")
            print(f"Trainable parameters: {trainable_params} / {all_params} ({trainable_params/all_params:.2%})")

    assert version.parse(accelerate.__version__) >= version.parse("0.16.0"), "accelerate 0.16.0 or above is required"

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        for i, model in enumerate(models):
            model.save_pretrained(os.path.join(output_dir, "unet"))

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

    def load_model_hook(models, input_dir):
        for i in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()
            model.from_pretrained(os.path.join(input_dir, "unet"))

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    
    # learning_rate = args.learning_rate
    basic_lr_per_img = 2e-4 / 8
    learning_rate = basic_lr_per_img * args.train_batch_size * accelerator.num_processes

    # Create param groups with different learning rates
    # unet (pretrained): 0.1x base_lr, bev_model.head (new): 1x base_lr
    param_groups = [
        {'params': [p for p in unet.parameters() if p.requires_grad], 'lr': learning_rate * 0.1, 'name': 'unet'},
    ]
    if args.task_loss_scale != 0:
        param_groups.append(
            {'params': [p for p in bev_model.head.parameters() if p.requires_grad], 'lr': learning_rate, 'name': 'bev_head'}
        )

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Warmup + CosineAnnealing Scheduler
    # Warmup: first 5% of steps, lr increases from 1% to 100%
    warmup_steps = int(args.max_train_steps * 0.01) * accelerator.num_processes
    max_train_steps = args.max_train_steps * accelerator.num_processes
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    # CosineAnnealing: remaining steps, lr smoothly decays to eta_min
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max_train_steps - warmup_steps, eta_min=1e-6)
    # Sequential: Warmup → CosineAnnealing
    lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]) # type: ignore
    
    accelerator.print(f"  LR Scheduler: Warmup({warmup_steps // accelerator.num_processes} steps) + CosineAnnealing")
    accelerator.print(f"  UNet LR: {learning_rate * 0.1}, BEV Head LR: {learning_rate}")


    with accelerator.main_process_first():
        train_dataset = CustomNuScenesDiffusionDataset(bev_cfg.data.train)
        val_dataset = CustomNuScenesDiffusionDataset(bev_cfg.data.val)
        # train_dataset = NuscDetDataset(bev_cfg.data.train)
        # val_dataset = NuscDetDataset(bev_cfg.data.val)

    if accelerator.num_processes > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=True,
            drop_last=True)
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
        shuffle=(train_sampler is None),
        pin_memory=True,
        # prefetch_factor=1,
        persistent_workers=True,
        collate_fn=partial(collate_fn, is_return_depth=bev_cfg.data_return_depth,
                            has_depth_any=bev_cfg.use_da3, use_layout_info=bev_cfg.use_layout, use_semantics=bev_cfg.use_semantics),
        # collate_fn=partial(collate_fn, is_return_depth=bev_cfg.data_return_depth, has_depth_any=True),
        sampler=train_sampler,
    )    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=False,
        # prefetch_factor=1,
        persistent_workers=True,
        collate_fn=partial(collate_fn, is_return_depth=bev_cfg.data_return_depth, 
                           has_depth_any=bev_cfg.use_da3, use_layout_info=bev_cfg.use_layout, use_semantics=bev_cfg.use_semantics),
        # collate_fn=partial(collate_fn, is_return_depth=bev_cfg.use_fusion, has_depth_any=True),
        sampler=val_sampler,
    )
    
    def get_dino_cond(rand_prob, dino_out):
        if rand_prob < args.uncond_prob:
            uncond = {k: v.clone() if isinstance(v, torch.Tensor) else v
                        for k, v in dino_out.items()}
            last_cls_u = torch.zeros_like(dino_out['last_cls'])  # (B,V,C_in)
            last_tokens_u = torch.zeros_like(dino_out['last_tokens'])  # (B,V,N,C_in)
            uncond['last_cls'] = last_cls_u
            uncond['last_tokens'] = last_tokens_u
            cond = uncond
        else:
            cond = dino_out
        return cond
           
    def get_segmaps_cond(rand_prob, segmaps):
        if rand_prob < args.uncond_prob_seg:
            cond = torch.zeros_like(segmaps)
        else:
            cond = segmaps
        return cond

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    
    unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer, lr_scheduler)

    weight_dtype = torch.float32

    # Move text_encode and vae to gpu and cast to weight_dtype
    bev_model.to(accelerator.device, dtype=weight_dtype)
    
    # Get DINOv2 feature extractor
    get_dino = GetDINOV2Feat()

    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    tb_writer = None
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:   
        tracker_config = dict(vars(args))

        if args.resume_from_checkpoint:
            resume_ckpt_number = args.resume_from_checkpoint.split("-")[-1]
            args.tracker_run_name = f"{args.tracker_run_name}-resume-{resume_ckpt_number}"

        init_kwargs = {}
        if args.report_to == "wandb":
            wandb.init(project=args.tracker_project_name,
                       name=args.tracker_run_name,
                       id=args.tracker_run_name)
            init_kwargs = {
                "wandb" : {
                    "name" : args.tracker_run_name
                }
            }

        accelerator.init_trackers(project_name=args.tracker_project_name, 
                                  config=tracker_config,
                                  init_kwargs=init_kwargs)
        
        if args.report_to == "tensorboard":
            tb_logdir = os.path.join(args.output_dir, "tensorboard_logs")
            tb_writer = SummaryWriter(log_dir=tb_logdir)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    is_training_sd21 = args.pretrained_model_name_or_path == "stabilityai/stable-diffusion-2-1"

    logger.info("***** Running training *****")
    logger.info(f"  Num accelerator processes = {accelerator.num_processes}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num dataloader = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Is SD21: {is_training_sd21}")

    accelerator.print("***** Running training *****")
    accelerator.print(f"  Num accelerator processes = {accelerator.num_processes}")
    accelerator.print(f"  Num examples = {len(train_dataset)}")
    accelerator.print(f"  Num Epochs = {args.num_train_epochs}")
    accelerator.print(f"  Num update steps per epoch = {num_update_steps_per_epoch}")
    accelerator.print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    accelerator.print(f"  Train datalooader sampler: {train_dataloader.sampler}")
    accelerator.print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    accelerator.print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    accelerator.print(f"  Total optimization steps = {args.max_train_steps}")
    accelerator.print(f"  Is SD21: {is_training_sd21}")
    # accelerator.print(f"  lr_scheduler:: schduler_type={args.lr_scheduler}, num_warmup_steps={args.lr_warmup_steps}, num_training_steps={args.max_train_steps}")

    global_step = 0
    first_epoch = 0
    step_cnt = 0
    step_threshold = args.enable_task_loss

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        resume_path = args.resume_from_checkpoint

        logger.info(f"Resuming from checkpoint {resume_path}")
        accelerator.load_state(resume_path)
        global_step = int(resume_path.split("-")[-1])

        resume_global_step = global_step * args.gradient_accumulation_steps
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
        # resume step indicates how many data we should skip in this epoch

        # change step_cnt
        step_cnt = global_step * args.gradient_accumulation_steps

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    device = accelerator.device
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # For Resume from checkpoint, Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % 10 == 0:
                    logger.info(f"skipping data {step} / {resume_step}")
                continue

            with accelerator.accumulate(unet):
                (sweep_imgs, mats, _, img_metas, gt_boxes, gt_labels, depth_maps, segmaps) = batch
                
                # Depth Anything or LiDAR depth
                sweep_depth = depth_maps.to(device=device) 
                segmaps = segmaps.to(device=device)
       
                # if len(depth_labels.shape) == 5:
                #     lidar_depth = depth_labels[:, 0, ...].to(device=device, non_blocking=True)
                if torch.cuda.is_available():
                    for key, value in mats.items():
                        mats[key] = value.to(device=device)
                    sweep_imgs = sweep_imgs.to(device=device)
                    gt_boxes = [gt_box.to(device=device) for gt_box in gt_boxes]
                    gt_labels = [gt_label.to(device=device) for gt_label in gt_labels]

                # DINO   
                rand_prob = np.random.rand()
                dino_out = get_dino(sweep_imgs, img_metas)
                dino_cond = get_dino_cond(rand_prob, dino_out)
                seg_cond = get_segmaps_cond(rand_prob, segmaps)
                # layout_cond = get_condition(gt_layout)
                
                # Get BEV
                with torch.no_grad():
                    latents = bev_model(sweep_imgs, sweep_depth, mats, img_metas, only_bev=True, dino_out=dino_out).detach()
                latents = latents.contiguous()

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                max_timestep = noise_scheduler.config.num_train_timesteps
                timesteps = torch.randint(0, max_timestep, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                # add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "sample":
                    target = latents
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                # Predict the noise residual and compute loss
                # model_pred = unet(noisy_latents, timesteps, mats, dino_cond, **layout_cond)[0]
                model_pred = unet(noisy_latents, timesteps, mats, dino_cond, seg_cond, sweep_depth)[0]

                if args.diffusion_loss_scale > 0:
                    denoise_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    denoise_loss = 0
                
                # if args.task_loss_scale > 0 and noise_scheduler.config.prediction_type == "sample" and global_step > step_threshold:
                if args.task_loss_scale > 0 and noise_scheduler.config.prediction_type == "sample":
                    task_loss = get_task_loss(model_pred, sweep_imgs, sweep_depth, mats, img_metas, gt_boxes, gt_labels)
                else:
                    task_loss = 0
                    
                total_loss = args.diffusion_loss_scale * denoise_loss + args.task_loss_scale * task_loss
                # total_loss = args.task_loss_scale * task_loss

                # get learning rate 
                # lr = lr_scheduler.get_last_lr()[0]
                lr_unet = optimizer.param_groups[0]['lr']
                lr_bev_head = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else None

                step_cnt += 1

                loss_dict = {
                    "step/step_cnt" : step_cnt,
                    "step/epoch": epoch,
                    "lr/lr_unet" : lr_unet,
                    "train/denoise_loss": denoise_loss,
                    "train/task_loss": task_loss,
                    "train/total_loss": total_loss,
                }

                if lr_bev_head is not None:
                    loss_dict["lr/lr_bev_head"] = lr_bev_head
                
                if accelerator.is_main_process:
                    for name, value in loss_dict.items():
                        if args.report_to == "wandb":
                            wandb.log({name : value}, step=step_cnt) 
                        elif args.report_to == "tensorboard":
                            tb_writer.add_scalar(name, value, global_step=step_cnt)

                loss = total_loss 

                # Gather the losses across all processes for logging (if we use distributed training).
                # avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                avg_loss = accelerator.reduce(loss, reduction="mean")
                                
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)

                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # save checkpoint 
                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        accelerator.save_state(save_path)
                        save_checkpoint(bev_model, filename=os.path.join(save_path, "bev_model.pth"))
                        logger.info(f"Saved state to {save_path}")
                        
                    unet.eval()
                    if global_step in [50000, 100000, 150000, 200000] and args.task_loss_scale > 0:
                        logger.info(f"Evaluating at epoch {epoch} step {global_step}")
                        with torch.no_grad():
                            eval_path = os.path.join(save_path, 'val')
                            eval_results = evaluate(unet=unet.module,
                                                    bev_model=bev_model,
                                                    get_dino=get_dino,
                                                    noise_scheduler=DDIM_scheduler,
                                                    dataset=val_dataset,
                                                    dataloader=val_dataloader,
                                                    bev_cfg=bev_cfg,
                                                    save_path=eval_path,
                                                    device=accelerator.device,
                                                    noise_timesteps=5,
                                                    denoise_timesteps=5,
                                                    num_inference_steps=5,
                                                    use_classifier_guidence=False)

                        # if accelerator.is_main_process and args.report_to == "wandb":
                        #     for metric, score in eval_results.items():
                        #         metric = f"val/{metric}"
                        #         wandb.log({metric: score}, step=step_cnt)   
                        # if accelerator.is_main_process and args.report_to == "tensorboard":
                        #     for metric, score in eval_results.items():
                        #         metric = f"val/{metric}"
                        #         tb_writer.add_scalar(f"val/{metric}", score, global_step=step_cnt)
                        
                        if accelerator.is_main_process and args.report_to == "wandb" and eval_results:
                            for metric, score in eval_results.items():
                                wandb.log({f"val/{metric}": score}, step=step_cnt)
                        if accelerator.is_main_process and args.report_to == "tensorboard" and eval_results:
                            for metric, score in eval_results.items():
                                tb_writer.add_scalar(f"val/{metric}", score, global_step=step_cnt)
                    unet.train()                         

            logs = {
                "loss": loss.detach().item(), 
                "lr_unet": optimizer.param_groups[0]['lr'],
                "epoch": epoch
            }
            if len(optimizer.param_groups) > 1:
                logs["lr_head"] = optimizer.param_groups[1]['lr']
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()

    accelerator.end_training()
    if tb_writer is not None:
        tb_writer.close()   
    
    
def parse_args():
     # put all arg parse here
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    parser.add_argument('--bev_config', 
                        default="",
                        help='test config file path')
    
    parser.add_argument('--bev_checkpoint', 
                        default="",
                        help='checkpoint file')
    
    parser.add_argument('--pretrained_unet_checkpoint', 
                        default=None,
                        help='checkpoint file')
    
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    
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
        "--uncond_prob", 
        default=0.2, 
        type=float, 
        help="The probability of replacing caption with empty string."
    )
    
    parser.add_argument(
        "--uncond_prob_seg", 
        default=0.2, 
        type=float, 
        help="The probability of replacing caption with empty string."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        # required=True,
        default='results/test',
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )


    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )

    parser.add_argument("--num_train_epochs", type=int, default=100)

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=24500,
        # required=True,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        # required=True,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )

    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )


    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=6,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="sample",
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'sample' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
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
        "--report_to",
        type=str,
        default=None,
        choices=[None, "wandb", "tensorboard"],
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        # required=True,
        default=8000,
        help=(
            "Save a checkpoint of the training state every X updates."
        ),
    )

    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        # required=True,
        default=10,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default=None
    )

    parser.add_argument(
        "--tracker_run_name",
        type=str,
        default=None
    )

    # # below are additional params
    parser.add_argument(
        "--task_loss_scale", 
        type=float, 
        default=0.0
    )
    
    
    parser.add_argument(
        "--depth_save_dir",
        type=str,
        default=None
    )
    
    parser.add_argument(
        "--depth_dir",
        type=str,
        default=None
    )

    parser.add_argument(
        "--semantic_dir",
        type=str,
        default=None
    )

    parser.add_argument(
        "--diffusion_loss_scale", 
        type=float, 
        default=0.0
    )
    
    parser.add_argument(
        "--enable_task_loss",
        type=int,
        default=30000,
        help=("Enable task loss after certain steps."),
    )
    

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options
    return args

if __name__ == "__main__":
    train()






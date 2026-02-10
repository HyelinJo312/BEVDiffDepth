#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

GPUS=4
PORT=${PORT:-29501}

BEV_CONFIG="./configs/bevdiffuser/dinobevdepth_sweeps_seg_v3.py"
PRETRAINED_MODEL="stabilityai/stable-diffusion-2-1"
PRETRAINED_UNET_CHECKPOINT="../../../results/stage1/BEVDiffDepth_constant_diff-1_task-0_seg_62_SPADEupdate_lidar/checkpoint-50000"
# PRETRAINED_UNET_CHECKPOINT="None"

TRAINING_PHASE="finetuning"

# set up wandb project
PROJ_NAME=BEVDiffuser
RUN_NAME=BEVDiffDepth_cosine_diff-0_task-1_seg_62_SPADEupdate_lidar
CHECKPOINT_STEP=50000
CHECKPOINT_LIMIT=5

# allow 500 extra steps to be safe
MAX_TRAINING_STEPS=100000
TRAIN_BATCH_SIZE=2
DATALOADER_NUM_WORKERS=4
GRADIENT_ACCUMMULATION_STEPS=1
MAX_GRAD_NORM=3.0  # 5-> 1.0

# loss and lr settings
LEARNING_RATE=1e-4
LR_SCHEDULER="cosine" # 'warmup_cosine', 'warmup_multistep', 'multistep', 'cosine'

UNCOND_PROB=0.2   # 0.2 -> 0.1
UNCOND_PROB_SEG=0.1   # 0.2 -> 0.1
PREDICTION_TYPE="sample" # "sample", "epsilon" or "v_prediction"
TASK_LOSS_SCALE=1.0 # 0.1
DIFFUSION_LOSS_SCLAE=0
OUTPUT_DIR="../../../results/stage2/${RUN_NAME}"
# RESUME_FROM="../../../results/stage1/BEVDiffDepth_cam-aware_dino_cosine_diff-1_task-1_seg_62/checkpoint-50000"

mkdir -p $OUTPUT_DIR

# export NCCL_SOCKET_IFNAME=lo
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_SHM_DISABLE=1
# export NCCL_DEBUG=INFO
# export NCCL_ASYNC_ERROR_HANDLING=1
# export PYTHONWARNINGS="ignore"
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# export TORCH_DISTRIBUTED_DEBUG="DETAIL"

# train!
PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
torchrun --nproc_per_node $GPUS \
    --master_port=29503 \
  $(dirname "$0")/finetune_v3_seg.py \
    --bev_config $BEV_CONFIG \
    --pretrained_unet_checkpoint $PRETRAINED_UNET_CHECKPOINT \
    --pretrained_model_name_or_path $PRETRAINED_MODEL \
    --training_phase $TRAINING_PHASE \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --gradient_accumulation_steps $GRADIENT_ACCUMMULATION_STEPS \
    --max_grad_norm $MAX_GRAD_NORM \
    --max_train_steps $MAX_TRAINING_STEPS \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler $LR_SCHEDULER \
    --output_dir $OUTPUT_DIR \
    --checkpoints_total_limit $CHECKPOINT_LIMIT \
    --checkpointing_steps $CHECKPOINT_STEP \
    --tracker_run_name $RUN_NAME \
    --tracker_project_name $PROJ_NAME \
    --uncond_prob $UNCOND_PROB \
    --uncond_prob_seg $UNCOND_PROB_SEG \
    --prediction_type $PREDICTION_TYPE \
    --task_loss_scale $TASK_LOSS_SCALE \
    --diffusion_loss_scale $DIFFUSION_LOSS_SCLAE \
    --report_to 'tensorboard' \
    # --resume_from_checkpoint $RESUME_FROM
    # --depth_dir $DEPTH_DIR \
    # --resume_from_checkpoint $RESUME_FROM
    # --bev_checkpoint $BEV_CHECKPOINT 
    # --resume_from_checkpoint $RESUME_FROM
    # --gradient_checkpointing \



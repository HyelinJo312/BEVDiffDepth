set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3

BEV_CONFIG="./configs/bevdiffuser/dinobevdepth_sweeps_seg_v2.py"

CHECKPOINT_DIR="../../../results/stage2/BEVDiffDepth_cosine_diff-1_task-1_seg_28_SPADEupdate_lidar/checkpoint-100000"

BEV_CHECKPOINT="../../../results/stage2/BEVDiffDepth_cosine_diff-1_task-1_seg_28_SPADEupdate_lidar/checkpoint-100000/bev_model.pth"
# "../../ckpts/bevformer_tiny_epoch_24.pth" 

PREDICTION_TYPE="sample"

# export NCCL_SOCKET_IFNAME=lo
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_SHM_DISABLE=1 
# export NCCL_DEBUG=INFO
# export NCCL_ASYNC_ERROR_HANDLING=1


# python -m torch.distributed.launch --master_port 9995 test_bev_diffuser_dino.py \
torchrun --nproc_per_node=4 \
    --master_port 9995 \
    test_bev_diffuser_dino_v2_seg.py \
    --bev_config $BEV_CONFIG \
    --bev_checkpoint $BEV_CHECKPOINT \
    --checkpoint_dir $CHECKPOINT_DIR \
    --prediction_type $PREDICTION_TYPE \
    --noise_timesteps 5 \
    --denoise_timesteps 5 \
    --num_inference_steps 5 \
    # --use_classifier_guidence \



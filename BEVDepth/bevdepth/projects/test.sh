set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3

BEV_CONFIG="../configs/bevdiffuser/layout_tiny_dino_2d.py"

CHECKPOINT_DIR="results/stage1/BEVDiffuser_tiny_GT-dino_only-dino/checkpoint-50000"

BEV_CHECKPOINT="results/stage1/BEVDiffuser_tiny_GT-dino_only-dino/checkpoint-50000/bev_model.pth"
# "../../ckpts/bevformer_tiny_epoch_24.pth" 

PREDICTION_TYPE="sample"

export NCCL_SOCKET_IFNAME=lo
export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_SHM_DISABLE=1 
# export NCCL_DEBUG=INFO
# export NCCL_ASYNC_ERROR_HANDLING=1


# python -m torch.distributed.launch --master_port 9995 test_bev_diffuser_dino.py \
torchrun --nproc_per_node=4 \
    --master_port 9995 \
    test_bev_diffuser_dino.py \
    --bev_config $BEV_CONFIG \
    --bev_checkpoint $BEV_CHECKPOINT \
    --checkpoint_dir $CHECKPOINT_DIR \
    --prediction_type $PREDICTION_TYPE \
    --noise_timesteps 5 \
    --denoise_timesteps 5 \
    --num_inference_steps 5 \
    --inversion False \
    # --use_classifier_guidence \



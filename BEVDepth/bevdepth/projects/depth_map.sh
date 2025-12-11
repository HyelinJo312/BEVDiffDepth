set -e

# export CUDA_VISIBLE_DEVICES=2

BEV_CONFIG="../configs/bevdiffuser/dino_tiny.py"


PREDICTION_TYPE="sample"

export NCCL_SOCKET_IFNAME=lo
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
# export NCCL_IB_DISABLE=1
# export NCCL_SHM_DISABLE=1 
# export NCCL_DEBUG=INFO
# export NCCL_ASYNC_ERROR_HANDLING=1

# python -m torch.distributed.launch --master_port 9995 extract_depth_map.py \
torchrun --nproc_per_node=4 \
    --master_port 9995 \
    extract_depth_map.py \
    --bev_config $BEV_CONFIG \
    # --use_classifier_guidence \



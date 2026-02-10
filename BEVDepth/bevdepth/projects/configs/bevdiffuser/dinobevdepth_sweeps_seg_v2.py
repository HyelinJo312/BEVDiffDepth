_base_ = [
    '../_base_/default_runtime.py'
]

import os

H = 900
W = 1600
# final_dim = (448, 798)
final_dim = (252, 700)
img_conf = dict(img_mean=[123.675, 116.28, 103.53],
                img_std=[58.395, 57.12, 57.375],
                to_rgb=True)
                
_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1
bev_h_ = 128
bev_w_ = 128
bev_dim = 80

CLASSES = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
]

# layout encoder
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
num_bboxes = 300
num_classes = len(CLASSES) + 2
use_3d_bbox = True


backbone_conf = {
    'x_bound': [-51.2, 51.2, 0.8],
    'y_bound': [-51.2, 51.2, 0.8],
    'z_bound': [-5, 3, 8],
    'd_bound': [2.0, 58.0, 0.5],
    'final_dim': final_dim,
    'output_channels': bev_dim,
    'downsample_factor': 14,  # 16  
    'use_soft_depth': True,
}

ida_aug_conf = {
    # 'resize_lim': (0.386, 0.55),
    # 'resize_lim': (0.50, 0.50),
    'resize_lim': (0.44, 0.55),
    'final_dim': final_dim,
    'rot_lim': (-5.4, 5.4),
    # 'rot_lim': (0.0, 0.0),
    'H': H,
    'W': W,
    'rand_flip': True,
    # 'rand_flip': False,
    'bot_pct_lim': (0.0, 0.0),
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams': 6,
}

bda_aug_conf = {
    'rot_lim': (-22.5, 22.5),
    # 'rot_lim': (0, 0),
    'scale_lim': (0.95, 1.05),
    # 'scale_lim': (1.0, 1.0),
    'flip_dx_ratio': 0.5, # 0.5 -> 0
    'flip_dy_ratio': 0.5 # 0.5 -> 0
}

bev_backbone = dict(
    type='ResNet',
    in_channels=bev_dim*2,# 256
    depth=18,
    num_stages=3,
    strides=(1, 2, 2),
    dilations=(1, 1, 1),
    out_indices=[0, 1, 2],
    norm_eval=False,
    base_channels=bev_dim*2, # 256
)

bev_neck = dict(type='SECONDFPN',
                # in_channels=[256, 256, 512, 1024],
                in_channels=[160, 160, 320, 640],
                upsample_strides=[1, 2, 4, 8],
                out_channels=[64, 64, 64, 64])

# unet = dict(
#     type='layout_diffusion.diffusion_unet_v2.UNetModel',
#     parameters=dict(
#         image_size=bev_h_,
#         use_fp16=False,
#         use_scale_shift_norm=True,
#         return_multiscale=False,
#         in_channels=_dim_,
#         out_channels=_dim_,
#         model_channels=256,
#         context_dim=256,  # 768 (original DINOv2)
#         # encoder_channels=256, # assert same as layout_encoder.hidden_dim
#         num_head_channels=32,
#         num_heads=-1,
#         num_heads_upsample=-1,
#         num_res_blocks=2,
#         num_attention_blocks=1,
#         resblock_updown=True,
#         use_spatial_transformer=True,
#         num_pre_downsample=1,
#         attention_resolutions=[ 4, 2, 1 ],
#         channel_mult=[ 1, 2, 4 ],
#         dropout=0.0,
#         use_checkpoint=False,)
# )

unet = dict(
    # type='layout_diffusion.layout_dino_diffusion_unet.LayoutDiffusionUNetModel',
    type='layout_diffusion.diffusion_unet_v2_seg.DiffusionUNetModel',
    parameters=dict(
        image_size=bev_h_,
        use_fp16=False,
        use_scale_shift_norm=True,
        return_multiscale=False,
        # in_channels=_dim_,
        in_channels=bev_dim*2,
        # out_channels=_dim_,
        out_channels=bev_dim*2,
        model_channels=256,
        context_dim=bev_dim*2,  
        resize_dim=final_dim,
        num_head_channels=32,
        num_heads=-1,
        num_heads_upsample=-1,
        num_res_blocks=2,
        num_attention_blocks=1,
        resblock_updown=True,
        use_spatial_transformer=True,
        num_pre_downsample=1,
        attention_ds=[ 4, 2, 1 ],
        channel_mult=[ 1, 2, 4 ],
        dropout=0.0,
        use_checkpoint=False,)
)


TASKS = [
    dict(num_class=1, class_names=['car']),
    dict(num_class=2, class_names=['truck', 'construction_vehicle']),
    dict(num_class=2, class_names=['bus', 'trailer']),
    dict(num_class=1, class_names=['barrier']),
    dict(num_class=2, class_names=['motorcycle', 'bicycle']),
    dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
]

common_heads = dict(reg=(2, 2),
                    height=(1, 2),
                    dim=(3, 2),
                    rot=(2, 2),
                    vel=(2, 2))

bbox_coder = dict(
    type='CenterPointBBoxCoder',
    post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_num=500,
    score_threshold=0.1,
    out_size_factor=4,
    voxel_size=[0.2, 0.2, 8],
    pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
    code_size=9,
)

train_cfg = dict(
    point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
    grid_size=[512, 512, 1],
    voxel_size=[0.2, 0.2, 8],
    out_size_factor=4,
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    # code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
    code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
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

head_conf = {
    'bev_backbone_conf': bev_backbone,
    'bev_neck_conf': bev_neck,
    'tasks': TASKS,
    'common_heads': common_heads,
    'bbox_coder': bbox_coder,
    'train_cfg': train_cfg,
    'test_cfg': test_cfg,
    'in_channels': 256,  # Equal to bev_neck output_channels.
    'loss_cls': dict(type='GaussianFocalLoss', reduction='mean'),
    'loss_bbox': dict(type='L1Loss', reduction='mean', loss_weight=0.25),
    'gaussian_overlap': 0.1,
    'min_radius': 2,
}


# data_root = 'BEVDepth/data/nuScenes/'
data_root = '../../data/nuScenes/'
data_use_cbgs = False
num_sweeps = 2
sweep_idxes = list()
key_idxes = [-1]
data_return_depth = True
use_fusion = True
use_semantics = True
use_da3 = False
use_layout = False
batch_size_per_device = 1


data = dict(
    batch_size_per_device=1,
    train=dict(
        type='CustomNuScenesDiffusionDataset_layout',
        data_root=data_root,
        info_paths=data_root + 'nuscenes_infos_train_temporal.pkl', 
        depth_path=data_root + 'nuscenes_depth_da3',
        semantic_path=data_root + 'nuscenes_semantic',
        ida_aug_conf=ida_aug_conf,
        bda_aug_conf=bda_aug_conf,
        classes=CLASSES,
        is_train=True,
        use_cbgs=data_use_cbgs,
        img_conf=img_conf,
        num_sweeps=num_sweeps,
        sweep_idxes=sweep_idxes,
        key_idxes=key_idxes,
        return_depth=data_return_depth,
        use_da3=use_da3,
        use_fusion=use_fusion,
        use_semantics=use_semantics,
        use_layout=use_layout,
        downsample_size=final_dim,
        box_type_3d='LiDAR',
        pc_range=point_cloud_range,
        use_3d_bbox=use_3d_bbox,
        num_classes=num_classes,
        num_bboxes=num_bboxes,
        ),
    test=dict(
        type='CustomNuScenesDiffusionDataset_layout',
        data_root=data_root,
        info_paths=data_root + 'nuscenes_infos_val_temporal.pkl',
        depth_path=data_root + 'nuscenes_depth_da3',
        semantic_path=data_root + 'nuscenes_semantic_val',
        ida_aug_conf=ida_aug_conf,
        bda_aug_conf=bda_aug_conf,
        classes=CLASSES,
        is_train=False,
        use_cbgs=data_use_cbgs,
        img_conf=img_conf,
        num_sweeps=num_sweeps,
        sweep_idxes=sweep_idxes,
        key_idxes=key_idxes,
        return_depth=data_return_depth,
        use_da3=use_da3,
        use_fusion=use_fusion,
        use_semantics=use_semantics,
        use_layout=use_layout,
        downsample_size=final_dim,
        box_type_3d='LiDAR',
        pc_range=point_cloud_range,
        use_3d_bbox=use_3d_bbox,
        num_classes=num_classes,
        num_bboxes=num_bboxes,
        ),
    val=dict(
        type='CustomNuScenesDiffusionDataset_layout',
        data_root=data_root,
        info_paths=data_root + 'nuscenes_infos_val_temporal.pkl',
        depth_path=data_root + 'nuscenes_depth_da3',
        semantic_path=data_root + 'nuscenes_semantic_val',
        ida_aug_conf=ida_aug_conf,
        bda_aug_conf=bda_aug_conf,
        classes=CLASSES,
        is_train=False,
        use_cbgs=data_use_cbgs,
        img_conf=img_conf,
        num_sweeps=num_sweeps,
        sweep_idxes=sweep_idxes,
        key_idxes=key_idxes,
        return_depth=data_return_depth,
        use_da3=use_da3,
        use_fusion=use_fusion,
        use_semantics=use_semantics,
        use_layout=use_layout,
        downsample_size=final_dim,
        box_type_3d='LiDAR',
        pc_range=point_cloud_range,
        use_3d_bbox=use_3d_bbox,
        num_classes=num_classes,
        num_bboxes=num_bboxes,
        ),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

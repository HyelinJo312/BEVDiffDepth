# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmcv.parallel import DataContainer as DC
# from bevdepth.datasets.nusc_det_dataset_v2 import NuscDetDataset
from bevdepth.datasets.nusc_det_dataset_v2_temporal import NuscDetDataset
import os
import torch.nn.functional as F

import torch
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmcv.parallel import DataContainer as DC
from bevdepth.datasets.nusc_det_dataset_v2_temporal import NuscDetDataset, img_transform, depth_transform, bev_transform
import os
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from pyquaternion import Quaternion
import mmcv

class CustomNuScenesDiffusionDataset(NuscDetDataset):
    def __init__(self, cfg):
        ida_aug_conf = cfg.ida_aug_conf
        bda_aug_conf = cfg.bda_aug_conf
        classes      = cfg.classes
        data_root    = cfg.data_root
        info_paths   = cfg.info_paths
        # depth_path   = cfg.depth_path
        is_train     = cfg.is_train
        use_cbgs     = cfg.use_cbgs
        num_sweeps   = cfg.num_sweeps
        img_conf     = cfg.img_conf
        return_depth = cfg.return_depth
        sweep_idxes  = cfg.sweep_idxes
        key_idxes    = cfg.key_idxes
        use_fusion   = cfg.use_fusion
        load_depth_dtype = np.float32
        
        super().__init__(
            ida_aug_conf=ida_aug_conf,
            bda_aug_conf=bda_aug_conf,
            classes=classes,
            data_root=data_root,
            info_paths=info_paths,
            # depth_path=depth_path,
            is_train=is_train,
            use_cbgs=use_cbgs,
            num_sweeps=num_sweeps,
            img_conf=img_conf,
            return_depth=return_depth,
            sweep_idxes=sweep_idxes,
            key_idxes=key_idxes,
            use_fusion=use_fusion,
            load_depth_dtype=load_depth_dtype,
        )
        self.ida_aug_conf = cfg.ida_aug_conf
        self.pc_range     = cfg.pc_range
        self.use_3d_bbox  = cfg.use_3d_bbox
        self.layout_num_classes  = cfg.num_classes
        self.num_bboxes   = cfg.num_bboxes
        self.object_names = list(self.classes) + ["__image__", "__null__"]
        # self.object_clips = self.embed_object_names()
        self.depth_path = cfg.depth_path
        self.use_semantics = cfg.use_semantics
        self.semantic_path = cfg.semantic_path
        self.downsample_size = cfg.downsample_size
        self.use_layout = cfg.use_layout
    
    def get_image(self, cam_infos, cams, lidar_infos=None):
        assert len(cam_infos) > 0
        sweep_imgs = list()
        sweep_sensor2ego_mats = list()
        sweep_intrin_mats = list()
        sweep_ida_mats = list()
        sweep_sensor2sensor_mats = list()
        sweep_timestamps = list()
        sweep_lidar_depth = list()
        # Depth Anything 3
        sweep_da3_depth = list()
        
        # New Semantics
        sweep_semantics = list()
        
        if self.return_depth or self.use_fusion:
            sweep_lidar_points = list()
            for lidar_info in lidar_infos:
                lidar_path = lidar_info['LIDAR_TOP']['filename'] 
                lidar_points = np.fromfile(os.path.join(self.data_root, lidar_path), dtype=np.float32, count=-1).reshape(-1, 5)[..., :4]
                # full_path = os.path.join(self.data_root, lidar_path)
                # arr = robust_fromfile(full_path, np.float32)
                # lidar_points = arr.reshape(-1, 5)[..., :4]
                sweep_lidar_points.append(lidar_points)
        for cam in cams:
            imgs = list()
            sensor2ego_mats = list()
            intrin_mats = list()
            ida_mats = list()
            sensor2sensor_mats = list()
            timestamps = list()
            lidar_depth = list()
            da3_depths = list()
            semantics = list()
            
            key_info = cam_infos[0]
            # [Fix] Sample augmentation parameters ONCE per camera
            resize, resize_dims, crop, flip, \
                rotate_ida = self.sample_ida_augmentation(
                    )
            
            for sweep_idx, cam_info in enumerate(cam_infos):

                img = Image.open(
                    os.path.join(self.data_root, cam_info[cam]['filename']))
                # img = Image.fromarray(img)
                w, x, y, z = cam_info[cam]['calibrated_sensor']['rotation']
                # sweep sensor to sweep ego
                sweepsensor2sweepego_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                sweepsensor2sweepego_tran = torch.Tensor(
                    cam_info[cam]['calibrated_sensor']['translation'])
                sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros(
                    (4, 4))
                sweepsensor2sweepego[3, 3] = 1
                sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
                sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran
                # sweep ego to global
                w, x, y, z = cam_info[cam]['ego_pose']['rotation']
                sweepego2global_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                sweepego2global_tran = torch.Tensor(
                    cam_info[cam]['ego_pose']['translation'])
                sweepego2global = sweepego2global_rot.new_zeros((4, 4))
                sweepego2global[3, 3] = 1
                sweepego2global[:3, :3] = sweepego2global_rot
                sweepego2global[:3, -1] = sweepego2global_tran

                # global sensor to cur ego
                w, x, y, z = key_info[cam]['ego_pose']['rotation']
                keyego2global_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                keyego2global_tran = torch.Tensor(
                    key_info[cam]['ego_pose']['translation'])
                keyego2global = keyego2global_rot.new_zeros((4, 4))
                keyego2global[3, 3] = 1
                keyego2global[:3, :3] = keyego2global_rot
                keyego2global[:3, -1] = keyego2global_tran
                global2keyego = keyego2global.inverse()

                # cur ego to sensor
                w, x, y, z = key_info[cam]['calibrated_sensor']['rotation']
                keysensor2keyego_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                keysensor2keyego_tran = torch.Tensor(
                    key_info[cam]['calibrated_sensor']['translation'])
                keysensor2keyego = keysensor2keyego_rot.new_zeros((4, 4))
                keysensor2keyego[3, 3] = 1
                keysensor2keyego[:3, :3] = keysensor2keyego_rot
                keysensor2keyego[:3, -1] = keysensor2keyego_tran
                keyego2keysensor = keysensor2keyego.inverse()
                keysensor2sweepsensor = (
                    keyego2keysensor @ global2keyego @ sweepego2global
                    @ sweepsensor2sweepego).inverse()
                sweepsensor2keyego = global2keyego @ sweepego2global @\
                    sweepsensor2sweepego
                sensor2ego_mats.append(sweepsensor2keyego)
                sensor2sensor_mats.append(keysensor2sweepsensor)
                intrin_mat = torch.zeros((4, 4))
                intrin_mat[3, 3] = 1
                intrin_mat[:3, :3] = torch.Tensor(
                    cam_info[cam]['calibrated_sensor']['camera_intrinsic'])
                if self.return_depth and (self.use_fusion or sweep_idx == 0):
                    point_depth = self.get_lidar_depth(
                        sweep_lidar_points[sweep_idx], img,
                        lidar_infos[sweep_idx], cam_info[cam])
                    point_depth_augmented = depth_transform(
                        point_depth, resize, self.ida_aug_conf['final_dim'],
                        crop, flip, rotate_ida)
                    lidar_depth.append(point_depth_augmented)
                img, ida_mat = img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate_ida,
                )
                ida_mats.append(ida_mat)
                img = mmcv.imnormalize(np.array(img), self.img_mean,
                                       self.img_std, self.to_rgb)
                img = torch.from_numpy(img).permute(2, 0, 1)
                imgs.append(img)
                intrin_mats.append(intrin_mat)
                timestamps.append(cam_info[cam]['timestamp'])
                
                # [Optimization] Load Depth with Augmentation
                filename_depth = cam_info[cam]['filename']
                depth_s = self.load_depth_from_filenames(
                    [filename_depth], 
                    resize_dims=resize_dims, 
                    crop=crop, 
                    flip=flip, 
                    rotate=rotate_ida
                ) # Returns (1, H, W)
                da3_depths.append(depth_s.squeeze(0))
                
                # [Fix] Semantics
                if self.semantic_path is not None and sweep_idx == 0:
                     sem_mask = self.load_single_semantic_mask(
                         filename_depth, self.semantic_path, 
                         resize_dims, crop, flip, rotate_ida
                     )
                     semantics.append(sem_mask)
            
            
            sweep_imgs.append(torch.stack(imgs))
            sweep_sensor2ego_mats.append(torch.stack(sensor2ego_mats))
            sweep_intrin_mats.append(torch.stack(intrin_mats))
            sweep_ida_mats.append(torch.stack(ida_mats))
            sweep_sensor2sensor_mats.append(torch.stack(sensor2sensor_mats))
            sweep_timestamps.append(torch.tensor(timestamps))
            if self.return_depth:
                sweep_lidar_depth.append(torch.stack(lidar_depth))
            
            sweep_da3_depth.append(torch.stack(da3_depths)) # (num_sweeps, H, W)
            if self.semantic_path is not None and len(semantics) > 0:
                sweep_semantics.append(semantics[0]) # (H, W) Keyframe only

        sweep_filenames = []
        for cam_info in cam_infos:
            sweep_filenames.append([cam_info[cam]['filename'] for cam in cams])
            
        # Get mean pose of all cams.
        ego2global_rotation = np.mean(
            [key_info[cam]['ego_pose']['rotation'] for cam in cams], 0)
        ego2global_translation = np.mean(
            [key_info[cam]['ego_pose']['translation'] for cam in cams], 0)
        img_metas = dict(
            box_type_3d=LiDARInstance3DBoxes,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation,
            filename=[key_info[cam]['filename'] for cam in cams],
            sweep_filenames=sweep_filenames,          
        )
        
        if lidar_infos is not None:
            key_lidar_info = lidar_infos[0]
            key_cam_infos = cam_infos[0]
            lidar2img_list = []
            for cam in cams:
                lidar2img = self.get_lidar2img(key_lidar_info, key_cam_infos[cam])
                lidar2img_list.append(lidar2img.astype(np.float32))
            img_metas['lidar2img'] = lidar2img_list

        ret_list = [
            torch.stack(sweep_imgs).permute(1, 0, 2, 3, 4),
            torch.stack(sweep_sensor2ego_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_intrin_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_ida_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_sensor2sensor_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_timestamps).permute(1, 0),
            img_metas,
        ]
        if self.return_depth:
            ret_list.append(torch.stack(sweep_lidar_depth).permute(1, 0, 2, 3))
        
        # Append DA3 Depth: (B, V, S, H, W) -> need to check expected output.
        # Original: `da3_depth = torch.stack(sweep_depths, dim=0) # (num_sweeps, V, H, W)`
        # Here sweep_da3_depth is list of (S, H, W) for each Cam V.
        # Stack -> (V, S, H, W). Permute -> (S, V, H, W)
        da3_tensor = torch.stack(sweep_da3_depth).permute(1, 0, 2, 3) 
        ret_list.append(da3_tensor)
        
        # Append Semantics: (B, V, H, W) -> (V, H, W)
        if self.semantic_path is not None and len(sweep_semantics) > 0:
             sem_tensor = torch.stack(sweep_semantics) # (V, H, W)
             ret_list.append(sem_tensor)

        return ret_list

    def __getitem__(self, idx):
        if self.use_cbgs:
            idx = self.sample_indices[idx]

        cams = self.choose_cams()
        frame_indices = self._get_prev_frame_indices(idx, num_frames=self.num_sweeps)

        cam_infos = []
        lidar_infos = []

        for fi in frame_indices:
            info = self.infos[fi]
            cam_infos.append(info['cam_infos'])
            lidar_infos.append(info['lidar_infos'])

        # Call our custom get_image which handles everything
        if self.return_depth or self.use_fusion:
            image_data_list = self.get_image(cam_infos, cams, lidar_infos)
        else:
            image_data_list = self.get_image(cam_infos, cams)
            
        (
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            sweep_timestamps,
            img_metas,
        ) = image_data_list[:7]

        img_metas['token'] = self.infos[idx]['sample_token']
    
        if self.is_train:
            gt_boxes, gt_labels = self.get_gt(self.infos[idx], cams)
        else:
            gt_boxes = sweep_imgs.new_zeros(0, 7)
            gt_labels = sweep_imgs.new_zeros(0, )

        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation(
        )
        bda_mat = sweep_imgs.new_zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot = bev_transform(gt_boxes, rotate_bda, scale_bda,
                                          flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot
        
        current_idx = 7
        if self.return_depth:
            gt_depth = image_data_list[current_idx]
            current_idx += 1
        else:
            gt_depth = None
            
        da3_depth = image_data_list[current_idx]
        current_idx += 1
        
        semantic_maps = None
        if self.semantic_path is not None and len(image_data_list) > current_idx:
            semantic_maps = image_data_list[current_idx]
            current_idx += 1
        
        ret_list = [
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            bda_mat,
            sweep_timestamps,
            img_metas,
            gt_boxes,
            gt_labels,
        ]
        
        if self.return_depth:
             ret_list.append(gt_depth)

        # Append our extras
        if da3_depth is not None:
             ret_list.append(da3_depth)
        
        if semantic_maps is not None:
             ret_list.append(semantic_maps)

        return ret_list

    def load_single_semantic_mask(self, filename, semantic_root, resize_dims, crop, flip, rotate, H=900, W=1600, ignore_label=-1):
        p = filename.lstrip("/")
        parts = p.split("/")
        if len(parts) < 3:
             raise ValueError(f"Bad filename: {p}")
        cam = parts[1]
        stem = os.path.splitext(parts[-1])[0]
        bin_path = os.path.join(semantic_root, "samples", cam, f"{stem}_mask.bin")
        
        try:
            x = np.fromfile(bin_path, dtype=np.int8)
            if x.size != H * W:
                 mask = torch.full((H, W), ignore_label, dtype=torch.int16)
            else:
                 # Keep as numpy or convert to tensor on CPU
                 mask = torch.from_numpy(x.reshape(H, W).astype(np.int16))
        except FileNotFoundError:
             mask = torch.full((H, W), ignore_label, dtype=torch.int16)
        
        # Augment
        resize, resize_dims, crop, flip, rotate = resize_dims, resize_dims, crop, flip, rotate
        # Note: self.semantic_transform expects mask (H, W) -> return (H, W) or (1, H, W)?
        mask_aug = self.semantic_transform(mask, resize_dims, crop, flip, rotate, ignore_label)
        return mask_aug

    def sample_ida_augmentation(self):
        H, W = self.ida_aug_conf['H'], self.ida_aug_conf['W']
        fH, fW = self.ida_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.ida_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.ida_aug_conf['bot_pct_lim'])) *
                newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.ida_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate_ida = np.random.uniform(*self.ida_aug_conf['rot_lim'])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.mean(self.ida_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate_ida = 0
        return resize, resize_dims, crop, flip, rotate_ida
    
    def semantic_transform(self, mask, resize_dims, crop, flip, rotate, ignore_label=-1):
        mask = mask.unsqueeze(1).float() 
        new_w, new_h = resize_dims
        x1, y1, x2, y2 = crop

        mask = F.interpolate(mask, size=(new_h, new_w), mode='nearest')
        mask = mask[:, :, y1:y2, x1:x2]
        if flip:
            mask = torch.flip(mask, dims=[-1])
        if rotate != 0:
            mask = TF.rotate(
                mask, 
                rotate, 
                interpolation=TF.InterpolationMode.NEAREST,
                fill=[float(ignore_label)]
            )
        mask = mask.squeeze(1).to(torch.int16)
        return mask
    
    def load_depth_from_filenames(self, filenames, resize_dims=None, crop=None, flip=False, rotate=0):
        view_depths = []
        for path in filenames:
            filename = path.split('.')[0]
            npy_path = os.path.join(self.depth_path, f"{filename}.npy")
            
            depth_mmap = np.load(npy_path, mmap_mode='r') 
            depth_tensor = torch.from_numpy(depth_mmap).float() 
            
            if resize_dims is not None:
                # depth: (H, W) -> (1, 1, H, W) for interpolation
                depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)
                
                # Resize
                new_w, new_h = resize_dims
                depth_tensor = F.interpolate(depth_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                
                # Crop
                x1, y1, x2, y2 = crop
                # Safety check
                if x2 > x1 and y2 > y1:
                    depth_tensor = depth_tensor[..., y1:y2, x1:x2]
                # Flip
                if flip:
                    depth_tensor = torch.flip(depth_tensor, dims=[-1]) 
                # Rotate
                if rotate != 0:
                     depth_tensor = TF.rotate(
                        depth_tensor, 
                        rotate, 
                        interpolation=TF.InterpolationMode.BILINEAR
                    )
                view_depths.append(depth_tensor.squeeze())
            else:
                 # Fallback if no aug params (should not happen with new get_image)
                 view_depths.append(depth_tensor)
        return torch.stack(view_depths, dim=0)  # (V, H, W)

    
    def get_layout_info(self, gt_boxes: torch.Tensor, gt_labels: torch.Tensor):
        if gt_boxes is None:
            gt_boxes = torch.zeros((0, 9), dtype=torch.float32)
        if gt_labels is None:
            gt_labels = torch.zeros((0,), dtype=torch.long)

        if not torch.is_tensor(gt_boxes):
            gt_boxes = torch.as_tensor(gt_boxes, dtype=torch.float32)
        if not torch.is_tensor(gt_labels):
            gt_labels = torch.as_tensor(gt_labels, dtype=torch.long)

        # ensure shapes
        if gt_boxes.numel() == 0:
            gt_boxes = gt_boxes.reshape(0, 9)
        if gt_labels.numel() == 0:
            gt_labels = gt_labels.reshape(0,)

        assert gt_boxes.dim() == 2 and gt_boxes.size(-1) == 9, \
            f"gt_boxes must be [N, 9], got {tuple(gt_boxes.shape)}"
        assert gt_labels.dim() == 1 and gt_labels.size(0) == gt_boxes.size(0), \
            f"gt_labels must be [N] and match gt_boxes N, got {tuple(gt_labels.shape)} vs N={gt_boxes.size(0)}"

        class_ids = gt_labels
        class_ids = (class_ids + len(self.classes)) % len(self.classes)

        layout_obj_classes = torch.full(
            (self.num_bboxes,), fill_value=self.layout_num_classes - 1, dtype=torch.long
        )
        layout_is_valid = torch.zeros((self.num_bboxes,), dtype=torch.float32)

        # slot 0: special token (기존 코드 유지)
        layout_obj_classes[0] = self.layout_num_classes - 2
        layout_is_valid[0] = 1.0

        default_obj_clip = torch.stack(
            [self.object_clips[int(cid)] for cid in layout_obj_classes]
        )

        # slot 1.. : gt 채우기
        num_valid = min(class_ids.numel(), self.num_bboxes - 1)
        if num_valid > 0:
            layout_obj_classes[1:1 + num_valid] = class_ids[:num_valid]
            layout_is_valid[1:1 + num_valid] = 1.0

        layout_obj_clip = torch.stack(
            [self.object_clips[int(cid)] for cid in layout_obj_classes]
        )

        if self.use_3d_bbox:
            layout_obj_bboxes = self.get_3d_layout_bboxes(gt_boxes)
        else:
            layout_obj_bboxes = self.get_2d_layout_bboxes(gt_boxes)

        layout = {
            "layout_obj_classes": layout_obj_classes,
            "layout_obj_bboxes": layout_obj_bboxes,
            "layout_obj_is_valid": layout_is_valid,
            "layout_obj_names": layout_obj_clip,
            "default_obj_names": default_obj_clip,
        }
        return layout

    
    def normalize_bbox(self, bbox):
        # normalize bbox into [0,1], ego at [0.5, 0.5] 
        x, y = torch.tensor_split(bbox[..., :2], 2, dim=-1)
        x = (x - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        y = (y - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        if bbox.shape[-1] > 2:
            z, x_size, y_size, z_size, yaw, vx, vy = torch.tensor_split(bbox[..., 2:], 7, dim=-1)
            z = (z - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
            x_size = x_size / (self.pc_range[3] - self.pc_range[0])
            y_size = y_size / (self.pc_range[4] - self.pc_range[1])
            z_size = z_size / (self.pc_range[5] - self.pc_range[2])
            return torch.cat((x, y, z, x_size, y_size, z_size, yaw, vx, vy), dim=-1)
        return torch.cat((x, y), dim=-1)
            
    def get_3d_layout_bboxes(self, gt_bboxes): 
        # 3d bbox: (xc, yc, zc, x_size, y_size, z_size, yaw, vx, vy) 
        # ego coordinate, origin at ego position, x towards right, y towards front 
        layout_bboxes = torch.zeros([self.num_bboxes, 9])
        layout_bboxes[0] = torch.FloatTensor([0, 0, 0, 1, 1, 1, 0, 0, 0])
        if isinstance(gt_bboxes, LiDARInstance3DBoxes):
            # (x, y) -> (x-0.5, x-0.5)
            gt_bboxes = self.normalize_bbox(gt_bboxes.tensor)
            gt_bboxes[..., :2] = gt_bboxes[..., :2] - 0.5
            num_valid = min(len(gt_bboxes), self.num_bboxes-1)
            layout_bboxes[1: 1+num_valid] = gt_bboxes
        return layout_bboxes
    
    def get_2d_layout_bboxes(self, gt_bboxes):
        # 2d bbox: (x0, y0, x1, y1), 
        # image coordinate, orgin at upper left, x towards right, y towards down
        layout_bboxes = torch.zeros([self.num_bboxes, 4])
        layout_bboxes[0] = torch.FloatTensor([0, 0, 1, 1])
        if isinstance(gt_bboxes, LiDARInstance3DBoxes):
            gt_bboxes = self.normalize_bbox(gt_bboxes.corners[..., :2]) # N x 8 x 2
            # (x, y) -> (x, 1-y)
            gt_bboxes[..., 1] = 1 - gt_bboxes[..., 1]
            gt_bboxes_min = gt_bboxes.min(dim=1).values # N x 2
            gt_bboxes_max = gt_bboxes.max(dim=1).values # N x 2
            gt_bboxes = torch.cat((gt_bboxes_min, gt_bboxes_max), dim=-1)
            num_valid = min(len(gt_bboxes), self.num_bboxes-1)
            layout_bboxes[1: 1+num_valid] = gt_bboxes
        return layout_bboxes
            


def collate_fn(data, is_return_depth=False, has_depth_any=False, use_layout_info=False, use_semantics=False):
    imgs_batch = list()
    sensor2ego_mats_batch = list()
    intrin_mats_batch = list()
    ida_mats_batch = list()
    sensor2sensor_mats_batch = list()
    bda_mat_batch = list()
    timestamps_batch = list()
    gt_boxes_batch = list()
    gt_labels_batch = list()
    img_metas_batch = list()
    depth_labels_batch = list()
    depth_batch = list()
    layout_batch = list()
    semantics_batch = list()
    for iter_data in data:
        (
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            bda_mat,
            sweep_timestamps,
            img_metas,
            gt_boxes,
            gt_labels,
        ) = iter_data[:10]
        cursor = 10
        # LiDAR
        if is_return_depth:
            gt_depth = iter_data[10]
            depth_labels_batch.append(gt_depth)
            cursor += 1
        # Depth Anything 3
        if has_depth_any and not is_return_depth:
            depth = iter_data[10]          # (V,H,W)
            depth_batch.append(depth)
            cursor += 1
        elif has_depth_any and is_return_depth:
            depth = iter_data[cursor]          # (V,H,W)
            depth_batch.append(depth)
            cursor += 1
        # Grounded SAM 
        if use_semantics:
            sem = iter_data[cursor]    # expected: (V H,W) torch tensor
            semantics_batch.append(sem)
            cursor += 1
        # # GT layout
        # if use_layout_info:
        #     layout = iter_data[cursor]
        #     layout_batch.append(layout)
        #     cursor += 1
        
        imgs_batch.append(sweep_imgs)
        sensor2ego_mats_batch.append(sweep_sensor2ego_mats)
        intrin_mats_batch.append(sweep_intrins)
        ida_mats_batch.append(sweep_ida_mats)
        sensor2sensor_mats_batch.append(sweep_sensor2sensor_mats)
        bda_mat_batch.append(bda_mat)
        timestamps_batch.append(sweep_timestamps)
        img_metas_batch.append(img_metas)
        gt_boxes_batch.append(gt_boxes)
        gt_labels_batch.append(gt_labels)
        
    mats_dict = dict()
    mats_dict['sensor2ego_mats'] = torch.stack(sensor2ego_mats_batch)
    mats_dict['intrin_mats'] = torch.stack(intrin_mats_batch)
    mats_dict['ida_mats'] = torch.stack(ida_mats_batch)
    mats_dict['sensor2sensor_mats'] = torch.stack(sensor2sensor_mats_batch)
    mats_dict['bda_mat'] = torch.stack(bda_mat_batch)
    
    # if use_layout_info:
    #     layout_info = dict()
    #     for info in layout_batch[0].keys():
    #         layout_info[info] = torch.stack([layout_batch[i][info] for i in range(len(layout_batch))])
        
    ret_list = [
        torch.stack(imgs_batch),
        mats_dict,
        torch.stack(timestamps_batch),
        img_metas_batch,
        gt_boxes_batch,
        gt_labels_batch,
    ]
    if is_return_depth:
        ret_list.append(torch.stack(depth_labels_batch))
    if has_depth_any:
        ret_list.append(torch.stack(depth_batch))  # (B,V,H,W)
    if use_semantics:
        ret_list.append(torch.stack(semantics_batch))  # (B,V,H,W)
    # if use_layout_info:
    #     ret_list.append(layout_info)

    return ret_list





# Copyright (c) OpenMMLab. All rights reserved.
import math
import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler
import random
from IPython import embed

class DistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        seed (int, optional): random seed used to shuffle the sampler if
            ``shuffle=True``. This number should be identical across all
            processes in the distributed group. Default: 0.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 seed=0):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed if seed is not None else 0

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += int(
                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                # add .numpy() to avoid bug when selecting indice in parrots.
                # TODO: check whether torch.randperm() can be replaced by
                # numpy.random.permutation().
                indice = indice[list(
                    torch.randperm(int(size), generator=g).numpy())].tolist()
                extra = int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                # pad indice
                tmp = indice.copy()
                for _ in range(extra // size):
                    indice.extend(tmp)
                indice.extend(tmp[:extra % size])
                indices.extend(indice)

        assert len(indices) == self.total_size

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
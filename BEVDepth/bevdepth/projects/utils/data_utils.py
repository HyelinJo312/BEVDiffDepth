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
from bevdepth.datasets.nusc_det_dataset_v2_temporal import NuscDetDataset
import os
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class CustomNuScenesDiffusionDataset(NuscDetDataset):
    def __init__(self, cfg):
        super().__init__(
            ida_aug_conf=cfg.ida_aug_conf,
            bda_aug_conf=cfg.bda_aug_conf,
            classes=cfg.classes,
            data_root=cfg.data_root,
            info_paths=cfg.info_paths,
            # depth_path=cfg.depth_path,
            is_train=cfg.is_train,
            use_cbgs=cfg.use_cbgs,
            num_sweeps=cfg.num_sweeps,
            img_conf=cfg.img_conf,
            return_depth=cfg.return_depth,
            sweep_idxes=cfg.sweep_idxes,
            key_idxes=cfg.key_idxes,
            use_fusion=cfg.use_fusion,
            load_depth_dtype=np.float32,
        )
        self.ida_aug_conf = cfg.ida_aug_conf
        self.pc_range     = cfg.pc_range
        self.use_3d_bbox  = cfg.use_3d_bbox
        self.layout_num_classes  = cfg.num_classes
        self.num_bboxes   = cfg.num_bboxes
        self.object_names = list(self.classes) + ["__image__", "__null__"]
        # self.object_clips = self.embed_object_names()
        self.use_da3 = cfg.use_da3
        self.use_semantics = cfg.use_semantics
        self.semantic_path = cfg.semantic_path
        self.downsample_size = cfg.downsample_size
        self.use_layout = cfg.use_layout
    
    def __getitem__(self, idx):
        data = super().__getitem__(idx) 
        if self.use_da3:
            # Sweep depths (Depth Anything 3)
            sweep_depths = []
            for filenames in data[7]['sweep_filenames']:   
                depth_s = self.load_depth_from_filenames(filenames)  # (V,H,W)
                sweep_depths.append(depth_s)
            da3_depth = torch.stack(sweep_depths, dim=0)  # (num_sweeps,V,H,W)
            data.append(da3_depth)
        
        # Segmentation mask
        if self.semantic_path is not None and os.path.exists(self.semantic_path):
            semantic_maps = self.load_semantic_from_filenames_v2(data[7]['filename'], self.semantic_path, self.downsample_size)
            data.append(semantic_maps)
     
        # layout = self.get_layout_info(data)
        # if self.use_layout:
        #     layout = self.get_layout_info(data[8], data[9])
        #     data.append(layout)
        return data
    
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
    
    def load_depth_from_filenames(self, filenames):
        """
        filenames: list[str]  # e.g. ["samples/CAM_FRONT/123.jpg", ...] length=V
        return: torch.FloatTensor (V, H, W)
        """
        view_depths = []
        for path in filenames:
            filename = path.split('.')[0]
            # cams = ["samples", "CAM_FRONT", "123.jpg"]
            npy_path = os.path.join(self.depth_path, f"{filename}.npy")
            # depth = np.load(npy_path).astype(self.load_depth_dtype, copy=False)
            # view_depths.append(torch.from_numpy(depth).float())
            # Use mmap_mode='r' for faster loading of large arrays without reading into memory immediately
            depth = np.load(npy_path, mmap_mode='r')
            view_depths.append(torch.from_numpy(depth.copy()).float())
        return torch.stack(view_depths, dim=0)  # (V, H, W)

    def load_semantic_from_filenames(
        self,
        filenames,         
        semantic_root,
        downsample_size,
        H=900, W=1600,
        ignore_label=-1,
        dtype=np.int8,
        strict=True,
    ):
        """
        Return:
            semantic_id: torch.Int16Tensor (V, H2, W2)
            - key frame only (T=1)
            - class-id map (unknown = -1)
        """
        out_h, out_w = downsample_size  # (448, 798)

        if len(filenames) == 0:
            # return torch.zeros((1, 0, out_h, out_w), dtype=torch.int16)
            return torch.zeros((0, out_h, out_w), dtype=torch.int16)

        masks = []
        for p in filenames:
            p = p.lstrip("/")
            parts = p.split("/")
            if len(parts) < 3:
                raise ValueError(f"Bad filename: {p}")

            cam = parts[1]
            stem = os.path.splitext(parts[-1])[0]
            bin_path = os.path.join(semantic_root, "samples", cam, f"{stem}_mask.bin")

            try:
                x = np.fromfile(bin_path, dtype=dtype)
                if x.size != H * W:
                    if strict:
                        raise ValueError(f"Size mismatch: {bin_path} ({x.size} != {H*W})")
                    mask = torch.full((H, W), ignore_label, dtype=torch.int16)
                else:
                    # Keep as numpy or convert to tensor on CPU
                    mask = torch.from_numpy(x.reshape(H, W).astype(np.int16))
            except FileNotFoundError:
                if strict:
                    raise FileNotFoundError(bin_path)
                mask = torch.full((H, W), ignore_label, dtype=torch.int16)
            masks.append(mask)
        masks_tensor = torch.stack(masks, dim=0) # (V, H, W)
        
        # (V, 1, H, W) for interpolate
        masks_tensor = masks_tensor.unsqueeze(1).float()
        
        # Interpolate expects (N, C, H, W). Here we treat V as N.
        masks_resized = F.interpolate(     # (V, 1, out_h, out_w)
            masks_tensor,
            size=(out_h, out_w),
            mode="nearest",
        )
        masks_resized = masks_resized.squeeze(1).to(torch.int16)   # (V, out_h, out_w)
        return masks_resized

    def load_semantic_from_filenames_v2(
        self,
        filenames,         
        semantic_root,
        downsample_size,
        H=900, W=1600,
        ignore_label=-1,
        dtype=np.int8,
        strict=True,
    ):
        out_h, out_w = downsample_size  # (448, 798)

        if len(filenames) == 0:
            return torch.zeros((0, out_h, out_w), dtype=torch.int16)
        masks = []
        for p in filenames:
            p = p.lstrip("/")
            parts = p.split("/")
            if len(parts) < 3:
                raise ValueError(f"Bad filename: {p}")

            cam = parts[1]
            stem = os.path.splitext(parts[-1])[0]
            bin_path = os.path.join(semantic_root, "samples", cam, f"{stem}_mask.bin")

            try:
                x = np.fromfile(bin_path, dtype=dtype)
                if x.size != H * W:
                    if strict:
                        raise ValueError(f"Size mismatch: {bin_path} ({x.size} != {H*W})")
                    mask = torch.full((H, W), ignore_label, dtype=torch.int16)
                else:
                    # Keep as numpy or convert to tensor on CPU
                    mask = torch.from_numpy(x.reshape(H, W).astype(np.int16))
            except FileNotFoundError:
                if strict:
                    raise FileNotFoundError(bin_path)
                mask = torch.full((H, W), ignore_label, dtype=torch.int16)
            masks.append(mask)

        masks_tensor = torch.stack(masks, dim=0)
        resize, resize_dims, crop, flip, rotate_ida = self.sample_ida_augmentation()
        masks_resized = self.semantic_transform(masks_tensor, resize_dims, crop, flip, rotate_ida, ignore_label) # (V, H, W)
        return masks_resized



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
            segmap = iter_data[cursor]    # expected: (V H,W) torch tensor
            semantics_batch.append(segmap)
            cursor += 1
        
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
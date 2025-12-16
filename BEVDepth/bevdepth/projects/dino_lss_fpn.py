# Copyright (c) Megvii Inc. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmdet3d.models import build_neck
from mmdet.models import build_backbone
from mmdet.models.backbones.resnet import BasicBlock
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
import numpy as np
try:
    from bevdepth.ops.voxel_pooling_inference import voxel_pooling_inference
    from bevdepth.ops.voxel_pooling_train import voxel_pooling_train
except ImportError:
    print('Import VoxelPooling fail.')
from bevdepth.projects.fm_feature import GetDINOV2Feat


__all__ = ['DINOLSSFPN']


class DINOLSSFPN(nn.Module):

    def __init__(self,
                 x_bound,
                 y_bound,
                 z_bound,
                 d_bound,
                 final_dim,
                 downsample_factor,
                 output_channels,
                #  img_backbone_conf,
                #  img_neck_conf,
                #  depth_net_conf,
                 use_da=False,
                 use_soft_depth=False):
        """Modified from `https://github.com/nv-tlabs/lift-splat-shoot`.

        Args:
            x_bound (list): Boundaries for x.
            y_bound (list): Boundaries for y.
            z_bound (list): Boundaries for z.
            d_bound (list): Boundaries for d.
            final_dim (list): Dimension for input images.
            downsample_factor (int): Downsample factor between feature map
                and input image.
            output_channels (int): Number of channels for the output
                feature map.
            img_backbone_conf (dict): Config for image backbone.
            img_neck_conf (dict): Config for image neck.
            depth_net_conf (dict): Config for depth net.
        """

        super(DINOLSSFPN, self).__init__()
        self.downsample_factor = downsample_factor
        self.d_bound = d_bound
        self.final_dim = final_dim
        self.output_channels = output_channels
        self.use_soft_depth = use_soft_depth
        
        self.register_buffer(
            'voxel_size',
            torch.Tensor([row[2] for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer(
            'voxel_coord',
            torch.Tensor([
                row[0] + row[2] / 2.0 for row in [x_bound, y_bound, z_bound]
            ]))
        self.register_buffer(
            'voxel_num',
            torch.LongTensor([(row[1] - row[0]) / row[2]
                              for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer('frustum', self.create_frustum())
        self.depth_channels, _, _, _ = self.frustum.shape

        # self.img_backbone = build_backbone(img_backbone_conf)
        # self.img_neck = build_neck(img_neck_conf)
        # self.depth_net = self._configure_depth_net(depth_net_conf)

        # self.img_neck.init_weights()
        # self.img_backbone.init_weights()
        
        self.use_da = use_da
        # self.dino = GetDINOV2Feat()


    def _downsample_lidar_depth_bv(self, lidar_depth):
        """
        Input:
            lidar_depth: [B, V, H, W]   (0 = invalid)
        Return:
            depth_min: [B*V, h, w]      (각 patch에서 0이 아닌 depth 중 최소값, 없으면 1e5)
            valid_mask: [B*V, h, w]     (True = 유효 depth 존재)
        """
        B, V, H, W = lidar_depth.shape
        ds = self.downsample_factor

        # [B, V, H, W] -> [B*V, H, W]
        depth = lidar_depth.view(B * V, H, W)

        # [B*V, h, ds, w, ds, 1]
        depth = depth.view(
            B * V,
            H // ds,
            ds,
            W // ds,
            ds,
            1,
        )
        # [B*V, h, w, 1, ds, ds]
        depth = depth.permute(0, 1, 3, 5, 2, 4).contiguous()
        # [B*V*h*w, ds*ds]
        depth = depth.view(-1, ds * ds)

        # 0.0 은 invalid로 간주하고 큰 값으로 치환
        depth_tmp = torch.where(
            depth == 0.0,
            depth.new_full(depth.shape, 1e5),
            depth,
        )
        # patch 내 최소 (0이 아닌 depth 중)
        depth_min = depth_tmp.min(dim=-1).values  # [B*V*h*w]

        h = H // ds
        w = W // ds
        depth_min = depth_min.view(B * V, h, w)   # [B*V, h, w]

        valid_mask = depth_min < 1e5  
        return depth_min, valid_mask, h, w
    
    def get_lidar_depth_one_hot(self, lidar_depth):
        """
        Input:
            lidar_depth: [B, V, H, W]  # 6-view LiDAR depth, 0 = invalid

        Output:
            depth_one_hot: [B*V, D, h, w]
        """
        B, V, H, W = lidar_depth.shape
        ds = self.downsample_factor
        D = self.depth_channels  # depth_channels = number of depth bins

        # [B, V, H, W] -> [B*V, H, W]
        gt_depths = lidar_depth.view(B * V, H, W)

        gt_depths = gt_depths.view(B * V, H // ds, ds, W // ds, ds, 1)
        # permute: [B*V, h, w, 1, ds, ds]
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()

        # [B*V*h*w, ds*ds]
        gt_depths = gt_depths.view(-1, ds * ds)

        gt_depths_tmp = torch.where(
            gt_depths == 0.0,
            1e5 * torch.ones_like(gt_depths),
            gt_depths,
        )
        
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values  # [B*V*h*w]

        # 다시 [B*V, h, w] 로 reshape
        h = H // ds
        w = W // ds
        gt_depths = gt_depths.view(B * V, h, w)

        d_min, d_max, d_interval = self.d_bound

        gt_depths = (gt_depths - (d_min - d_interval)) / d_interval

        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths,
            torch.zeros_like(gt_depths),
        )
  
        gt_depths_one_hot = F.one_hot(
            gt_depths.long(),
            num_classes=self.depth_channels + 1,
        )  # [B*V, h, w, D+1]
        gt_depths_one_hot = gt_depths_one_hot.view(-1, self.depth_channels + 1)

        gt_depths_one_hot = gt_depths_one_hot[:, 1:]  # drop invalid class
        gt_depths_one_hot = gt_depths_one_hot.view(B * V, h, w, D)

        depth_one_hot = gt_depths_one_hot.permute(0, 3, 1, 2).contiguous()
        return depth_one_hot.float()
    
    def get_lidar_depth_soft_one_hot(self, lidar_depth):
        """
        Input:
            lidar_depth: [B, V, H, W]
        Output:
            depth_soft: [B*V, D, h, w]
        """
        B, V, H, W = lidar_depth.shape
        depth_min, valid_mask, h, w = self._downsample_lidar_depth_bv(lidar_depth)
        D = self.depth_channels
        d_min, d_max, d_interval = self.d_bound
        device = lidar_depth.device
        dtype = depth_min.dtype

        pos = (depth_min - d_min) / d_interval    # [B*V, h, w], 실수
        pos = pos.clamp(0.0, float(D - 1e-6))

        lower = torch.floor(pos).long()           # [B*V, h, w]  in [0, D-1]
        upper = torch.clamp(lower + 1, max=D - 1)

        w_upper = (pos - lower.to(pos.dtype)).clamp(0.0, 1.0)
        w_lower = 1.0 - w_upper

        # invalid / 범위 밖 depth는 weight 0
        w_lower = w_lower * valid_mask
        w_upper = w_upper * valid_mask

        depth_soft = torch.zeros(B * V, D, h, w, device=device, dtype=dtype)

        # (B*V, 1, h, w)
        lower_idx = lower.unsqueeze(1)
        upper_idx = upper.unsqueeze(1)
        w_lower_exp = w_lower.unsqueeze(1)
        w_upper_exp = w_upper.unsqueeze(1)

        depth_soft.scatter_add_(1, lower_idx, w_lower_exp)
        depth_soft.scatter_add_(1, upper_idx, w_upper_exp)

        return depth_soft.float()

    def get_lidar_depth_gaussian(self, lidar_depth, sigma=None, eps=1e-6):
        """
        Input:
            lidar_depth: [B, V, H, W]   # 0 = invalid
            sigma:       float or None  # Gaussian 표준편차 (meter 단위)
                                         # None이면 depth bin interval 기준으로 자동 설정
            eps:         float          # numerical 안정성용
        Output:
            depth_gauss: [B*V, D, h, w]
        """
        B, V, H, W = lidar_depth.shape
        device = lidar_depth.device

        # 1) BEVDepth 스타일 downsample (patch 최소값 사용)
        depth_min, valid_mask, h, w = self._downsample_lidar_depth_bv(lidar_depth)
        # depth_min: [B*V, h, w]
        # valid_mask: [B*V, h, w]

        D = self.depth_channels
        d_min, d_max, d_interval = self.d_bound
        depth_min = depth_min.to(torch.float32)

        # sigma 설정 (없으면 bin 간격 기준으로)
        if sigma is None:
            # 예: bin 간격의 1.0~1.5배 정도 (원하면 튜닝 가능)
            sigma = float(d_interval) * 1.0
        sigma = float(sigma)

        # 2) depth bin center 생성: [D]
        #    c_k = d_min + k * d_interval
        bin_centers = torch.arange(
            D, device=device, dtype=depth_min.dtype
        ) * d_interval + d_min  # [D]

        # 3) Gaussian weight 계산
        # depth_min: [B*V, h, w] → [B*V, 1, h, w]
        depth_exp = depth_min.unsqueeze(1)                    # [B*V, 1, h, w]
        centers_exp = bin_centers.view(1, D, 1, 1)            # [1, D, 1, 1]

        # (d - c)^2
        diff = depth_exp - centers_exp                        # [B*V, D, h, w]
        gauss = torch.exp(-0.5 * (diff / sigma) ** 2)         # [B*V, D, h, w]

        # 4) invalid patch는 전부 0으로
        valid_mask_exp = valid_mask.unsqueeze(1)              # [B*V, 1, h, w]
        gauss = gauss * valid_mask_exp                        # invalid → 0

        # 5) depth 차원(D)에 대해 normalize (sum=1)
        sum_gauss = gauss.sum(dim=1, keepdim=True)            # [B*V, 1, h, w]

        # sum=0 (invalid patch)인 곳은 그대로 0 유지
        # sum>0인 곳만 normalize
        norm_factor = sum_gauss + eps
        gauss = torch.where(
            sum_gauss > 0,
            gauss / norm_factor,
            torch.zeros_like(gauss),
        )

        return gauss  # [B*V, D, h, w]


    def _forward_voxel_net(self, img_feat_with_depth):
        if self.use_da:
            # BEVConv2D [n, c, d, h, w] -> [n, h, c, w, d]
            img_feat_with_depth = img_feat_with_depth.permute(
                0, 3, 1, 4,
                2).contiguous()  # [n, c, d, h, w] -> [n, h, c, w, d]
            n, h, c, w, d = img_feat_with_depth.shape
            img_feat_with_depth = img_feat_with_depth.view(-1, c, w, d)
            img_feat_with_depth = (
                self.depth_aggregation_net(img_feat_with_depth).view(
                    n, h, c, w, d).permute(0, 2, 4, 1, 3).contiguous())
        return img_feat_with_depth

    def create_frustum(self):
        """Generate frustum"""
        # make grid in image plane
        ogfH, ogfW = self.final_dim
        fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor
        d_coords = torch.arange(*self.d_bound,
                                dtype=torch.float).view(-1, 1,
                                                        1).expand(-1, fH, fW)
        D, _, _ = d_coords.shape
        x_coords = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(
            1, 1, fW).expand(D, fH, fW)
        y_coords = torch.linspace(0, ogfH - 1, fH,
                                  dtype=torch.float).view(1, fH,
                                                          1).expand(D, fH, fW)
        paddings = torch.ones_like(d_coords)

        # D x H x W x 3
        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
        return frustum

    def get_geometry(self, sensor2ego_mat, intrin_mat, ida_mat, bda_mat):
        """Transfer points from camera coord to ego coord.

        Args:
            rots(Tensor): Rotation matrix from camera to ego.
            trans(Tensor): Translation matrix from camera to ego.
            intrins(Tensor): Intrinsic matrix.
            post_rots_ida(Tensor): Rotation matrix for ida.
            post_trans_ida(Tensor): Translation matrix for ida
            post_rot_bda(Tensor): Rotation matrix for bda.

        Returns:
            Tensors: points ego coord.
        """
        batch_size, num_cams, _, _ = sensor2ego_mat.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum
        ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
        points = ida_mat.inverse().matmul(points.unsqueeze(-1))
        # cam_to_ego
        points = torch.cat(
            (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
             points[:, :, :, :, :, 2:]), 5)

        combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat))
        points = combine.view(batch_size, num_cams, 1, 1, 1, 4,
                              4).matmul(points)
        if bda_mat is not None:
            bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(
                batch_size, num_cams, 1, 1, 1, 4, 4)
            points = (bda_mat @ points).squeeze(-1)
        else:
            points = points.squeeze(-1)
        return points[..., :3]


    def load_pca(self, path, device):
        data = np.load(path)
        mu = torch.from_numpy(data["mu"]).to(device)        # (C_IN,)
        P = torch.from_numpy(data["P"]).to(device)          # (C_REDUCED, C_IN)
        return mu, P

    def apply_pca(self, feats, mu, P):
        """
        feats: (..., C_IN)
        mu:   (C_IN,)
        P:    (C_REDUCED, C_IN)
        Return:
            feats_reduced: (..., C_REDUCED)
        """
        orig_shape = feats.shape[:-1]
        C = feats.shape[-1]
        assert C == mu.numel(), f"Expected feature dim {mu.numel()}, got {C}"

        x = feats.reshape(-1, C)           # (N_all, C_IN)
        x = x - mu                         
        # (N_all, C_IN) @ (C_IN, C_REDUCED)^T = (N_all, C_REDUCED)
        # P shape: (C_REDUCED, C_IN)
        x_reduced = x @ P.t()
        x_reduced = x_reduced.reshape(*orig_shape, P.size(0))  # (..., C_REDUCED)
        return x_reduced


    # def get_cam_feats(self, imgs, img_metas):
    #     """Get feature maps from images."""
    #     batch_size, num_sweeps, num_cams, num_channels, imH, imW = imgs.shape  # [B, N, V, C, H, W]

    #     # imgs = imgs.flatten().view(batch_size * num_sweeps * num_cams,
    #     #                            num_channels, imH, imW)
    #     # img_feats = self.img_neck(self.img_backbone(imgs))[0]  # PCA 해서 가져와야함 
    #     dino_output = self.dino(imgs, img_metas)
    #     img_feats = dino_output['last_tokens'] # [B, V, N, C]
    #     patch_hw = dino_output['patch_hw'] # (H2/14, W2/14)  (35, 58)
        
    #     # img_feats = img_feats.reshape(batch_size*num_sweeps, num_cams, img_feats.shape[-1], patch_hw[0], patch_hw[1])  # [B*N, V, C, H2/14, W2/14]

    #     return img_feats, patch_hw  # [B, V, N, C]  N=35*58



    def _forward_single_sweep(self,
                              sweep_index,
                              sweep_imgs,
                              lidar_depth,
                              mats_dict,
                              img_metas,
                              is_return_depth=False,
                              only_bev=True,
                              dino_out=None):
        """Forward function for single sweep.

        Args:
            sweep_index (int): Index of sweeps.
            sweep_imgs (Tensor): Input images.
            mats_dict (dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            is_return_depth (bool, optional): Whether to return depth.
                Default: False.

        Returns:
            Tensor: BEV feature map.
        """
        batch_size, num_sweeps, num_cams, num_channels, img_height, img_width = sweep_imgs.shape
        # img_feats, patch_hw = self.get_cam_feats(sweep_imgs, img_metas)   # [B, V, N, C]
        img_feats = dino_out['last_tokens'] # [B, V, N, C]
        patch_hw = dino_out['patch_hw'] # (H2/14, W2/14)  (35, 58)

        # get PCA 
        mu, P = self.load_pca("BEVDepth/pca_ckpts/pca_sckit_768_to_256.npz", device='cuda')
        source_features = self.apply_pca(img_feats, mu, P)  # (B, V, N, 128) N=35*58
        source_features = source_features.reshape(batch_size*num_cams, source_features.shape[-1], patch_hw[0], patch_hw[1])  # [B*V, C, H2/14, W2/14] (6, 128, 35, 58)
        
        # depth = depth_feature[:, :self.depth_channels].softmax(   # [6, 112, 16, 44] -> [B*V, D, H, W]
        #     dim=1, dtype=depth_feature.dtype)
        
        depth = lidar_depth # [B, V, H, W]
        
        if self.use_soft_depth:
            depth = self.get_lidar_depth_gaussian(lidar_depth)  # (B*V, D, H, W)
        else:
            depth = self.get_lidar_depth_one_hot(lidar_depth)   # (B*V, D, H, W)
        
        geom_xyz = self.get_geometry(
            mats_dict['sensor2ego_mats'][:, sweep_index, ...],
            mats_dict['intrin_mats'][:, sweep_index, ...],
            mats_dict['ida_mats'][:, sweep_index, ...],
            mats_dict.get('bda_mat', None),
        )
        geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) /
                    self.voxel_size).int()
        
        if self.training or self.use_da and not only_bev:
            img_feat_with_depth = depth.unsqueeze(                     
                1) * source_features.unsqueeze(2)    # output_channels = 80, depth_channels = 112

            img_feat_with_depth = self._forward_voxel_net(img_feat_with_depth)  # [6, 80, 112, 16, 44]

            img_feat_with_depth = img_feat_with_depth.reshape(   # [1, 6, 80, 112, 16, 44]
                batch_size,
                num_cams,
                img_feat_with_depth.shape[1],
                img_feat_with_depth.shape[2],
                img_feat_with_depth.shape[3],
                img_feat_with_depth.shape[4],
            )

            img_feat_with_depth = img_feat_with_depth.permute(0, 1, 3, 4, 5, 2)    # [1, 6, 112, 16, 44, 80]

            feature_map = voxel_pooling_train(geom_xyz,              # feature_map = [1, 80, 128, 128]
                                              img_feat_with_depth.contiguous(),
                                              self.voxel_num.cuda())
        elif only_bev:
            feature_map = voxel_pooling_inference(
                geom_xyz, 
                depth, 
                source_features.contiguous(),
                self.voxel_num.cuda())
        
        # if is_return_depth:
        #     # final_depth has to be fp32, otherwise the depth
        #     # loss will colapse during the traing process.
        #     return feature_map.contiguous(
        #     ), depth_feature[:, :self.depth_channels].softmax(dim=1)
        return feature_map.contiguous()


    def forward(self,
                sweep_imgs,
                lidar_depth, 
                mats_dict,
                img_metas,
                timestamps=None,
                is_return_depth=False,
                only_bev=True,
                dino_out=None):
        """Forward function.

        Args:
            sweep_imgs(Tensor): Input images with shape of (B, num_sweeps,
                num_cameras, 3, H, W).
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            timestamps(Tensor): Timestamp for all images with the shape of(B,
                num_sweeps, num_cameras).

        Return:
            Tensor: bev feature map.
        """
        batch_size, num_sweeps, num_cams, num_channels, img_height, img_width = sweep_imgs.shape
        
        key_frame_res = self._forward_single_sweep(
            0,
            sweep_imgs[:, 0:1, ...],
            lidar_depth,    
            mats_dict,
            img_metas,
            is_return_depth=is_return_depth,
            only_bev=only_bev, dino_out=dino_out)
        if num_sweeps == 1:
            return key_frame_res

        key_frame_feature = key_frame_res[
            0] if is_return_depth else key_frame_res

        ret_feature_list = [key_frame_feature]
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                feature_map = self._forward_single_sweep(
                    sweep_index,
                    sweep_imgs[:, sweep_index:sweep_index + 1, ...],
                    lidar_depth,
                    mats_dict,
                    img_metas,
                    is_return_depth=is_return_depth, 
                    only_bev=only_bev, dino_out=dino_out)
                ret_feature_list.append(feature_map)

        if is_return_depth:
            return torch.cat(ret_feature_list, 1), key_frame_res[1]
        else:
            return torch.cat(ret_feature_list, 1)



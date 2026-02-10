import torch
import torch.nn as nn
import torch.nn.functional as F
from bevdepth.ops.voxel_pooling_train import voxel_pooling_train


class SegEmbedEncoder(nn.Module):
    """
    Encodes segmentation class-id map to single-scale feature map.
    Input: [B, V, H, W] segmentation ids
    Output: [B*V, C, H, W] feature map
    """
    def __init__(self, num_classes, embed_dim, out_channels):
        super().__init__()
        self.embed = nn.Embedding(num_classes+1, embed_dim)
        
        # Projection: [B*V, embed_dim, H, W] -> [B*V, out_channels, H, W]
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels, 3, padding=1),
            nn.SiLU(),
        )

    def forward(self, seg_id):
        """
        Args:
            seg_id: [B, V, H, W] (class indices)
        Returns:
            seg_emb: [B*V, C_emb, H, W]
        """
        B, V, H, W = seg_id.shape
        
        # Flatten batch and view for conv processing
        seg_id = seg_id.view(B * V, H, W)  # [B*V, H, W]
        
        # -1 → 0 (unknown), 1-16 → 1-16 
        seg_id = seg_id.clamp(min=-1, max=16)  
        seg_id = seg_id + 1  # [-1, 16] → [0, 17]
        seg_id = seg_id.clamp(min=0, max=16)   # class label range: [0, 16]
        
        seg_emb = self.embed(seg_id)  # [B*V, H, W, embed_dim]
        seg_emb = seg_emb.permute(0, 3, 1, 2).contiguous()
        seg_emb = self.proj(seg_emb)
        
        return seg_emb


class SegBEVEncoder(nn.Module):
    """
    Multi-scale encoder for seg_bev after voxel pooling.
    Takes single seg_bev [B, C, bevH, bevW] and outputs multi-scale features.
    """
    def __init__(self, in_channels, channel_mult=(2, 4, 8)):
        super().__init__()
        
        self.scale1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * channel_mult[0], 3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.scale2 = nn.Sequential(
            nn.Conv2d(in_channels * channel_mult[0], in_channels * channel_mult[1], 3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.scale4 = nn.Sequential(
            nn.Conv2d(in_channels * channel_mult[1], in_channels * channel_mult[2], 3, stride=2, padding=1),
            nn.SiLU(),
        )

    def forward(self, seg_bev):
        """
        Args:
            seg_bev: [B, C, bevH, bevW]
        Returns:
            dict with keys {1, 2, 4}, each value is [B, C', H', W']
        """
        s1 = self.scale1(seg_bev)  # [B, C, H, W]
        s2 = self.scale2(s1)       # [B, C*2, H//2, W//2]
        s4 = self.scale4(s2)       # [B, C*4, H//4, W//4]
        
        return {
            1: s1,  # [B, C, bevH, bevW]
            2: s2,  # [B, C*2, bevH//2, bevW//2]
            4: s4,  # [B, C*4, bevH//4, bevW//4]
        }


class SegBEVLSS(nn.Module):
    """
    BEVDepth-style lift-splat for multi-view segmentation ID map -> BEV class distribution.
    Uses voxel_pooling_inference(geom_xyz, depth_features, context_features, voxel_num).
    """

    def __init__(
        self,
        x_bound, 
        y_bound,
        z_bound,
        d_bound,
        final_dim,
        downsample_factor=14,
        num_classes=16,
        embed_dim=32,
        emb_channels=64,
        depth_sigma=2.0,
        eps=1e-6,
    ):
        super().__init__()
        self.final_dim = final_dim
        self.downsample_factor = int(downsample_factor)
        self.d_bound = d_bound
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)
        self.emb_channels = int(emb_channels)
        self.depth_sigma = float(depth_sigma)
        self.eps = float(eps)
        
        # Segmentation embedding encoder (single output)
        self.seg_encoder = SegEmbedEncoder(
            num_classes=num_classes,
            embed_dim=embed_dim,
            out_channels=emb_channels,
        )
        
        # Multi-scale BEV encoder (after voxel pooling)
        self.seg_bev_encoder = SegBEVEncoder(
            in_channels=emb_channels,
            channel_mult=(2, 4, 8),
        )

        self.register_buffer(
            "voxel_size",
            torch.Tensor([row[2] for row in [x_bound, y_bound, z_bound]])
        )
        self.register_buffer(
            "voxel_coord",
            torch.Tensor([row[0] + row[2] / 2.0 for row in [x_bound, y_bound, z_bound]])
        )
        self.register_buffer(
            "voxel_num",
            torch.LongTensor([(row[1] - row[0]) / row[2] for row in [x_bound, y_bound, z_bound]])
        )

        self.register_buffer("frustum", self.create_frustum())
        self.depth_channels, _, _, _ = self.frustum.shape  # D


    # ---------- same frustum / geometry as BEVDepth ----------
    def create_frustum(self):
        ogH, ogW = self.final_dim
        ds = self.downsample_factor
        fH, fW = ogH // ds, ogW // ds

        d_coords = torch.arange(*self.d_bound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = d_coords.shape

        x_coords = torch.linspace(0, ogW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        y_coords = torch.linspace(0, ogH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        paddings = torch.ones_like(d_coords)

        return torch.stack((x_coords, y_coords, d_coords, paddings), -1)  # [D,fH,fW,4]

    def get_geometry(self, sensor2ego_mat, intrin_mat, ida_mat, bda_mat):
        B, V, _, _ = sensor2ego_mat.shape

        points = self.frustum  # [D,fH,fW,4]
        ida_mat = ida_mat.view(B, V, 1, 1, 1, 4, 4)
        points = ida_mat.inverse().matmul(points.unsqueeze(-1)) 

        points = torch.cat(
            (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
             points[:, :, :, :, :, 2:]),
            5
        )

        combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat))
        points = combine.view(B, V, 1, 1, 1, 4, 4).matmul(points)

        if bda_mat is not None:
            bda_mat = bda_mat.unsqueeze(1).repeat(1, V, 1, 1).view(B, V, 1, 1, 1, 4, 4)
            points = (bda_mat @ points).squeeze(-1)
        else:
            points = points.squeeze(-1)

        return points[..., :3]  # [B,V,D,fH,fW,3]


    def downsample_depth(self, depth):
        assert depth.dim() == 4, f"Expected [B,V,H,W], got {depth.shape}"
        B, V, H, W = depth.shape
        ds = self.downsample_factor
        assert H % ds == 0 and W % ds == 0, f"H,W must be divisible by ds={ds}, got {(H,W)}"

        # NaN/inf 방어 (DepthAnything에서 종종 발생)
        d_min, d_max, _ = self.d_bound
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        # [B,V,H,W] -> [B*V,H,W]
        depth = depth.view(B * V, H, W)

        # [B*V,H,W] -> [B*V,h,ds,w,ds,1]
        h, w = H // ds, W // ds
        depth = depth.view(B * V, h, ds, w, ds, 1)

        # [B*V,h,ds,w,ds,1] -> [B*V,h,w,1,ds,ds]
        depth = depth.permute(0, 1, 3, 5, 2, 4).contiguous()
        # [B*V*h*w, ds*ds]
        depth = depth.view(-1, ds * ds)

        # valid: depth > 0 
        valid = depth > 0.0

        depth_tmp = torch.where(valid, depth, depth.new_full(depth.shape, 1e5))
        depth_min = depth_tmp.min(dim=-1).values  # [B*V*h*w]

        depth_min = depth_min.view(B * V, h, w)   # [B*V,h,w]
        valid_mask = depth_min < 1e5              # patch에 valid가 하나라도 있었으면 True
        return depth_min, valid_mask, h, w

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, V, H, W] (LiDAR)
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.contiguous().view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample_factor * self.downsample_factor)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample_factor,
                                   W // self.downsample_factor)

        gt_depths = (gt_depths -
                     (self.d_bound[0] - self.d_bound[2])) / self.d_bound[2] 
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels + 1)
        gt_depths = gt_depths[..., 1:]
        gt_depths = gt_depths.permute(0, 3, 1, 2).contiguous().float()
        return gt_depths
    
    def depth_to_bev_bin(self, depth, d_bound, sigma=1.0, eps=1e-8):
        """
        depth: (..., H, W)  or (..., ) whatever, but we'll assume depth is (..., h, w)
        return: (..., h, w, D)
        """
        d_min, d_max, d_step = d_bound
        D = int((d_max - d_min) / d_step)
        valid = torch.isfinite(depth) & (depth >= d_min) & (depth < d_max)
        
        depth = depth.clamp(d_min, d_max - 1e-6)
        depth_idx = (depth - d_min) / d_step  # same shape as depth

        bins = torch.arange(D, device=depth.device, dtype=depth_idx.dtype)
        bins = bins.view(*([1] * depth.ndim), D)  # (1,...,1,D) with correct rank

        diff = bins - depth_idx.unsqueeze(-1)     # (..., h, w, D)
        prob = torch.exp(-(diff ** 2) / (2 * sigma ** 2))
        prob = prob * valid.unsqueeze(-1).to(prob.dtype)
        prob = prob / (prob.sum(dim=-1, keepdim=True) + eps)
        return prob

    def depth_to_onehot(
        self,
        depth_min,        # [B*V, h, w]  (meter)
        valid_mask,       # [B*V, h, w]  (bool)
        eps=1e-6,
    ):
        """
        Return:
            depth_prob: [B*V, D, h, w]  one-hot (Dirac) distribution
        """
        d_min, d_max, d_step = self.d_bound
        D = self.depth_channels  # must match frustum D

        depth = depth_min.to(torch.float32)

        # in-range validity
        in_range = (depth >= d_min) & (depth < d_max) & torch.isfinite(depth)
        valid = valid_mask & in_range

        # compute bin centers convention
        idx = torch.round((depth - d_min) / d_step).long()
        # clamp and mask invalid
        idx = idx.clamp(0, D - 1)
        idx = torch.where(valid, idx, idx.new_zeros(idx.shape))  # dummy index for invalid

        # one-hot: [B*V,h,w,D] then permute to [B*V,D,h,w]
        one_hot = F.one_hot(idx, num_classes=D).to(depth.dtype)  # [B*V,h,w,D]
        depth_prob = one_hot.permute(0, 3, 1, 2).contiguous()    # [B*V,D,h,w]

        # invalid -> all zeros
        depth_prob = depth_prob * valid.unsqueeze(1).to(depth_prob.dtype)
        return depth_prob


    def forward(self, seg_id, depth, mats_dict, sweep_index=0):
        """
        Args:
            seg_id: [B, V, H, W] - multi-view segmentation class-id map
            depth:  [B, V, H, W] - LiDAR depth map
            mats_dict: camera matrices
            sweep_index: sweep index for temporal data
        Returns:
            dict with keys {1, 2, 4}:
                1: [B, C, bevH, bevW]
                2: [B, C*2, bevH//2, bevW//2]
                4: [B, C*4, bevH//4, bevW//4]
        """
        assert seg_id.dim() == 4, f'seg_id is {seg_id.shape}'

        B, V, H, W = seg_id.shape

        fH = H // self.downsample_factor
        fW = W // self.downsample_factor

        # Get depth probability: [B*V, D, fH, fW]
        depth_prob = self.get_downsampled_gt_depth(depth)
        if depth_prob.dim() == 4 and depth_prob.shape[-1] == self.depth_channels:
            depth_prob = depth_prob.permute(0, 3, 1, 2).contiguous()

        # Downsample seg_id to match depth resolution using interpolate
        seg_downsized = F.interpolate(
            seg_id.float().view(B * V, 1, H, W),
            size=(fH, fW),
            mode='nearest'
        ).squeeze(1).long()  # [B*V, fH, fW]
        seg_downsized = seg_downsized.view(B, V, fH, fW)  # [B, V, fH, fW]
        
        # Get single seg embedding: [B*V, C, fH, fW]
        seg_emb = self.seg_encoder(seg_downsized)  # [B*V, C, fH, fW]

        # geometry
        geom_xyz = self.get_geometry(
            mats_dict["sensor2ego_mats"][:, sweep_index, ...],
            mats_dict["intrin_mats"][:, sweep_index, ...],
            mats_dict["ida_mats"][:, sweep_index, ...],
            mats_dict.get("bda_mat", None),
        )
        geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) / self.voxel_size).int()
        geom_xyz = geom_xyz.contiguous()

        # Voxel pooling: [B*V, C, fH, fW] -> [B, C, bevH, bevW]
        # Outer product (same as BaseLSSFPN): [B*V, 1, D, fH, fW] * [B*V, C, 1, fH, fW] -> [B*V, C, D, fH, fW]
        D = depth_prob.shape[1]  # depth_prob: [B*V, D, fH, fW]
        C = seg_emb.shape[1]     # seg_emb: [B*V, C, fH, fW]
        img_feat_with_depth = depth_prob.unsqueeze(1) * seg_emb.unsqueeze(2)  # [B*V, C, D, fH, fW]
        
        # Reshape to [B, V, D, fH, fW, C] for voxel_pooling_train
        img_feat_with_depth = img_feat_with_depth.view(B, V, C, D, fH, fW)
        img_feat_with_depth = img_feat_with_depth.permute(0, 1, 3, 4, 5, 2).contiguous()  # [B, V, D, fH, fW, C]
        
        seg_bev = voxel_pooling_train(
            geom_xyz.contiguous(),
            img_feat_with_depth.contiguous(),
            self.voxel_num.cuda(),
        )  # [B, C, bevH, bevW]

        # Multi-scale encoding from single seg_bev
        seg_bev_dict = self.seg_bev_encoder(seg_bev)
        
        return seg_bev_dict  # {1: [B,C,H,W], 2: [B,C*2,H//2,W//2], 4: [B,C*4,H//4,W//4]}
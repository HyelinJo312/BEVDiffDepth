import torch
import torch.nn as nn
import torch.nn.functional as F
from bevdepth.ops.voxel_pooling_inference import voxel_pooling_inference


class SegBEVLSS(nn.Module):
    """
    BEVDepth-style lift-splat for multi-view segmentation ID map -> BEV class distribution.
    Uses voxel_pooling_inference(geom_xyz, depth_features, context_features, voxel_num).
    """

    def __init__(
        self,
        x_bound=[-51.2, 51.2, 0.8], 
        y_bound=[-51.2, 51.2, 0.8],
        z_bound=[-5.0, 3.0, 0.8],
        d_bound=[2.0, 58.0, 0.5],
        final_dim=(252, 700),
        downsample_factor=14,
        num_classes=17,
        depth_sigma=2.0,
        eps=1e-6,
        out_bev_size=64,              # <-- diffusion condition base resolution
        w_min=1e-3,                   # <-- valid threshold in BEV
        resize_mode="area",
    ):
        super().__init__()
        self.final_dim = final_dim
        self.downsample_factor = int(downsample_factor)
        self.d_bound = d_bound
        self.num_classes = int(num_classes)
        self.depth_sigma = float(depth_sigma)
        self.eps = float(eps)
        self.out_bev_size = int(out_bev_size)
        self.w_min = float(w_min)
        self.resize_mode = str(resize_mode)

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
        gt_depths = gt_depths.permute(0, 3, 1, 2).contiguous()
        return gt_depths.float()
    
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

    # ---------- seg: downsample + onehot ----------
    def seg2onehot(self, seg_id, fH, fW):
        """
        seg_id: [B,V,H,W] int16, values in {-1, 1..16} (0은 없음)
        return: onehot [B*V, C, fH, fW] where
            - valid class(1..16) -> one-hot
            - unknown(-1) or 0 -> all-zero (no contribution)
        """
        B, V, H, W = seg_id.shape
        C = self.num_classes  # 17 # TODO: 17 -> 16으로 수정 

        # resize id-map (nearest)
        seg = seg_id.view(B * V, 1, H, W).float()
        seg = F.interpolate(seg, size=(fH, fW), mode="nearest").squeeze(1).long()

        # valid: 1..16만 사용 (0과 -1은 unknown 취급)
        valid = (seg >= 1) & (seg < C)  # TODO: valid = (seg >= 1) & (seg <= C) 

        # one_hot은 음수 못 받으니, 임시로 0으로 채운 뒤 valid로 마스킹
        seg_valid = torch.where(valid, seg, torch.zeros_like(seg))
        onehot = F.one_hot(seg_valid, num_classes=C).permute(0, 3, 1, 2).float()

        # invalid는 all-zero로
        onehot = onehot * valid.unsqueeze(1)
        return onehot.contiguous()
    
    def _downsize_bev(self, x):
        # x: [B,C,H,W]
        H, W = x.shape[-2:]
        target = self.out_bev_size
        if H == target and W == target:
            return x
        if (H % target == 0) and (W % target == 0):
            kH, kW = H // target, W // target
            return F.avg_pool2d(x, kernel_size=(kH, kW), stride=(kH, kW))
        return F.interpolate(x, size=(target, target), mode="area")
    
    def rescale_weight(self, w_bev):
        """
        w_bev:    [B, 1, H, W] (Raw Density Weight)
        """
        threshold = 0.05 # 0.08
        sharpness = 15  # 18
        
        # log1p(x) = log(1 + x)
        w_map = torch.log1p(w_bev)

        # 2. Min-Max Scaling
        w_min = w_map.min()
        w_max = w_map.max()
        w_norm = (w_map - w_min) / (w_max - w_min + self.eps)

        # 3. Soft-thresholding
        w_mask = torch.sigmoid(sharpness * (w_norm - threshold))
        # normalize to 0~1
        w_mask = (w_mask - w_mask.min()) / (w_mask.max() - w_mask.min() + self.eps)
        return w_norm


    def forward(self, seg_id, depth, mats_dict, sweep_index=0):
        """
        seg_id: [B,V,H,W]
        depth:  [B,V,H,W]
        """
        assert seg_id.dim() == 4, f'seg_id is {seg_id.shape}'

        B, V, H, W = seg_id.shape
        device = seg_id.device

        fH = H // self.downsample_factor
        fW = W // self.downsample_factor

        depth_prob = self.get_downsampled_gt_depth(depth)  # [B*V, D, h, w]

        # # depth prob (da3)
        # depth_down, valid_mask, h, w = self.downsample_depth(depth)  # [B*V,fH,fW]
        # depth_prob = self.depth_to_bev_bin(depth_down, self.d_bound, sigma=2)
        # depth_prob = depth_prob * valid_mask.unsqueeze(-1)  # invalid patch -> all zeros
        
        # down_depth, valid_mask, h, w = self.downsample_depth(depth)   # [B*V,h,w]
        # depth_prob = self.depth_to_onehot(down_depth, valid_mask)  # [B*V, D, h, w]

        # If depth_to_bev_bin returns [B*V,fH,fW,D], convert -> [B*V,D,fH,fW]
        if depth_prob.dim() == 4 and depth_prob.shape[-1] == self.depth_channels:
            depth_prob = depth_prob.permute(0, 3, 1, 2).contiguous()
            
        # depth_prob = depth_prob.view(B, V, self.depth_channels, fH, fW).contiguous()
        
        # seg onehot
        seg_onehot = self.seg2onehot(seg_id, fH, fW)  # [B*V,C,fH,fW]

        # geometry
        geom_xyz = self.get_geometry(
            mats_dict["sensor2ego_mats"][:, sweep_index, ...],
            mats_dict["intrin_mats"][:, sweep_index, ...],
            mats_dict["ida_mats"][:, sweep_index, ...],
            mats_dict.get("bda_mat", None),
        )
        geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) / self.voxel_size).int() # type: ignore
        geom_xyz = geom_xyz.contiguous()

        # pool seg
        seg_bev = voxel_pooling_inference(
            geom_xyz,
            depth_prob.contiguous(),          # [B*V,D,fH,fW]
            seg_onehot.contiguous(),          # [B*V,C,fH,fW]
            self.voxel_num.cuda(),
        )  # [B,C,bevH,bevW]

        # pool weight
        ones = torch.ones((B*V, 1, fH, fW), device=device, dtype=seg_bev.dtype)
        w_bev = voxel_pooling_inference(
            geom_xyz,
            depth_prob.contiguous(),
            ones.contiguous(),
            self.voxel_num.cuda(),
        )  # [B,1,bevH,bevW]

        seg_bev_resized = self._downsize_bev(seg_bev)
        w_bev_resized = self._downsize_bev(w_bev)
        valid = (w_bev_resized > self.w_min)
        
        seg_bev_norm = torch.where(valid, seg_bev_resized / (w_bev_resized + self.eps), torch.zeros_like(seg_bev_resized))

        # optional: class-wise renorm (-> class prob)
        seg_sum = seg_bev_norm.sum(dim=1, keepdim=True)
        seg_prob_64 = torch.where(valid & (seg_sum > 0), seg_bev_norm / (seg_sum + self.eps), seg_bev_norm)
        
        w_bev_scaled = self.rescale_weight(w_bev_resized)

        return seg_prob_64, w_bev_scaled
    
        # # 1) normalization 
        # seg_bev = seg_bev / (w_bev + self.eps)  # [B,C,bevH,bevW]

        # # 2) resize
        # seg_bev_down = F.interpolate(seg_bev, size=(64, 64), mode="bilinear", align_corners=False)
        # w_bev_down   = F.interpolate(w_bev, size=(64, 64), mode="bilinear", align_corners=False)

        # 3) renormalization
        # seg_bev_down = seg_bev_down / (seg_bev_down.sum(dim=1, keepdim=True) + self.eps)
        # seg_sum = seg_bev_down.sum(dim=1, keepdim=True)  # [B,1,H,W]
        # valid = (w_bev_down > 0.0001) & (seg_sum > 0)
        # seg_bev_mask = torch.where(valid, seg_bev_down / (seg_sum + self.eps), seg_bev_down)
        # # 4) Gating with weight
        # gate = (w_bev_down / (w_bev_down + k)).clamp(0, 1)   # k는 스케일용, 예: 1.0
        # seg_for_spade = seg_bev_down * gate                  # 또는 seg_for_spade = seg_bev_down; gate는 SPADE 내부에서 사용
        
        # return seg_bev_mask, w_bev_down
        
        

# ==================== seg pyramid utils ====================


def _downsample_prob(x: torch.Tensor, target_hw: int) -> torch.Tensor:
    """
    x: [B,C,H,W] soft map
    Downsample with avg_pool2d if integer factor else area.
    """
    H, W = x.shape[-2:]
    if (H, W) == (target_hw, target_hw):
        return x
    if H % target_hw == 0 and W % target_hw == 0:
        kH, kW = H // target_hw, W // target_hw
        return F.avg_pool2d(x, kernel_size=(kH, kW), stride=(kH, kW))
    return F.interpolate(x, size=(target_hw, target_hw), mode="area")

def _upsample_prob(x: torch.Tensor, target_hw: int) -> torch.Tensor:
    """
    x: [B,C,H,W] soft map
    Upsample with bilinear (smooth) for SPADE conditioning.
    """
    H, W = x.shape[-2:]
    if (H, W) == (target_hw, target_hw):
        return x
    return F.interpolate(x, size=(target_hw, target_hw), mode="bilinear", align_corners=False)

@torch.no_grad()
def get_seg_pyramid(seg_prob_64, w_64=None, w_min=1e-3, eps=1e-6, renorm=True):
    """
    Returns dict:
      segs['64'], segs['32'], segs['16'], segs['up32'], segs['up64']
    where 'up32' is 16->32 upsampled seg, 'up64' is 16->64 upsampled seg.
    If w_64 is given, we down/up it similarly and mask invalid regions.
    """
    seg64 = seg_prob_64.float()

    # down path segs
    seg32 = _downsample_prob(seg64, 32)
    seg16 = _downsample_prob(seg64, 16)

    # optional weight handling (highly recommended)
    if w_64 is not None:
        w64 = w_64.float()
        w32 = _downsample_prob(w64, 32)  # avg/area 둘 다 OK for weight
        w16 = _downsample_prob(w64, 16)

        def _mask_and_renorm(seg, w):
            valid = (w > w_min)
            if renorm:
                s = seg.sum(dim=1, keepdim=True)
                seg = torch.where(valid & (s > 0), seg / (s + eps), torch.zeros_like(seg))
            else:
                seg = torch.where(valid, seg, torch.zeros_like(seg))
            return seg, w

        seg64, w64 = _mask_and_renorm(seg64, w64)
        seg32, w32 = _mask_and_renorm(seg32, w32)
        seg16, w16 = _mask_and_renorm(seg16, w16)
    else:
        w64 = w32 = w16 = None

    # up path segs (use smooth bilinear upsample from the coarsest or from matching scale)
    up32 = _upsample_prob(seg16, 32)
    up64 = _upsample_prob(seg16, 64)

    out = {
        "1": seg64,
        "2": seg32,
        "4": seg16,
        "up32": up32,
        "up64": up64,
    }
    if w_64 is not None:
        out.update({"w1": w64, "w2": w32, "w4": w16})
    return out


@torch.no_grad()
def make_seg_pyramid_by_ds(
    seg_prob_base,   # [B,C,base,base] e.g. base=64
    w_base,   # [B,1,base,base]
    base_size,
    ds_list=(1, 2, 4),             # ds=1->64, ds=2->32, ds=4->16
    w_min=1e-3,
    eps=1e-6,
    renorm=True,
    include_weight=False,  # True면 [B,C+1,H,W]로 concat해서 반환
):
    """
    Returns:
      seg_by_ds: dict[int, Tensor]
        key: ds (1,2,4,...)
        value:
          - if include_weight=False: seg [B,C,H,W]
          - if include_weight=True : concat([seg,w]) [B,C+1,H,W]
    """
    assert seg_prob_base.dim() == 4, "seg_prob_base must be [B,C,H,W]"
    assert seg_prob_base.shape[-2:] == (base_size, base_size), f"expected base {base_size}x{base_size}"

    if w_base is not None:
        assert w_base.dim() == 4 and w_base.shape[1] == 1
        assert w_base.shape[-2:] == (base_size, base_size)

    seg_base = seg_prob_base.detach().float()
    w_base_f = w_base.detach().float() if w_base is not None else None

    seg_by_ds = {}

    for ds in ds_list:
        target_hw = int(base_size // ds)
        seg = _downsample_prob(seg_base, target_hw)

        if w_base_f is not None:
            w = _downsample_prob(w_base_f, target_hw)
            valid = (w > w_min)

            if renorm:
                seg_sum = seg.sum(dim=1, keepdim=True)  # [B,1,H,W]
                seg = torch.where(valid & (seg_sum > 0), seg / (seg_sum + eps), torch.zeros_like(seg))
            else:
                seg = torch.where(valid, seg, torch.zeros_like(seg))

            if include_weight:
                seg = torch.cat([seg, w], dim=1)  # [B,C+1,H,W]
        else:
            if renorm:
                seg_sum = seg.sum(dim=1, keepdim=True)
                seg = torch.where(seg_sum > 0, seg / (seg_sum + eps), torch.zeros_like(seg))

        seg_by_ds[int(ds)] = seg

    return seg_by_ds

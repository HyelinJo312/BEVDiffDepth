import torch.nn as nn
import torch
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig
# from omegaconf import OmegaConf
import torch.nn.functional as F
from einops import repeat
import os
import torchvision.transforms as T
from collections import OrderedDict
# import lightning as L
from typing import Optional
import math
from transformers import AutoImageProcessor, AutoModel, DPTForDepthEstimation
from transformers import CLIPProcessor, CLIPVisionModel
from torchvision.utils import save_image
import numpy as np
import cv2

NUM_DECONV = 3
NUM_FILTERS = [32, 32, 32]
DECONV_KERNELS = [2, 2, 2]
VIT_MODEL = 'google/vit-base-patch16-224'



def pad_to_make_square(x):
    y = 255*((x+1)/2)
    y = torch.permute(y, (0,2,3,1))
    bs, _, h, w = x.shape
    if w>h:
        patch = torch.zeros(bs, w-h, w, 3).to(x.device)
        y = torch.cat([y, patch], axis=1)
    else:
        patch = torch.zeros(bs, h, h-w, 3).to(x.device)
        y = torch.cat([y, patch], axis=2)
    return y.to(torch.int)


class EmbeddingAdapter(nn.Module):
    def __init__(self, emb_dim=768):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, texts, gamma):
        emb_transformed = self.fc(texts)
        texts = texts + gamma * emb_transformed
        texts = repeat(texts, 'n c -> n b c', b=1)
        return texts

    
class GetDINOv2Cond(nn.Module):
    def __init__(
        self,
        encoder: str = 'vitb',     # ['vits', 'vitb', 'vitl'] → small/base/large
        features: int = 256,       # output dim for BEVDiffuser conditioning
        device: str = 'cuda',
        patch: int = 14,
        input_size: int = 518,     
        symmetric_pad: bool = True,
    ):
        super().__init__()
        assert encoder in ['vits', 'vitb', 'vitl']
        self.device = device
        self.input_min = input_size
        self.patch = patch
        self.symmetric_pad = symmetric_pad

        # self.model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(self.device)
        self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)
        self.model.requires_grad_(False)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

        self.hidden_dim = self.model.config.hidden_size  # 384/768/1024

    def image_preprocess(self, x):
        """
        x: (N, 3, H, W) float tensor in [0,1] or [0,255]
        -> (x_norm: (N, 3, H2, W2) on self.device, ImageNet normalized,
            Hp, Wp: patch grid size = (H2//patch, W2//patch))
        """
        assert x.ndim == 4 and x.shape[1] == 3, f"Expected (N,3,H,W), got {x.shape}"
        x = x.to(dtype=torch.float32)
        # if x.max() > 1.5:      # likely [0,255]
        #     x = x / 255.0
        # x = x.clamp_(0.0, 1.0)

        N, C, H, W = x.shape
        # keep aspect ration: min(H1, W1) >= input_min
        scale = max(self.input_min / H, self.input_min / W)
        H1, W1 = int(round(H * scale)), int(round(W * scale))

        x = F.interpolate(x, size=(H1, W1), mode='bicubic', align_corners=False)
        
        H2 = (H1 + self.patch - 1) // self.patch * self.patch
        W2 = (W1 + self.patch - 1) // self.patch * self.patch
        pad_h, pad_w = H2 - H1, W2 - W1

        if self.symmetric_pad:
            top = pad_h // 2; bottom = pad_h - top
            left = pad_w // 2; right = pad_w - left
        else:
            top = 0; bottom = pad_h; left = 0; right = pad_w

        x = F.pad(x, (left, right, top, bottom), mode='replicate')
        x = x.to(self.device, non_blocking=True)
        
        extra_geom = {
            'scale': scale,
            'H2W2': (H2, W2),
            'padding': (top, left),
            'patch_size': self.patch,
        }

        Hp, Wp = H2 // self.patch, W2 // self.patch
        return x, Hp, Wp, extra_geom
    
    def forward(self, images, img_metas, n_layers=4):
        """
        images: (B, 6, C, H, W) or (6, C, H, W) when bs=1
        """
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a torch.Tensor")

        if images.ndim == 5:
            B, V, C, H, W = images.shape
            x = images.reshape(B * V, C, H, W).contiguous()
        elif images.ndim == 4:
            V, C, H, W = images.shape
            B  = 1
            x = images.reshape(B * V, C, H, W).contiguous()
        else:
            raise ValueError(f"Unexpected tensor shape: {images.shape}")
        
        x, Hp, Wp, extra_geom = self.image_preprocess(x)

        # Dinov2 forward
        with torch.no_grad():
            outputs = self.model(pixel_values=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        hs_selected = hidden_states[-n_layers:] if n_layers > 0 else [hidden_states[-1]]
        
        feats_out = []
        cls_out = []
        for h in hs_selected:
            cls_tok = h[:, 0]          # (B*V, C_dino)
            tok    = h[:, 1:]          # (B*V, Hp*Wp, C_dino)
            # tok = h[:, 1:, :] 
            cls_tok = cls_tok.view(B, V, self.hidden_dim)                  # (B,V,C)
            tok_seq = tok.view(B, V, Hp * Wp, self.hidden_dim)             # (B,V,N,C)
            feats_out.append(tok_seq)
            cls_out.append(cls_tok)

        last_tok, last_cls = feats_out[-1], cls_out[-1]

        return {
            'feature_type': 'dinov2',
            'features': feats_out,          # list[(B,V,N,C),(B,V,C)]
            'patch_hw': (Hp, Wp),
            'last_cls': last_cls,           # (B, V, C)
            'last_tokens': last_tok,        # (B, V, N, C)
            'img_metas': img_metas,
            'geom': extra_geom
        }


class GetDINOV2Feat(nn.Module):
    def __init__(
        self,
        encoder: str = 'vitb',     # ['vits', 'vitb', 'vitl']
        device: str = 'cuda',
        patch: int = 14,
        symmetric_pad: bool = True,
    ):
        super().__init__()
        assert encoder in ['vits', 'vitb', 'vitl']
        self.device = device
        self.patch = patch
        self.symmetric_pad = symmetric_pad

        # DINOv2 backbone
        self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)
        self.model.requires_grad_(False)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

        self.hidden_dim = self.model.config.hidden_size  # e.g. 768 (base)

    def image_preprocess(self, x):
        """
        x: (N, 3, H, W) float tensor
        Return:
          x_pad: (N, 3, H2, W2)
          Hp, Wp: H2/patch, W2/patch
          extra_geom: dict(scale, H2W2, padding, patch_size)
        """
        assert x.ndim == 4 and x.shape[1] == 3, f"Expected (N,3,H,W), got {x.shape}"
        x = x.to(dtype=torch.float32)

        N, C, H, W = x.shape
        scale = 1.0
        H1, W1 = H, W

        # patch align (ceil) → replicate padding
        H2 = (H1 + self.patch - 1) // self.patch * self.patch
        W2 = (W1 + self.patch - 1) // self.patch * self.patch
        pad_h, pad_w = H2 - H1, W2 - W1

        if self.symmetric_pad:
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
        else:
            top = 0
            bottom = pad_h
            left = 0
            right = pad_w

        # replicate padding
        x = F.pad(x, (left, right, top, bottom), mode='replicate')
        x = x.to(self.device, non_blocking=True)

        extra_geom = {
            'scale': scale,                
            'H2W2': (H2, W2),              
            'padding': (top, left),        
            'patch_size': self.patch,      # 14
        }

        Hp, Wp = H2 // self.patch, W2 // self.patch
        return x, Hp, Wp, extra_geom

    def forward(self, images, img_metas, n_layers=4):
        """
        images: (B, N, V, C, H, W) 
        """
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a torch.Tensor")
        
        B, N, V, C, H, W = images.shape
        x = images.reshape(B * N * V, C, H, W).contiguous()
    
        # aspect ratio 
        x, Hp, Wp, extra_geom = self.image_preprocess(x)  # x: (B*V, 3, H2, W2)

        # Dinov2 forward
        with torch.no_grad():
            outputs = self.model(pixel_values=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        hs_selected = hidden_states[-n_layers:] if n_layers > 0 else [hidden_states[-1]]

        feats_out = []
        cls_out = []
        for h in hs_selected:
            # h: (B*V, 1+N, C_dino)
            cls_tok = h[:, 0]          # (B*V, C_dino)
            tok    = h[:, 1:]          # (B*V, Hp*Wp, C_dino)

            cls_tok = cls_tok.view(B, V, self.hidden_dim)               # (B, V, C)
            tok_seq = tok.view(B, V, Hp * Wp, self.hidden_dim)          # (B, V, N, C)
            feats_out.append(tok_seq)
            cls_out.append(cls_tok)

        last_tok, last_cls = feats_out[-1], cls_out[-1]

        return {
            'feature_type': 'dinov2',
            # 'features': feats_out,        # list[(B,V,N,C)]
            'patch_hw': (Hp, Wp),           # (H2/14, W2/14)  (35, 58)
            'last_cls': last_cls,           # (B, V, C)
            'last_tokens': last_tok,        # (B, V, N, C) 
            'img_metas': img_metas,
            'geom': extra_geom              
        }

   

class GetDPTDepth(nn.Module):
    def __init__(
        self,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.model = DPTForDepthEstimation.from_pretrained("facebook/dpt-dinov2-small-kitti").to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/dpt-dinov2-small-kitti")

    @torch.no_grad()
    def forward(self, images: torch.Tensor, img_metas=None, save_dir=None):
        """
        images: (B, V, C, H, W) or (V, C, H, W) when B=1
        """
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a torch.Tensor")

        if images.ndim == 5:
            B, V, C, H, W = images.shape
            x = images.reshape(B * V, C, H, W)  # (B*V, C, H, W)
        elif images.ndim == 4:
            V, C, H, W = images.shape
            B = 1
            x = images.reshape(B * V, C, H, W)
        else:
            raise ValueError(f"Unexpected tensor shape: {images.shape}")

        x_cpu = x.detach().cpu()  # (B*V, C, H, W)

        inputs = self.image_processor(images=x_cpu,return_tensors="pt") # (B*V, C, H, W) tensor
        pixel_values = inputs["pixel_values"].to(self.device)  # (B*V, 3, H_d, W_d)

        # ---- DPT forward ----
        outputs = self.model(pixel_values=pixel_values)
        predicted_depth = outputs.predicted_depth  # (B*V, H_d, W_d)

        depth_resized = F.interpolate(
            predicted_depth.unsqueeze(1),  # (B*V, 1, H_d, W_d)
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)  # (B*V, H, W)
      
        depth_maps = depth_resized.view(B, V, H, W).cpu()
        
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

            for b in range(B):
                meta = img_metas[b] if img_metas is not None else None
                filenames = None
                if meta is not None and "filename" in meta:
                    filenames = meta["filename"]
                    if isinstance(filenames, (str, os.PathLike)):
                        filenames = [filenames]

                for v in range(V):
                    depth_map = depth_maps[b, v]  # (H, W)

                    if filenames is not None:
                        img_path = filenames[v]
                        base_name = os.path.basename(img_path)          # xxx.jpg
                        name_wo_ext = os.path.splitext(base_name)[0]    # xxx
                        cam_name = os.path.basename(os.path.dirname(img_path))
                        depth_name = f"{name_wo_ext}_dpt_depth"
                    else:
                        cam_name = "UNKNOWN_CAM"
                        depth_name = f"sample{b}_view{v}_dpt_depth"

                    cam_dir = os.path.join(save_dir, cam_name)
                    os.makedirs(cam_dir, exist_ok=True)

                    npy_path = os.path.join(cam_dir, depth_name + ".npz")
                    # np.save(npy_path, depth_map.numpy().astype(np.float32))
                    np.savez_compressed(npy_path, depth=depth_map.numpy().astype(np.float32))

                    color_png_path = os.path.join(cam_dir, depth_name + "_color.png")
                    save_colored_depth_cv2(depth_map, color_png_path, gamma=0.5)



class GetDPTDepthV2(nn.Module):
    '''
    Extract depth from raw image (1600x900)
    '''
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device

        # DPT-DINOv2 small (KITTI) depth model
        self.model = DPTForDepthEstimation.from_pretrained("facebook/dpt-dinov2-small-kitti").to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()

        self.image_processor = AutoImageProcessor.from_pretrained("facebook/dpt-dinov2-small-kitti")

    @torch.no_grad()
    def forward(self, images: torch.Tensor, img_metas, save_dir=None):
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a torch.Tensor")
        if img_metas is None:
            raise ValueError("img_metas is required to load raw images.")

        if images.ndim == 5:
            B, V, C, H_img, W_img = images.shape
        elif images.ndim == 4:
            V, C, H_img, W_img = images.shape
            B = 1
        else:
            raise ValueError(f"Unexpected tensor shape: {images.shape}")

        raw_images = []  # length = B * V

        for b in range(B):
            meta = img_metas[b]
            filenames = meta.get("filename", None)
            if isinstance(filenames, (str, os.PathLike)):
                filenames = [filenames]

            if filenames is None:
                raise ValueError("img_metas must contain 'filename' for raw loading.")

            assert len(filenames) == V, \
                f"Expected {V} filenames, got {len(filenames)} at batch {b}."

            for v in range(V):
                img_path = filenames[v]
                img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    raise FileNotFoundError(f"Failed to read image: {img_path}")
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                raw_images.append(img_rgb)

        inputs = self.image_processor(images=raw_images,return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)  # (B*V, 3, H_d, W_d)

        outputs = self.model(pixel_values=pixel_values)
        predicted_depth = outputs.predicted_depth  # (B*V, H_d, W_d)
        _, H_d, W_d = predicted_depth.shape

        depth_raw = predicted_depth.view(B, V, H_d, W_d)

        depth_out = []

        for b in range(B):
            meta = img_metas[b]
            ori_shapes = meta["ori_shape"]    # list of (H_ori, W_ori, 3)
            img_shapes = meta["img_shape"]    # list of (H_img_v, W_img_v, 3)

            depth_b_list = []

            for v in range(V):
                d = depth_raw[b, v]  # (H_d, W_d)

                H_ori, W_ori, _ = ori_shapes[v]       # 예: (450, 800, 3)
                H_img_v, W_img_v, _ = img_shapes[v]   # 예: (480, 800, 3)

                # 3-1) resize to ori_shape
                d_resized = F.interpolate(
                    d.unsqueeze(0).unsqueeze(0),  # (1,1,H_d,W_d)
                    size=(H_ori, W_ori),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(0).squeeze(0)  # (H_ori, W_ori)

                # 3-2) ori_shape → img_shape
                pad_bottom = H_img_v - H_ori
                pad_right = W_img_v - W_ori
                pad_top = 0; pad_left = 0

                if pad_bottom < 0 or pad_right < 0:
                    raise ValueError(
                        f"img_shape smaller than ori_shape at b={b}, v={v}: "
                        f"ori=({H_ori},{W_ori}), img=({H_img_v},{W_img_v})")

                d_padded = F.pad(
                    d_resized.unsqueeze(0).unsqueeze(0),  # (1,1,H_ori,W_ori)
                    pad=(pad_left, pad_right, pad_top, pad_bottom),  # (l,r,t,b)
                    mode="replicate",
                ).squeeze(0).squeeze(0)  # (H_img_v, W_img_v)

                depth_b_list.append(d_padded)

            depth_b = torch.stack(depth_b_list, dim=0)   # (V, H_img_v, W_img_v)
            depth_out.append(depth_b)

        depth_out = torch.stack(depth_out, dim=0)  # (B, V, H_img, W_img)
        depth_out_cpu = depth_out.cpu()
  
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

            for b in range(B):
                meta = img_metas[b]
                filenames = meta.get("filename", None)
                if isinstance(filenames, (str, os.PathLike)):
                    filenames = [filenames]

                for v in range(V):
                    depth_map = depth_out_cpu[b, v]  # (H_img, W_img)

                    if filenames is not None:
                        img_path = filenames[v]
                        base_name = os.path.basename(img_path)          # xxx.jpg
                        name_wo_ext = os.path.splitext(base_name)[0]    # xxx
                        cam_name = os.path.basename(os.path.dirname(img_path))  # CAM_FRONT 등
                        depth_name = f"{name_wo_ext}_dpt_depth"
                    else:
                        cam_name = "UNKNOWN_CAM"
                        depth_name = f"sample{b}_view{v}_dpt_depth"

                    cam_dir = os.path.join(save_dir, cam_name)
                    os.makedirs(cam_dir, exist_ok=True)

                    npz_path = os.path.join(cam_dir, depth_name + ".npz")
                    # np.save(npy_path, depth_map.numpy().astype(np.float32))
                    np.savez_compressed(npz_path, depth=depth_map.numpy().astype(np.float32))

                    color_png_path = os.path.join(cam_dir, depth_name + "_color.png")
                    save_colored_depth_cv2(depth_map, color_png_path, gamma=0.5)

        return depth_out_cpu  # (B, V, H_img, W_img)



def save_colored_depth_cv2(depth_map: torch.Tensor, save_path: str, gamma=0.5):
    d = depth_map.clone().cpu().numpy()
    # 0~1 normalize
    d = d - d.min()
    if d.max() > 0:
        d = d / (d.max() + 1e-8)
    d = d ** gamma
    d_8bit = (d * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(d_8bit, cv2.COLORMAP_INFERNO)
    cv2.imwrite(save_path, colored)

class GetCLIPCond(nn.Module):
    """
    CLIP condition encoder (HF-only, Tensor-only).

    Inputs:
      - images: torch.Tensor of shape (B, C, H, W) or (B, V, C, H, W), RGB
                values can be in [0,1] or [0,255]

    Returns (preserves view dim if provided):
      - 'features': list of tuples per selected layer:
          [(tok_seq: (B,V,N,C), cls_tok: (B,V,C)), ...]
      - 'patch_hw': (Hp, Wp)
      - 'last_cls': (B, V, C)              # global token from last selected layer
      - 'last_tokens': (B, V, N, C)        # patch tokens from last selected layer
      - 'img_metas': passthrough
      - 'clip_geom': dict with preprocess meta (scale, H2W2, padding)
    """
    def __init__(
        self,
        model_id = "openai/clip-vit-base-patch16",  # e.g., "openai/clip-vit-large-patch14-336"
        device='cuda',
        input_size=None,  # if None, use model's vision_config.image_size
        symmetric_pad=True,
    ):
        super().__init__()
        self.device = device
        
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        
        vcfg = self.model.config
        self.hidden_dim = vcfg.hidden_size
        self.patch = vcfg.patch_size                   # CLIP ViT patch
        self.base_image_size = vcfg.image_size         # model's nominal input size (e.g., 224)
        self.input_min = input_size if None else self.base_image_size
        self.symmetric_pad = symmetric_pad

    # @torch.inference_mode()
    def image_preprocess(self, x: torch.Tensor):
        """
        x: (N, 3, H, W) float tensor in [0,1] or [0,255]
        -> (x_norm: (N, 3, H2, W2) on self.device, CLIP normalized,
            Hp, Wp: patch grid size = (H2//patch, W2//patch))
        """
        assert x.ndim == 4 and x.shape[1] == 3, f"Expected (N,3,H,W), got {x.shape}"
        x = x.to(dtype=torch.float32)
        if x.max() > 1.5:  # likely [0,255]
            x = x / 255.0
        x = x.clamp_(0.0, 1.0)

        N, C, H, W = x.shape
        
        # resize (bicubic)
        x = F.interpolate(x, size=(self.input_min, self.input_min), mode='bicubic', align_corners=False)

        x = x.to(self.device, non_blocking=True)
        
        H2 = W2 = self.input_min
        Hp, Wp = H2 // self.patch, W2 // self.patch

        extra_geom = {
            'scale': None,
            'H2W2': (H2, W2),
            'padding': None,
            'target_input_min': self.input_min,
            'patch': self.patch,
        }

        Hp, Wp = H2 // self.patch, W2 // self.patch
        return x, Hp, Wp, extra_geom

    # @torch.inference_mode()
    def forward(self, images: torch.Tensor, img_metas=None, n_layers: int = 4):
        """
        images: (B, V, C, H, W) or (V, C, H, W) (bs=1) or (B, C, H, W)
        """
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a torch.Tensor")

        if images.ndim == 5:
            B, V, C, H, W = images.shape
            x = images.reshape(B * V, C, H, W).contiguous()
        elif images.ndim == 4:
            # (B, C, H, W) or (V, C, H, W) when bs=1
            if images.shape[0] == 3 and images.shape[1] != 3:
                # guard for ambiguous shapes
                raise ValueError(f"Ambiguous shape {images.shape}; expected (B,3,H,W) or (V,3,H,W).")
            if images.shape[1] == 3:  # (B,3,H,W)
                B = images.shape[0]; V = 1
            else:                     # (V,3,H,W) (assume bs=1)
                B = 1; V = images.shape[0]
            x = images.reshape(B * V, 3, images.shape[-2], images.shape[-1]).contiguous()
        else:
            raise ValueError(f"Unexpected tensor shape: {images.shape}")

        # Preprocess
        x, Hp, Wp, extra_geom = self.image_preprocess(x)

        # CLIP vision forward 
        outputs = self.model(pixel_values=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # list[ (B*V, 1+N, C) ]

        hs_selected = hidden_states[-n_layers:] if n_layers > 0 else [hidden_states[-1]]

        feats_out = []
        for h in hs_selected:
            # h: (B*V, 1+N, C)
            cls_tok = h[:, 0]           # (B*V, C)
            tok    = h[:, 1:]           # (B*V, N, C) with N = Hp*Wp

            cls_tok = cls_tok.view(B, V, self.hidden_dim)          # (B,V,C)
            tok_seq = tok.view(B, V, Hp * Wp, self.hidden_dim)     # (B,V,N,C)
            feats_out.append((tok_seq, cls_tok))

        last_tok, last_cls = feats_out[-1][0], feats_out[-1][1]

        return {
            'feature_type': 'clip',
            'features': feats_out,          # list[(B,V,N,C),(B,V,C)]
            'patch_hw': (Hp, Wp),
            'last_cls': last_cls,           # (B, V, C)
            'last_tokens': last_tok,        # (B, V, N, C)
            'img_metas': img_metas,
            'geom': extra_geom
        }





    
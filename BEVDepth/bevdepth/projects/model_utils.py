# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from BEVFormer
#   (https://github.com/fundamentalvision/BEVFormer)
# Copyright (c) 2022 BEVFormer authors, licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import os
import torch
import importlib
from mmcv import Config
from mmcv.runner import (load_checkpoint, wrap_fp16_model)
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
import pytorch_lightning as pl
from argparse import ArgumentParser
from bevdepth.exps.nuscenes.dino_exp import BEVDepthLightningModel  # 네가 보여준 클래스


def get_bev_model(args):
    cfg = Config.fromfile(args.bev_config)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.bev_config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # set tf32
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        # init_dist(args.launcher, **cfg.dist_params)
        
    cfg.model.pretrained = None
    model = build_model(cfg.model, 
                        train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.bev_checkpoint, map_location='cpu')
    
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
        
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
        
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=True)
    model.eval()
    return model



def build_unet(cfg):
    def get_obj_from_str(string, reload=False):
        module, cls = string.rsplit(".", 1)
        if reload:
            module_imp = importlib.import_module(module)
            importlib.reload(module_imp)

        return getattr(importlib.import_module(module, package=None), cls)
    
    layout_encoder = get_obj_from_str(cfg.parameters.layout_encoder.type)(
        **cfg.parameters.layout_encoder.parameters
    )

    model_kwargs = dict(**cfg.parameters)
    model_kwargs.pop('layout_encoder')
    return get_obj_from_str(cfg.type)(
        layout_encoder=layout_encoder,
        **model_kwargs,
    )

def instantiate_from_config(cfg):
    def get_obj_from_str(string, reload=False):
        module, cls = string.rsplit(".", 1)
        if reload:
            module_imp = importlib.import_module(module)
            importlib.reload(module_imp)
            
        return getattr(importlib.import_module(module, package=None), cls)
    
    return get_obj_from_str(cfg.type)(**cfg.get("parameters", dict()))



def get_bevdepth_model(args, return_lightning=False):
    """
    BEVDepthLightningModel 을 checkpoint 없이 생성하여
    BEVFormer teacher처럼 사용할 수 있게 NN module만 반환하는 함수.

    args expected fields:
    - args.device        (optional, default 'cuda:0')
    - args.freeze        (optional, default True)
    - args.extra_hparams (optional, dict override)
    """

    # ----------------------------
    # 1) args value gathering
    # ----------------------------
    device = getattr(args, "device", "cuda:0")
    freeze = getattr(args, "freeze", True)
    extra_hparams = getattr(args, "extra_hparams", None)

    # ----------------------------
    # 2) Lightning hparams 생성
    # ----------------------------
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument("--seed", type=int, default=0)

    parser = BEVDepthLightningModel.add_model_specific_args(parent_parser)

    parser.set_defaults(
        profiler='simple',
        deterministic=False,
        max_epochs=24,
        accelerator='ddp',
        num_sanity_val_steps=0,
        gradient_clip_val=5,
        limit_val_batches=0,
        enable_checkpointing=False,  # checkpoint 안 쓰므로 disable 가능
        precision=16,
        default_root_dir='./outputs/bev_depth_uninitialized'
    )

    # CLI가 아니므로 default hparams 가져오기
    hparams = parser.parse_args([])

    if extra_hparams is not None:
        for k, v in extra_hparams.items():
            setattr(hparams, k, v)

    if getattr(hparams, "seed", None) is not None:
        pl.seed_everything(hparams.seed)

    # ----------------------------
    # 3) LightningModule 생성 (checkpoint load X)
    # ----------------------------
    lit_model = BEVDepthLightningModel(**vars(hparams))
    lit_model.eval()
    lit_model.to(device)

    if freeze:
        lit_model.freeze()

    # ----------------------------
    # 4) 반환 형태 결정
    # ----------------------------
    if return_lightning:
        return lit_model  # dataloader 포함된 LightningModule 자체 반환

    # 실제 backbone module (DINOBEVDepth)
    model = lit_model.model
    model.to(device)
    model.eval()

    if freeze:
        model.requires_grad_(False)

    return model
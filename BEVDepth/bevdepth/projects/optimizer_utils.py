import math
from copy import deepcopy
import torch

# mmcv 1.x / mmdet 2.x 기준 (mmengine 계열이면 import 경로가 다릅니다)
from mmcv.runner import build_optimizer  # <- bev_model에 대해 paramwise_cfg를 해석해 param_groups 생성

def build_bev_param_groups_from_cfg(bev_model, bev_optimizer_cfg):
    tmp = build_optimizer(bev_model, deepcopy(bev_optimizer_cfg))  # Constructor가 그룹 만들어줌
    groups = []
    for g in tmp.param_groups:
        d = {k: v for k, v in g.items() if k != "params"}
        d["params"] = list(g["params"])  # 참조 분리
        groups.append(d)
    del tmp
    return groups

def split_decay_groups(named_params, base_lr, weight_decay):
    """bias/norm은 WD=0, 나머지는 WD=weight_decay로 두 개 그룹 생성."""
    decay, no_decay = [], []
    for n, p in named_params:
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith(".bias") or "norm" in n.lower() or "bn" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    groups = []
    if decay:
        groups.append({"params": decay, "lr": base_lr, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "lr": base_lr, "weight_decay": 0.0})
    return groups


# def build_joint_optimizer(unet, bev_model, bev_optimizer_cfg,
#                           unet_lr, unet_betas=(0.9, 0.999), unet_weight_decay=0.0, unet_eps=1e-8):

#     bev_param_groups = build_bev_param_groups_from_cfg(bev_model, bev_optimizer_cfg)

#     unet_group = {
#         "params": list(unet.parameters()),
#         "lr": unet_lr,
#         "betas": unet_betas,
#         "weight_decay": unet_weight_decay,
#         "eps": unet_eps,
#     }

#     optimizer = torch.optim.AdamW([unet_group] + bev_param_groups)
#     return optimizer, len(bev_param_groups)


def build_joint_optimizer( unet, bev_model, fuser, bev_optimizer_cfg,
                            unet_lr, unet_betas=(0.9, 0.999), unet_weight_decay=0.0, unet_eps=1e-8):
    """
    Optimizer를 단일 AdamW로 구성:
    [ UNet 그룹 ] + [ BEV의 paramwise_cfg 그룹들 ] + [ Fuser의 decay/no_decay 그룹들 ]
    """
    bev_groups = build_bev_param_groups_from_cfg(bev_model, bev_optimizer_cfg)

    base_bev_lr = bev_optimizer_cfg.get("lr", 2e-4)
    bev_wd      = bev_optimizer_cfg.get("weight_decay", 0.01)

    fuser_groups = split_decay_groups(fuser.named_parameters(), base_bev_lr, bev_wd)

    unet_group = {
        "params": list(unet.parameters()),
        "lr": unet_lr, "betas": unet_betas,
        "weight_decay": unet_weight_decay, "eps": unet_eps,
    }

    optimizer = torch.optim.AdamW([unet_group] + bev_groups + fuser_groups)
    return optimizer, len(bev_groups), len(fuser_groups)



def cosine_warmup_lambda(step, *, total_iters, warmup_iters, warmup_ratio, min_lr_ratio):
    # warm up
    if step < warmup_iters:
        start = warmup_ratio
        end = 1.0
        pct = step / max(1, warmup_iters)
        return start + (end - start) * pct
    # cosine
    t = step - warmup_iters
    T = max(1, total_iters - warmup_iters)
    cos = 0.5 * (1 + math.cos(math.pi * t / T))
    return min_lr_ratio + (1 - min_lr_ratio) * cos


# def build_groupwise_lambda_scheduler(optimizer,
#                                      lambda_unet,
#                                      lambda_bev,
#                                      bev_groups_count):
#     """
#     - optimizer.param_groups 순서가 [UNet] + [BEV의 여러 그룹]이라고 가정
#     - 각 그룹에 대응하는 lr_lambda를 구성해 LambdaLR 반환
#     """
#     lr_lambda_list = [lambda_unet] + [lambda_bev] * bev_groups_count
#     return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_list)


def build_groupwise_lambda_scheduler(optimizer, lambda_unet, lambda_bev, bev_groups_count, fuser_groups_count):
    """
    param_groups 순서: [UNet] + [BEV x bev_groups_count] + [Fuser x fuser_groups_count]
    Fuser는 BEV와 동일 스케줄러(lambda_bev) 사용.
    """
    lr_lambdas = [lambda_unet] + [lambda_bev] * bev_groups_count + [lambda_bev] * fuser_groups_count
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambdas)
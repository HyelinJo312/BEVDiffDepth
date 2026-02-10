"""
Learning Rate Scheduler Utilities for BEV Diffusion Training.

Provides factory functions for creating step-based LR schedulers:
- MultiStepLR: Step-based decay at specified milestones
- SequentialLR: Warmup + CosineAnnealingLR
"""

import math
from typing import List, Optional, Union
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    MultiStepLR,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    LambdaLR,
)


def build_multistep_scheduler(
    optimizer: Optimizer,
    milestones: List[int],
    gamma: float = 0.1,
) -> MultiStepLR:
    """
    Build a step-based MultiStepLR scheduler.
    
    Args:
        optimizer: The optimizer to schedule.
        milestones: List of step indices at which to decay the LR.
        gamma: Multiplicative factor of LR decay. Default: 0.1
        
    Returns:
        MultiStepLR scheduler (call .step() after each optimizer.step())

    """
    return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    max_train_steps: int,
    num_processes: int = 1,
    warmup_ratio: float = 0.01,
    warmup_start_factor: float = 0.01,
    eta_min: float = 1e-6,
) -> SequentialLR:
    """
    Build a Warmup + CosineAnnealingLR scheduler for distributed training.
    
    - Warmup phase: LR linearly increases from (warmup_start_factor * base_lr) to base_lr
    - Cosine phase: LR decays from base_lr to eta_min following cosine curve
    
    Note: For distributed training with accelerate, scheduler.step() is called
    independently on each process. The num_processes parameter scales the step
    counts accordingly so the scheduler behaves as intended.
    
    Args:
        optimizer: The optimizer to schedule.
        max_train_steps: Total number of training steps (before scaling).
        num_processes: Number of distributed processes (accelerator.num_processes). Default: 1
        warmup_ratio: Fraction of total steps for warmup. Default: 0.01 (1%)
        warmup_start_factor: Initial LR = base_lr * warmup_start_factor. Default: 0.01
        eta_min: Minimum LR at the end of cosine decay. Default: 1e-6
        
    Returns:
        SequentialLR scheduler (call .step() after each optimizer.step())
    """
    # Scale steps for distributed training
    scaled_max_steps = max_train_steps * num_processes
    warmup_steps = int(max_train_steps * warmup_ratio) * num_processes
    cosine_steps = scaled_max_steps - warmup_steps
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=eta_min,
    )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )
    
    return scheduler


def build_warmup_multistep_scheduler(
    optimizer: Optimizer,
    max_train_steps: int,
    milestones: List[int],
    num_processes: int = 1,
    warmup_ratio: float = 0.01,
    warmup_start_factor: float = 0.01,
    gamma: float = 0.1,
) -> SequentialLR:
    """
    Build a Warmup + MultiStepLR scheduler for distributed training.
    
    - Warmup phase: LR linearly increases from (warmup_start_factor * base_lr) to base_lr
    - MultiStep phase: LR decays by gamma at each milestone step
    
    Note: milestones should be step indices BEFORE scaling by num_processes.
          The function will scale them internally for distributed training.
    
    Args:
        optimizer: The optimizer to schedule.
        max_train_steps: Total number of training steps (before scaling).
        milestones: List of step indices at which to decay the LR (before scaling).
        num_processes: Number of distributed processes (accelerator.num_processes). Default: 1
        warmup_ratio: Fraction of total steps for warmup. Default: 0.01 (1%)
        warmup_start_factor: Initial LR = base_lr * warmup_start_factor. Default: 0.01
        gamma: Multiplicative factor of LR decay. Default: 0.1
        
    Returns:
        SequentialLR scheduler
    """
    # Scale steps for distributed training
    warmup_steps = int(max_train_steps * warmup_ratio) * num_processes
    scaled_milestones = [m * num_processes for m in milestones]
    
    # Adjust milestones to be relative to after warmup
    adjusted_milestones = [m - warmup_steps for m in scaled_milestones if m > warmup_steps]
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    
    multistep_scheduler = MultiStepLR(
        optimizer,
        milestones=adjusted_milestones,
        gamma=gamma,
    )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, multistep_scheduler],
        milestones=[warmup_steps],
    )
    
    return scheduler


def get_scheduler_info(
    scheduler_type: str,
    max_train_steps: int,
    num_processes: int = 1,
    warmup_ratio: float = 0.01,
    milestones: Optional[List[int]] = None,
    gamma: float = 0.1,
    eta_min: float = 1e-6,
) -> str:
    """
    Get a human-readable description of the scheduler configuration.
    Shows the user-facing step counts (before num_processes scaling).
    
    Args:
        scheduler_type: One of 'multistep', 'cosine', 'warmup_multistep', 'warmup_cosine'
        max_train_steps: Total training steps (before scaling)
        num_processes: Number of distributed processes
        warmup_ratio: Warmup ratio (for warmup-based schedulers)
        milestones: Step milestones (for multistep schedulers, before scaling)
        gamma: Decay factor (for multistep schedulers)
        eta_min: Minimum LR (for cosine schedulers)
        
    Returns:
        String description of the scheduler
    """
    # Show user-facing steps (before scaling)
    warmup_steps = int(max_train_steps * warmup_ratio)
    
    if scheduler_type == 'multistep':
        return f"MultiStepLR(milestones={milestones}, gamma={gamma})"
    
    elif scheduler_type == 'cosine':
        return f"CosineAnnealingLR(T_max={max_train_steps}, eta_min={eta_min})"
    
    elif scheduler_type == 'warmup_cosine':
        return (
            f"Warmup({warmup_steps} steps) + "
            f"CosineAnnealing(T_max={max_train_steps - warmup_steps}, eta_min={eta_min})"
        )
    
    elif scheduler_type == 'warmup_multistep':
        return (
            f"Warmup({warmup_steps} steps) + "
            f"MultiStepLR(milestones={milestones}, gamma={gamma})"
        )
    
    else:
        return f"Unknown scheduler type: {scheduler_type}"


def build_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    max_train_steps: int,
    num_processes: int = 1,
    warmup_ratio: float = 0.01,
    warmup_start_factor: float = 0.01,
    milestones: Optional[List[int]] = None,
    gamma: float = 0.1,
    eta_min: float = 1e-6,
):
    """
    Factory function to build various LR schedulers for distributed training.
    
    Important: For distributed training with accelerate, each process calls
    scheduler.step() independently. This function handles the scaling internally
    using num_processes, so you should pass the original (unscaled) max_train_steps
    and milestones.
    
    Args:
        optimizer: The optimizer to schedule.
        scheduler_type: One of:
            - 'multistep': MultiStepLR (no warmup)
            - 'cosine': CosineAnnealingLR (no warmup)
            - 'warmup_cosine': Warmup + CosineAnnealingLR
            - 'warmup_multistep': Warmup + MultiStepLR
        max_train_steps: Total number of training steps (BEFORE num_processes scaling).
        num_processes: Number of distributed processes (accelerator.num_processes). Default: 1
        warmup_ratio: Fraction of steps for warmup. Default: 0.01
        warmup_start_factor: Initial LR factor during warmup. Default: 0.01
        milestones: Step indices for LR decay (BEFORE num_processes scaling). Default: None
        gamma: Decay factor for multistep. Default: 0.1
        eta_min: Minimum LR for cosine. Default: 1e-6
        
    Returns:
        LR scheduler instance
        
    Example:
        >>> # With 4 GPUs, if you want warmup to end at step 1000:
        >>> scheduler = build_scheduler(
        ...     optimizer,
        ...     scheduler_type='warmup_cosine',
        ...     max_train_steps=100000,  # Original unscaled value
        ...     num_processes=4,          # accelerator.num_processes
        ...     warmup_ratio=0.01,
        ... )
    """
    if scheduler_type == 'multistep':
        if milestones is None:
            raise ValueError("milestones must be provided for 'multistep' scheduler")
        # Scale milestones for distributed training
        scaled_milestones = [m * num_processes for m in milestones]
        return build_multistep_scheduler(optimizer, scaled_milestones, gamma)
    
    elif scheduler_type == 'cosine':
        scaled_max_steps = max_train_steps * num_processes
        return CosineAnnealingLR(optimizer, T_max=scaled_max_steps, eta_min=eta_min)
    
    elif scheduler_type == 'warmup_cosine':
        return build_warmup_cosine_scheduler(
            optimizer, max_train_steps, num_processes, warmup_ratio, warmup_start_factor, eta_min
        )
    
    elif scheduler_type == 'warmup_multistep':
        if milestones is None:
            raise ValueError("milestones must be provided for 'warmup_multistep' scheduler")
        return build_warmup_multistep_scheduler(
            optimizer, max_train_steps, milestones, num_processes, warmup_ratio, warmup_start_factor, gamma
        )
    
    else:
        raise ValueError(
            f"Unknown scheduler_type: {scheduler_type}. "
            f"Choose from: 'multistep', 'cosine', 'warmup_cosine', 'warmup_multistep'"
        )

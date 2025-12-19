"""
Optimizer and Learning Rate Scheduler
Uses PyTorch built-in schedulers with custom warmup strategies.
"""
import torch
import torch.optim as optim
import math
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from typing import Optional


def get_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = 'adamw',
    lr: float = 0.0001,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.98),
    eps: float = 1e-9
) -> optim.Optimizer:
    """
    Create optimizer.
    
    Args:
        model: Model to optimize
        optimizer_type: 'adam' or 'adamw'
        lr: Learning rate
        weight_decay: Weight decay for AdamW
        betas: Adam beta parameters
        eps: Adam epsilon
        
    Returns:
        Optimizer instance
    """
    # Separate parameters that should/shouldn't have weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Don't apply weight decay to biases and layer norm
        if 'bias' in name or 'norm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    if optimizer_type == 'adam':
        return optim.Adam(param_groups, lr=lr, betas=betas, eps=eps)
    elif optimizer_type == 'adamw':
        return optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def get_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = 'warmup',
    d_model: int = 512,
    warmup_steps: int = 4000,
    total_steps: Optional[int] = None
):
    """
    Create learning rate scheduler using PyTorch built-in schedulers.
    
    Args:
        optimizer: Optimizer instance
        scheduler_type: 'warmup', 'cosine', or 'none'
        d_model: Model dimension (for warmup scheduler)
        warmup_steps: Number of warmup steps
        total_steps: Total training steps (for cosine scheduler)
        
    Returns:
        PyTorch LR Scheduler instance or None
    """
    if scheduler_type == 'warmup':
        # Transformer warmup schedule: d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
        def lr_lambda(step):
            step = max(step, 1)  # Avoid division by zero
            return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
        
        return LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    elif scheduler_type == 'cosine':
        if total_steps is None:
            raise ValueError("total_steps required for cosine scheduler")
        
        # Warmup + Cosine Decay using SequentialLR
        # Phase 1: Linear warmup
        def warmup_lambda(step):
            return float(step) / float(max(1, warmup_steps))
        
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
        
        # Phase 2: Cosine annealing
        cosine_steps = total_steps - warmup_steps
        cosine_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=cosine_steps,
            eta_min=0
        )
        
        # Combine both phases
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        return scheduler
    
    elif scheduler_type == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


if __name__ == "__main__":
    # Test schedulers with PyTorch built-in
    import matplotlib.pyplot as plt
    
    # Create dummy model and optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Test warmup scheduler
    scheduler = get_scheduler(optimizer, scheduler_type='warmup', d_model=512, warmup_steps=4000)
    
    lrs = []
    for step in range(1, 20001):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])
    
    plt.figure(figsize=(10, 5))
    plt.plot(lrs)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Warmup Learning Rate Schedule (PyTorch LambdaLR)')
    plt.savefig('warmup_schedule.png')
    print("Saved learning rate schedule plot")

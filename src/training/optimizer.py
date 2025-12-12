"""
Optimizer and Learning Rate Scheduler
Implements Adam/AdamW optimizers and Warmup learning rate scheduler.
"""
import torch
import torch.optim as optim
import math
from typing import Optional


class WarmupScheduler:
    """
    Learning Rate Scheduler with Warmup
    
    Implements the learning rate schedule from "Attention Is All You Need":
    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    
    Args:
        optimizer: PyTorch optimizer
        d_model: Model dimension
        warmup_steps: Number of warmup steps
        factor: Scaling factor for learning rate
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        factor: float = 1.0
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = 0
        self._rate = 0
    
    def step(self):
        """Update learning rate and step."""
        self._step += 1
        rate = self._get_rate()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = rate
        self._rate = rate
    
    def _get_rate(self) -> float:
        """Compute learning rate for current step."""
        step = self._step
        return self.factor * (
            self.d_model ** (-0.5) *
            min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        )
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self._rate
    
    def state_dict(self) -> dict:
        """Return scheduler state."""
        return {
            'step': self._step,
            'rate': self._rate
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load scheduler state."""
        self._step = state_dict['step']
        self._rate = state_dict['rate']


class CosineWarmupScheduler:
    """
    Cosine Annealing with Warmup
    
    Linear warmup followed by cosine decay.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr: Minimum learning rate
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self._step = 0
        self._rate = 0
    
    def step(self):
        """Update learning rate and step."""
        self._step += 1
        rate = self._get_rate()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = rate
        self._rate = rate
    
    def _get_rate(self) -> float:
        """Compute learning rate for current step."""
        if self._step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * self._step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self._step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self._rate
    
    def state_dict(self) -> dict:
        """Return scheduler state."""
        return {
            'step': self._step,
            'rate': self._rate
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load scheduler state."""
        self._step = state_dict['step']
        self._rate = state_dict['rate']


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
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        scheduler_type: 'warmup', 'cosine', or 'none'
        d_model: Model dimension (for warmup scheduler)
        warmup_steps: Number of warmup steps
        total_steps: Total training steps (for cosine scheduler)
        
    Returns:
        Scheduler instance or None
    """
    if scheduler_type == 'warmup':
        return WarmupScheduler(optimizer, d_model, warmup_steps)
    elif scheduler_type == 'cosine':
        if total_steps is None:
            raise ValueError("total_steps required for cosine scheduler")
        return CosineWarmupScheduler(optimizer, warmup_steps, total_steps)
    elif scheduler_type == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


if __name__ == "__main__":
    # Test schedulers
    import matplotlib.pyplot as plt
    
    # Create dummy model and optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Test warmup scheduler
    scheduler = WarmupScheduler(optimizer, d_model=512, warmup_steps=4000)
    
    lrs = []
    for step in range(20000):
        scheduler.step()
        lrs.append(scheduler.get_lr())
    
    plt.figure(figsize=(10, 5))
    plt.plot(lrs)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Warmup Learning Rate Schedule')
    plt.savefig('warmup_schedule.png')
    print("Saved learning rate schedule plot")

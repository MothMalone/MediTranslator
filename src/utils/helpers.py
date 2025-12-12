"""
Helper Functions
Various utility functions for training and evaluation.
"""
import torch
import torch.nn as nn
import random
import numpy as np
from typing import Optional


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the device to use for computation.
    
    Args:
        device: Device string ('cuda', 'cpu', or None for auto)
        
    Returns:
        torch.device instance
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: Whether to count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_number(n: int) -> str:
    """
    Format a large number with K/M/B suffixes.
    
    Args:
        n: Number to format
        
    Returns:
        Formatted string
    """
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)


def get_model_summary(model: nn.Module) -> str:
    """
    Get a summary of the model architecture.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model summary string
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    summary = []
    summary.append("=" * 60)
    summary.append("Model Summary")
    summary.append("=" * 60)
    summary.append(f"Total parameters: {format_number(total_params)} ({total_params:,})")
    summary.append(f"Trainable parameters: {format_number(trainable_params)} ({trainable_params:,})")
    summary.append(f"Non-trainable parameters: {format_number(total_params - trainable_params)}")
    summary.append("=" * 60)
    
    return "\n".join(summary)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    loss: float,
    filepath: str,
    scheduler=None,
    config: dict = None
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        step: Current step
        loss: Current loss
        filepath: Output path
        scheduler: Optional scheduler state
        config: Optional configuration
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'loss': loss
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if config is not None:
        checkpoint['config'] = config
    
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler=None,
    device: torch.device = None
) -> dict:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load to
        
    Returns:
        Checkpoint dictionary with metadata
    """
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


if __name__ == "__main__":
    # Test helper functions
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")
    
    # Test with a simple model
    model = nn.Linear(100, 50)
    print(get_model_summary(model))

"""
Trainer Module
Main training loop for the Transformer model.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Callable
import os
import logging
from tqdm import tqdm
import time

from .loss import LabelSmoothingLoss, CrossEntropyLoss
from .metrics import MetricsTracker, compute_token_accuracy

logger = logging.getLogger(__name__)

class Trainer:
    """
    Trainer class for Transformer model.
    
    Handles:
        - Training loop with gradient accumulation
        - Validation
        - Checkpointing
        - Logging
        - Early stopping
    
    Args:
        model: Transformer model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Device to train on
        tgt_vocab: Target vocabulary (for loss function)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
        tgt_vocab=None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        
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
from .optimizer import get_optimizer, get_scheduler
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
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Training config
        train_config = config.get('training', {})
        self.epochs = train_config.get('epochs', 50)
        self.grad_accum_steps = train_config.get('gradient_accumulation_steps', 1)
        self.max_grad_norm = train_config.get('max_grad_norm', 1.0)
        self.save_every = train_config.get('save_every', 1000)
        self.eval_every = train_config.get('eval_every', 500)
        self.log_every = train_config.get('log_every', 100)
        self.early_stopping_patience = train_config.get('early_stopping_patience', 5)
        
        # Paths
        paths = config.get('paths', {})
        self.checkpoint_dir = paths.get('checkpoint_dir', 'checkpoints')
        self.log_dir = paths.get('log_dir', 'logs')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Loss function
        vocab_config = config.get('vocab', {})
        if tgt_vocab is not None:
            tgt_vocab_size = len(tgt_vocab)  # Use actual vocab size
        else:
            tgt_vocab_size = vocab_config.get('tgt_vocab_size', 32000)  # Fallback to config
        smoothing = train_config.get('label_smoothing', 0.0)
        
        if smoothing > 0.0:
            self.criterion = LabelSmoothingLoss(
                vocab_size=tgt_vocab_size,
                padding_idx=0,
                smoothing=smoothing
            )
            logger.info(f"Using Label Smoothing Loss (smoothing={smoothing})")
        else:
            self.criterion = CrossEntropyLoss(padding_idx=0)
            logger.info("Using Cross-Entropy Loss")
        
        # Optimizer
        model_config = config.get('model', {})
        self.optimizer = get_optimizer(
            model,
            optimizer_type=train_config.get('optimizer', 'adamw'),
            lr=train_config.get('learning_rate', 0.0001),
            weight_decay=train_config.get('weight_decay', 0.01),
            betas=tuple(train_config.get('betas', [0.9, 0.98])),
            eps=train_config.get('eps', 1e-9)
        )
        
        # Calculate total steps for scheduler (if needed)
        steps_per_epoch = len(train_loader) // train_config.get('gradient_accumulation_steps', 1)
        total_steps = steps_per_epoch * train_config.get('epochs', 50)
        
        # Scheduler
        self.scheduler = get_scheduler(
            self.optimizer,
            scheduler_type=train_config.get('scheduler', 'warmup'),
            d_model=model_config.get('d_model', 512),
            warmup_steps=train_config.get('warmup_steps', 4000),
            total_steps=total_steps
        )
        
        # Metrics
        self.metrics = MetricsTracker()
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self, epoch: int):
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
        """
        self.model.train()
        
        epoch_loss = 0.0
        epoch_tokens = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Unpack batch
            src, tgt, src_lengths, tgt_lengths = batch
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            # Prepare input and target
            # Input: all tokens except last
            # Target: all tokens except first (shifted)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Forward pass
            logits = self.model(src, tgt_input)
            
            # Compute loss
            loss = self.criterion(logits, tgt_output)
            
            # Normalize loss for gradient accumulation
            loss = loss / self.grad_accum_steps
            
            # Backward pass
            loss.backward()
            
            # Count tokens
            n_tokens = (tgt_output != 0).sum().item()
            epoch_loss += loss.item() * self.grad_accum_steps * n_tokens
            epoch_tokens += n_tokens
            
            # Update metrics
            self.metrics.update(
                loss.item() * self.grad_accum_steps,
                n_tokens,
                self.scheduler.get_lr() if self.scheduler else self.optimizer.param_groups[0]['lr']
            )
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # Gradient clipping
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Scheduler step
                if self.scheduler:
                    self.scheduler.step()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.log_every == 0:
                    train_loss, train_ppl = self.metrics.log_train_step(
                        self.global_step, epoch
                    )
                    lr = self.scheduler.get_lr() if self.scheduler else self.optimizer.param_groups[0]['lr']
                    progress_bar.set_postfix({
                        'loss': f'{train_loss:.4f}',
                        'ppl': f'{train_ppl:.2f}',
                        'lr': f'{lr:.2e}'
                    })
                
                # Evaluation
                if self.global_step % self.eval_every == 0:
                    val_loss = self.validate()
                    self.model.train()
                    
                    # Check for improvement
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                        self.save_checkpoint('best_model.pt')
                    else:
                        self.patience_counter += 1
                
                # Save checkpoint
                if self.global_step % self.save_every == 0:
                    self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')
        
        return epoch_loss / max(epoch_tokens, 1)
    
    @torch.no_grad()
    def validate(self) -> float:
        """
        Validate the model.
        
        Returns:
            Validation loss
        """
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        
        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            src, tgt, src_lengths, tgt_lengths = batch
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            logits = self.model(src, tgt_input)
            
            loss = self.criterion(logits, tgt_output)
            
            n_tokens = (tgt_output != 0).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
        
        val_loss = total_loss / max(total_tokens, 1)
        val_loss, val_ppl = self.metrics.log_val_step(val_loss, self.global_step)
        
        logger.info(f"Validation - Loss: {val_loss:.4f}, PPL: {val_ppl:.2f}")
        
        return val_loss
    
    def train(self):
        """Main training loop."""
        logger.info(f"Starting training for {self.epochs} epochs")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            train_loss = self.train_epoch(epoch)
            
            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch+1}/{self.epochs} completed in {epoch_time:.2f}s - Train Loss: {train_loss:.4f}")
            
            # Validate at end of epoch
            val_loss = self.validate()
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
                logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
            
            # Save epoch checkpoint
            self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/3600:.2f} hours")
        
        # Save final metrics
        self.metrics.save(os.path.join(self.log_dir, 'metrics.json'))
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Loaded checkpoint from {filepath}")


if __name__ == "__main__":
    print("Trainer module loaded successfully")

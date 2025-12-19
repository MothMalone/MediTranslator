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
from ..inference.greedy_search import greedy_decode
from ..evaluation.bleu import corpus_bleu

logger = logging.getLogger(__name__)

# Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available. Install with: pip install wandb")


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
        tgt_vocab=None,
        src_vocab=None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.tgt_vocab = tgt_vocab
        self.src_vocab = src_vocab
        
        # Training config
        train_config = config.get('training', {})
        self.epochs = train_config.get('epochs', 50)
        self.grad_accum_steps = train_config.get('gradient_accumulation_steps', 1)
        self.max_grad_norm = train_config.get('max_grad_norm', 1.0)
        self.save_every = train_config.get('save_every', 1000)
        self.eval_every = train_config.get('eval_every', 500)
        self.log_every = train_config.get('log_every', 100)
        self.early_stopping_patience = train_config.get('early_stopping_patience', 5)
        self.compute_bleu = train_config.get('compute_bleu', True)
        self.bleu_max_samples = train_config.get('bleu_max_samples', 1000)
        self.use_mixed_precision = train_config.get('use_mixed_precision', False)
        
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
        
        # Checkpoint management
        self.keep_last_n_checkpoints = train_config.get('keep_last_n_checkpoints', 2)
        self.epoch_checkpoints = []  # Track epoch checkpoint files
        
        # Mixed Precision Training
        self.scaler = None
        if self.use_mixed_precision and self.device.type == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
            logger.info("Mixed Precision Training (FP16) enabled")
        elif self.use_mixed_precision:
            logger.warning("Mixed precision requested but CUDA not available")
            self.use_mixed_precision = False
        
        # Weights & Biases
        self.use_wandb = train_config.get('use_wandb', False)
        if self.use_wandb and WANDB_AVAILABLE:
            # Initialize wandb
            wandb_config = config.get('wandb', {})
            project_name = wandb_config.get('project', 'transformer-mt')
            run_name = config.get('version', {}).get('name', 'experiment')
            
            wandb.init(
                project=project_name,
                name=run_name,
                config={
                    'model': model_config,
                    'training': train_config,
                    'vocab': vocab_config
                }
            )
            wandb.watch(self.model, log='all', log_freq=1000)
            logger.info("Weights & Biases logging enabled")
        elif self.use_wandb:
            logger.warning("wandb requested but not available")
            self.use_wandb = False
    
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
            
            # Forward pass with optional mixed precision
            if self.use_mixed_precision:
                with torch.amp.autocast(device_type='cuda'):
                    logits = self.model(src, tgt_input)
                    loss = self.criterion(logits, tgt_output)
            else:
                logits = self.model(src, tgt_input)
                loss = self.criterion(logits, tgt_output)
            
            # Normalize loss for gradient accumulation
            loss = loss / self.grad_accum_steps
            
            # Backward pass with gradient scaling
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Count tokens
            n_tokens = (tgt_output != 0).sum().item()
            epoch_loss += loss.item() * self.grad_accum_steps * n_tokens
            epoch_tokens += n_tokens
            
            # Update metrics
            self.metrics.update(
                loss.item() * self.grad_accum_steps,
                n_tokens,
                self.optimizer.param_groups[0]['lr']
            )
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # Gradient clipping
                if self.max_grad_norm > 0:
                    if self.use_mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                
                # Optimizer step
                if self.use_mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
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
                    lr = self.optimizer.param_groups[0]['lr']
                    progress_bar.set_postfix({
                        'loss': f'{train_loss:.4f}',
                        'ppl': f'{train_ppl:.2f}',
                        'lr': f'{lr:.2e}'
                    })
                    
                    # Log to wandb
                    if self.use_wandb:
                        wandb.log({
                            'train/loss': train_loss,
                            'train/perplexity': train_ppl,
                            'train/learning_rate': lr,
                            'train/epoch': epoch,
                            'train/step': self.global_step
                        }, step=self.global_step)
                
                # Evaluation
                if self.global_step % self.eval_every == 0:
                    val_loss = self.validate()
                    self.model.train()
                    
                    # Log validation to wandb (BLEU is logged inside validate())
                    if self.use_wandb:
                        wandb.log({
                            'val/loss': val_loss,
                            'val/perplexity': torch.exp(torch.tensor(val_loss)).item(),
                            'val/best_loss': self.best_val_loss
                        }, step=self.global_step)
                    
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
        
        # For BLEU calculation
        hypotheses = []
        references = []
        sample_count = 0
        
        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            src, tgt, src_lengths, tgt_lengths = batch
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Forward pass (causal mask auto-generated by decoder)
            logits = self.model(src, tgt_input)
            
            loss = self.criterion(logits, tgt_output)
            
            n_tokens = (tgt_output != 0).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            
            # Generate translations for BLEU (limited samples to save time)
            if self.compute_bleu and self.tgt_vocab is not None and sample_count < self.bleu_max_samples:
                generated = greedy_decode(
                    self.model,
                    src,
                    max_length=tgt.size(1),
                    bos_idx=self.tgt_vocab.bos_idx if hasattr(self.tgt_vocab, 'bos_idx') else 2,
                    eos_idx=self.tgt_vocab.eos_idx if hasattr(self.tgt_vocab, 'eos_idx') else 3,
                    pad_idx=self.tgt_vocab.pad_idx if hasattr(self.tgt_vocab, 'pad_idx') else 0
                )
                
                # Convert to tokens
                for i in range(min(src.size(0), self.bleu_max_samples - sample_count)):
                    # Hypothesis
                    hyp_ids = generated[i].cpu().tolist()
                    hyp_tokens = []
                    for idx in hyp_ids:
                        if idx in [self.tgt_vocab.pad_idx, self.tgt_vocab.bos_idx]:  # Skip PAD and BOS
                            continue
                        if idx == self.tgt_vocab.eos_idx:  # Stop at EOS
                            break
                        token = self.tgt_vocab.idx2token.get(idx, '<unk>') if hasattr(self.tgt_vocab, 'idx2token') else str(idx)
                        hyp_tokens.append(token)
                    
                    # Reference
                    ref_ids = tgt[i].cpu().tolist()
                    ref_tokens = []
                    for idx in ref_ids:
                        if idx in [self.tgt_vocab.pad_idx, self.tgt_vocab.bos_idx]:  # Skip PAD and BOS
                            continue
                        if idx == self.tgt_vocab.eos_idx:  # Stop at EOS
                            break
                        token = self.tgt_vocab.idx2token.get(idx, '<unk>') if hasattr(self.tgt_vocab, 'idx2token') else str(idx)
                        ref_tokens.append(token)
                    
                    if hyp_tokens and ref_tokens:
                        hypotheses.append(hyp_tokens)
                        references.append([ref_tokens])
                        sample_count += 1
                    
                    if sample_count >= self.bleu_max_samples:
                        break
        
        val_loss = total_loss / max(total_tokens, 1)
        val_loss, val_ppl = self.metrics.log_val_step(val_loss, self.global_step)
        
        # Calculate BLEU score
        bleu_score = 0.0
        if self.compute_bleu and hypotheses:
            try:
                bleu_score = corpus_bleu(hypotheses, references, max_n=4) * 100  # Convert to percentage
                logger.info(f"Validation - Loss: {val_loss:.4f}, PPL: {val_ppl:.2f}, BLEU: {bleu_score:.2f}")
            except Exception as e:
                logger.warning(f"Failed to compute BLEU: {e}")
                logger.info(f"Validation - Loss: {val_loss:.4f}, PPL: {val_ppl:.2f}")
        else:
            logger.info(f"Validation - Loss: {val_loss:.4f}, PPL: {val_ppl:.2f}")
        
        # Log BLEU to wandb if available
        if self.use_wandb and bleu_score > 0:
            wandb.log({'val/bleu': bleu_score}, step=self.global_step)
        
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
    
    def cleanup_old_checkpoints(self):
        """Remove old epoch checkpoints, keeping only the last N."""
        while len(self.epoch_checkpoints) > self.keep_last_n_checkpoints:
            old_checkpoint = self.epoch_checkpoints.pop(0)
            try:
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)
                    logger.info(f"Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to remove {old_checkpoint}: {e}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # Track epoch checkpoints (not best_model.pt or step checkpoints)
        is_epoch_checkpoint = filename.startswith('checkpoint_epoch_')
        
        # Cleanup old checkpoints before saving new one to ensure space
        if is_epoch_checkpoint:
            self.cleanup_old_checkpoints()
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        try:
            torch.save(checkpoint, filepath)
            logger.info(f"Saved checkpoint to {filepath}")
            
            # Add to tracking list
            if is_epoch_checkpoint:
                self.epoch_checkpoints.append(filepath)
        except Exception as e:
            logger.error(f"Failed to save checkpoint {filepath}: {e}")
            raise
    
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

"""
Metrics Tracking
Tracks training metrics like loss, perplexity, and learning rate.
"""
import torch
import math
from typing import Dict, List, Optional
from collections import defaultdict
import json
import logging

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Tracks and computes training/evaluation metrics.
    
    Metrics tracked:
        - Loss (training and validation)
        - Perplexity
        - Learning rate
        - Tokens per second
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.train_losses = []
        self.val_losses = []
        self.train_perplexities = []
        self.val_perplexities = []
        self.learning_rates = []
        self.steps = []
        self.epochs = []
        
        # Running statistics
        self._running_loss = 0.0
        self._running_tokens = 0
        self._running_steps = 0
    
    def update(
        self,
        loss: float,
        n_tokens: int,
        learning_rate: Optional[float] = None
    ):
        """
        Update running metrics.
        
        Args:
            loss: Loss value for current batch
            n_tokens: Number of tokens in batch
            learning_rate: Current learning rate
        """
        self._running_loss += loss * n_tokens
        self._running_tokens += n_tokens
        self._running_steps += 1
        
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
    
    def get_average_loss(self) -> float:
        """Get average loss over accumulated batches."""
        if self._running_tokens == 0:
            return 0.0
        return self._running_loss / self._running_tokens
    
    def get_perplexity(self, loss: Optional[float] = None) -> float:
        """
        Compute perplexity from loss.
        
        Args:
            loss: Loss value (uses running average if not provided)
            
        Returns:
            Perplexity value
        """
        if loss is None:
            loss = self.get_average_loss()
        return math.exp(min(loss, 100))  # Cap to avoid overflow
    
    def log_train_step(self, step: int, epoch: int):
        """
        Log metrics for a training step.
        
        Args:
            step: Current step number
            epoch: Current epoch number
        """
        loss = self.get_average_loss()
        ppl = self.get_perplexity(loss)
        
        self.train_losses.append(loss)
        self.train_perplexities.append(ppl)
        self.steps.append(step)
        
        # Reset running stats
        self._running_loss = 0.0
        self._running_tokens = 0
        self._running_steps = 0
        
        return loss, ppl
    
    def log_val_step(self, loss: float, epoch: int):
        """
        Log metrics for validation.
        
        Args:
            loss: Validation loss
            epoch: Current epoch
        """
        ppl = self.get_perplexity(loss)
        
        self.val_losses.append(loss)
        self.val_perplexities.append(ppl)
        self.epochs.append(epoch)
        
        return loss, ppl
    
    def get_best_val_loss(self) -> float:
        """Get best validation loss."""
        if not self.val_losses:
            return float('inf')
        return min(self.val_losses)
    
    def get_summary(self) -> Dict:
        """Get summary of tracked metrics."""
        return {
            'train_loss': self.train_losses[-1] if self.train_losses else None,
            'val_loss': self.val_losses[-1] if self.val_losses else None,
            'train_ppl': self.train_perplexities[-1] if self.train_perplexities else None,
            'val_ppl': self.val_perplexities[-1] if self.val_perplexities else None,
            'best_val_loss': self.get_best_val_loss(),
            'learning_rate': self.learning_rates[-1] if self.learning_rates else None,
            'n_steps': len(self.steps),
            'n_epochs': len(set(self.epochs)) if self.epochs else 0
        }
    
    def save(self, filepath: str):
        """Save metrics to JSON file."""
        data = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_perplexities': self.train_perplexities,
            'val_perplexities': self.val_perplexities,
            'learning_rates': self.learning_rates,
            'steps': self.steps,
            'epochs': self.epochs
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved metrics to {filepath}")
    
    def load(self, filepath: str):
        """Load metrics from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.train_losses = data.get('train_losses', [])
        self.val_losses = data.get('val_losses', [])
        self.train_perplexities = data.get('train_perplexities', [])
        self.val_perplexities = data.get('val_perplexities', [])
        self.learning_rates = data.get('learning_rates', [])
        self.steps = data.get('steps', [])
        self.epochs = data.get('epochs', [])
        
        logger.info(f"Loaded metrics from {filepath}")


def compute_token_accuracy(
    logits: torch.Tensor,
    target: torch.Tensor,
    padding_idx: int = 0
) -> float:
    """
    Compute token-level accuracy.
    
    Args:
        logits: Model predictions (batch, seq_len, vocab)
        target: Target indices (batch, seq_len)
        padding_idx: Padding token index
        
    Returns:
        Accuracy as a float
    """
    # Get predictions
    predictions = logits.argmax(dim=-1)
    
    # Create mask for non-padding tokens
    mask = (target != padding_idx)
    
    # Compute accuracy
    correct = (predictions == target) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    
    return accuracy.item()


if __name__ == "__main__":
    # Test metrics tracker
    tracker = MetricsTracker()
    
    # Simulate training
    for step in range(100):
        loss = 5.0 - step * 0.04  # Decreasing loss
        n_tokens = 1000
        lr = 0.001 * min(1.0, step / 10)  # Warmup
        
        tracker.update(loss, n_tokens, lr)
        
        if (step + 1) % 10 == 0:
            train_loss, train_ppl = tracker.log_train_step(step, epoch=0)
            print(f"Step {step+1}: Loss={train_loss:.4f}, PPL={train_ppl:.2f}")
    
    # Simulate validation
    val_loss, val_ppl = tracker.log_val_step(2.5, epoch=0)
    print(f"Validation: Loss={val_loss:.4f}, PPL={val_ppl:.2f}")
    
    # Get summary
    print("\nSummary:", tracker.get_summary())

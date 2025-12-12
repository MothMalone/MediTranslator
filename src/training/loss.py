"""
Loss Functions
Implements Cross-Entropy Loss with Label Smoothing for Machine Translation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss (Cross-Entropy with Label Smoothing)
    
    Instead of using hard targets (one-hot), we use soft targets where
    the correct class has probability (1 - smoothing) and the remaining
    probability is distributed among other classes.
    
    Args:
        vocab_size: Size of the vocabulary
        padding_idx: Index of padding token (will be ignored)
        smoothing: Label smoothing value (0.0 = no smoothing)
    """
    
    def __init__(
        self,
        vocab_size: int,
        padding_idx: int = 0,
        smoothing: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
        # KL Divergence loss
        self.criterion = nn.KLDivLoss(reduction='sum')
    
    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute label smoothing loss.
        
        Args:
            logits: Model output logits (batch * seq_len, vocab_size) or (batch, seq_len, vocab_size)
            target: Target indices (batch * seq_len,) or (batch, seq_len)
            
        Returns:
            Scalar loss value
        """
        # Flatten if needed
        if logits.dim() == 3:
            logits = logits.reshape(-1, logits.size(-1))
        if target.dim() == 2:
            target = target.reshape(-1)
        
        # Apply log softmax to get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Create mask for non-padding positions
        mask = (target != self.padding_idx)
        
        # More memory-efficient label smoothing
        # Instead of creating full distribution, compute loss directly
        n_tokens = mask.sum()
        
        if n_tokens == 0:
            return torch.tensor(0.0, device=logits.device)
        
        # Select log probs for true labels
        true_log_probs = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        
        # Smoothing loss: uniform distribution over all non-padding tokens
        smooth_loss = -log_probs.sum(dim=-1)  # Sum over vocab
        smooth_loss = smooth_loss + log_probs[:, self.padding_idx]  # Exclude padding
        
        # Combined loss
        loss = -self.confidence * true_log_probs - (self.smoothing / (self.vocab_size - 1)) * smooth_loss
        
        # Mask padding and average
        loss = (loss * mask).sum() / n_tokens
        
        return loss


class CrossEntropyLoss(nn.Module):
    """
    Standard Cross-Entropy Loss for sequence modeling.
    
    Args:
        padding_idx: Index of padding token (will be ignored)
    """
    
    def __init__(self, padding_idx: int = 0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=padding_idx,
            reduction='mean'
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.
        
        Args:
            logits: Model output (batch, seq_len, vocab_size)
            target: Target indices (batch, seq_len)
            
        Returns:
            Scalar loss value
        """
        # Reshape for cross-entropy
        # (batch, seq_len, vocab) -> (batch * seq_len, vocab)
        logits = logits.reshape(-1, logits.size(-1))
        target = target.reshape(-1)
        
        return self.criterion(logits, target)


def get_loss_function(
    loss_type: str,
    vocab_size: int,
    padding_idx: int = 0,
    smoothing: float = 0.1
) -> nn.Module:
    """
    Get loss function by name.
    
    Args:
        loss_type: 'cross_entropy' or 'label_smoothing'
        vocab_size: Vocabulary size
        padding_idx: Padding index
        smoothing: Label smoothing value
        
    Returns:
        Loss function module
    """
    if loss_type == 'cross_entropy':
        return CrossEntropyLoss(padding_idx)
    elif loss_type == 'label_smoothing':
        return LabelSmoothingLoss(vocab_size, padding_idx, smoothing)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    batch_size = 2
    seq_len = 10
    vocab_size = 1000
    
    # Random logits and targets
    logits = torch.randn(batch_size, seq_len, vocab_size)
    target = torch.randint(0, vocab_size, (batch_size, seq_len))
    target[0, -2:] = 0  # Add padding
    
    # Test CrossEntropyLoss
    ce_loss = CrossEntropyLoss(padding_idx=0)
    loss_ce = ce_loss(logits, target)
    print(f"Cross-Entropy Loss: {loss_ce.item():.4f}")
    
    # Test LabelSmoothingLoss
    ls_loss = LabelSmoothingLoss(vocab_size, padding_idx=0, smoothing=0.1)
    logits_flat = logits.view(-1, vocab_size)
    target_flat = target.view(-1)
    loss_ls = ls_loss(logits_flat, target_flat)
    print(f"Label Smoothing Loss: {loss_ls.item():.4f}")

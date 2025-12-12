import torch
import torch.nn as nn
import torch.nn.functional as F

    
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
            reduction="mean"
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
    
   

    
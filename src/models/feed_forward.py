"""
Feed-Forward Network
Implements the Position-wise Feed-Forward Network from the Transformer.

Reference: "Attention Is All You Need" (Vaswani et al., 2017)

FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
or with GELU: FFN(x) = GELU(xW_1 + b_1)W_2 + b_2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    Two linear transformations with a non-linear activation in between.
    
    Args:
        d_model: Model dimension (input and output dimension)
        d_ff: Inner layer dimension (typically 4 * d_model)
        dropout: Dropout probability
        activation: Activation function ('relu' or 'gelu')
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # First linear layer: d_model -> d_ff
        self.linear1 = nn.Linear(d_model, d_ff)
        
        # Second linear layer: d_ff -> d_model
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feed-forward network.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # First linear + activation
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Second linear
        x = self.linear2(x)
        x = self.dropout(x)
        
        return x


class GatedFeedForward(nn.Module):
    """
    Gated Feed-Forward Network (SwiGLU variant)
    
    Used in more recent transformer variants like LLaMA.
    
    FFN_SwiGLU(x) = (Swish(xW_1) ⊙ xV) W_2
    
    Args:
        d_model: Model dimension
        d_ff: Inner layer dimension
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # Gate
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated feed-forward network."""
        # SwiGLU: Swish(xW_1) ⊙ xW_3
        gate = F.silu(self.w1(x))  # Swish activation
        x = gate * self.w3(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x


if __name__ == "__main__":
    # Test feed-forward network
    batch_size = 2
    seq_len = 10
    d_model = 512
    d_ff = 2048
    
    # Create FFN
    ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff, dropout=0.1)
    
    # Create random input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Apply FFN
    output = ffn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in ffn.parameters())}")

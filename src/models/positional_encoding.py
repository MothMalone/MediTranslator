"""
Positional Encoding
Implements Sinusoidal Positional Encoding from the Transformer paper.

Reference: "Attention Is All You Need" (Vaswani et al., 2017)

PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding
    
    Adds positional information to token embeddings using sine and cosine
    functions of different frequencies.
    
    Args:
        d_model: Model dimension
        max_seq_length: Maximum sequence length
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        d_model: int,
        max_seq_length: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        
        # Position indices: (max_seq_length, 1)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        
        # Dimension indices for computing the division term
        # div_term = 10000^(2i/d_model) = exp(2i * -log(10000)/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: (1, max_seq_length, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but should be saved)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            Output tensor with positional encoding added (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        
        # Add positional encoding (broadcasting over batch dimension)
        x = x + self.pe[:, :seq_len, :]
        
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Encoding
    
    Alternative to sinusoidal encoding where positions are learned embeddings.
    
    Args:
        d_model: Model dimension
        max_seq_length: Maximum sequence length
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        d_model: int,
        max_seq_length: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(max_seq_length, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional encoding to input embeddings.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            Output tensor with positional encoding added
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        x = x + self.embedding(positions)
        
        return self.dropout(x)


if __name__ == "__main__":
    # Test positional encoding
    d_model = 512
    max_len = 100
    batch_size = 2
    seq_len = 20
    
    # Create positional encoding
    pe = PositionalEncoding(d_model=d_model, max_seq_length=max_len)
    
    # Create random input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Apply positional encoding
    output = pe(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Visualize positional encoding pattern
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 5))
    plt.imshow(pe.pe[0, :50, :].numpy(), aspect='auto', cmap='RdBu')
    plt.xlabel('Dimension')
    plt.ylabel('Position')
    plt.title('Sinusoidal Positional Encoding')
    plt.colorbar()
    plt.savefig('positional_encoding.png')
    print("Saved positional encoding visualization")

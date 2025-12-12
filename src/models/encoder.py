"""
Transformer Encoder
Implements the Encoder part of the Transformer architecture.

Reference: "Attention Is All You Need" (Vaswani et al., 2017)

Encoder Layer:
    1. Multi-Head Self-Attention
    2. Add & Layer Normalization
    3. Feed-Forward Network
    4. Add & Layer Normalization
"""
import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention
from .feed_forward import FeedForwardNetwork


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        activation: Activation function for FFN
        norm_first: If True, apply LayerNorm before attention/FFN (Pre-LN)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu',
        norm_first: bool = False
    ):
        super().__init__()
        
        self.norm_first = norm_first
        
        # Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Feed-Forward Network
        self.feed_forward = FeedForwardNetwork(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through encoder layer.
        
        Args:
            src: Source sequence (batch, seq_len, d_model)
            src_mask: Attention mask (optional)
            src_key_padding_mask: Padding mask (batch, seq_len), True = padded
            
        Returns:
            Encoded output (batch, seq_len, d_model)
        """
        # Prepare mask for multi-head attention
        mask = None
        if src_key_padding_mask is not None:
            # Expand mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)
        
        if self.norm_first:
            # Pre-LN: Norm -> Attention -> Add
            src2 = self.norm1(src)
            src2, _ = self.self_attention(src2, src2, src2, mask)
            src = src + self.dropout1(src2)
            
            # Pre-LN: Norm -> FFN -> Add
            src2 = self.norm2(src)
            src2 = self.feed_forward(src2)
            src = src + self.dropout2(src2)
        else:
            # Post-LN (original): Attention -> Add -> Norm
            src2, _ = self.self_attention(src, src, src, mask)
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            
            # Post-LN: FFN -> Add -> Norm
            src2 = self.feed_forward(src)
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        
        return src


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder (stack of encoder layers)
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of encoder layers
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        activation: Activation function
        norm_first: Pre-LN or Post-LN
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu',
        norm_first: bool = False
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first
            )
            for _ in range(n_layers)
        ])
        
        # Final layer normalization (for Pre-LN variant)
        self.norm = nn.LayerNorm(d_model) if norm_first else None
    
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through all encoder layers.
        
        Args:
            src: Source sequence (batch, seq_len, d_model)
            src_mask: Attention mask
            src_key_padding_mask: Padding mask (batch, seq_len)
            
        Returns:
            Encoded output (batch, seq_len, d_model)
        """
        output = src
        
        for layer in self.layers:
            output = layer(output, src_mask, src_key_padding_mask)
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output


if __name__ == "__main__":
    # Test encoder
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    n_layers = 6
    d_ff = 2048
    
    # Create encoder
    encoder = TransformerEncoder(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=0.1
    )
    
    # Create random input
    src = torch.randn(batch_size, seq_len, d_model)
    
    # Create padding mask (last 2 positions are padded)
    src_key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    src_key_padding_mask[:, -2:] = True
    
    # Forward pass
    output = encoder(src, src_key_padding_mask=src_key_padding_mask)
    
    print(f"Input shape: {src.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in encoder.parameters()):,}")

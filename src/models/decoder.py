"""
Transformer Decoder
Implements the Decoder part of the Transformer architecture.

Reference: "Attention Is All You Need" (Vaswani et al., 2017)

Decoder Layer:
    1. Masked Multi-Head Self-Attention
    2. Add & Layer Normalization
    3. Multi-Head Cross-Attention (Encoder-Decoder Attention)
    4. Add & Layer Normalization
    5. Feed-Forward Network
    6. Add & Layer Normalization
"""
import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention
from .feed_forward import FeedForwardNetwork


class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer
    
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
        
        # Masked Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Multi-Head Cross-Attention (Encoder-Decoder Attention)
        self.cross_attention = MultiHeadAttention(
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
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through decoder layer.
        
        Args:
            tgt: Target sequence (batch, tgt_len, d_model)
            memory: Encoder output (batch, src_len, d_model)
            tgt_mask: Causal mask for self-attention (tgt_len, tgt_len)
            memory_mask: Mask for cross-attention (optional)
            tgt_key_padding_mask: Target padding mask (batch, tgt_len)
            memory_key_padding_mask: Source padding mask (batch, src_len)
            
        Returns:
            Decoded output (batch, tgt_len, d_model)
        """
        # Prepare masks
        self_attn_mask = self._prepare_self_attention_mask(tgt, tgt_mask, tgt_key_padding_mask)
        cross_attn_mask = self._prepare_cross_attention_mask(memory_key_padding_mask)
        
        if self.norm_first:
            # Pre-LN variant
            # Masked Self-Attention
            tgt2 = self.norm1(tgt)
            tgt2, _ = self.self_attention(tgt2, tgt2, tgt2, self_attn_mask)
            tgt = tgt + self.dropout1(tgt2)
            
            # Cross-Attention
            tgt2 = self.norm2(tgt)
            tgt2, _ = self.cross_attention(tgt2, memory, memory, cross_attn_mask)
            tgt = tgt + self.dropout2(tgt2)
            
            # Feed-Forward
            tgt2 = self.norm3(tgt)
            tgt2 = self.feed_forward(tgt2)
            tgt = tgt + self.dropout3(tgt2)
        else:
            # Post-LN (original) variant
            # Masked Self-Attention
            tgt2, _ = self.self_attention(tgt, tgt, tgt, self_attn_mask)
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            
            # Cross-Attention
            tgt2, _ = self.cross_attention(tgt, memory, memory, cross_attn_mask)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
            
            # Feed-Forward
            tgt2 = self.feed_forward(tgt)
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)
        
        return tgt
    
    def _prepare_self_attention_mask(
        self,
        tgt: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
        tgt_key_padding_mask: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Prepare combined mask for self-attention."""
        batch_size, tgt_len = tgt.size(0), tgt.size(1)
        device = tgt.device
        
        # Create causal mask if not provided
        if tgt_mask is None:
            tgt_mask = torch.triu(
                torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool),
                diagonal=1
            )
        
        # Expand to (batch, 1, tgt_len, tgt_len)
        mask = tgt_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, tgt_len, tgt_len)
        
        # Combine with padding mask
        if tgt_key_padding_mask is not None:
            # (batch, tgt_len) -> (batch, 1, 1, tgt_len)
            padding_mask = tgt_key_padding_mask.unsqueeze(1).unsqueeze(2)
            mask = mask | padding_mask
        
        return mask
    
    def _prepare_cross_attention_mask(
        self,
        memory_key_padding_mask: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Prepare mask for cross-attention."""
        if memory_key_padding_mask is None:
            return None
        
        # (batch, src_len) -> (batch, 1, 1, src_len)
        return memory_key_padding_mask.unsqueeze(1).unsqueeze(2)


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder (stack of decoder layers)
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of decoder layers
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
            TransformerDecoderLayer(
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
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through all decoder layers.
        
        Args:
            tgt: Target sequence (batch, tgt_len, d_model)
            memory: Encoder output (batch, src_len, d_model)
            tgt_mask: Causal mask for self-attention
            memory_mask: Mask for cross-attention
            tgt_key_padding_mask: Target padding mask
            memory_key_padding_mask: Source padding mask
            
        Returns:
            Decoded output (batch, tgt_len, d_model)
        """
        output = tgt
        
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask
            )
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output


if __name__ == "__main__":
    # Test decoder
    batch_size = 2
    src_len = 10
    tgt_len = 8
    d_model = 512
    n_heads = 8
    n_layers = 6
    d_ff = 2048
    
    # Create decoder
    decoder = TransformerDecoder(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=0.1
    )
    
    # Create random inputs
    tgt = torch.randn(batch_size, tgt_len, d_model)
    memory = torch.randn(batch_size, src_len, d_model)  # Encoder output
    
    # Forward pass
    output = decoder(tgt, memory)
    
    print(f"Target shape: {tgt.shape}")
    print(f"Memory shape: {memory.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in decoder.parameters()):,}")

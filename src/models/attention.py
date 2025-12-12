"""
Attention Mechanisms
Implements Scaled Dot-Product Attention and Multi-Head Attention from scratch.

Reference: "Attention Is All You Need" (Vaswani et al., 2017)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    
    Args:
        dropout: Dropout probability for attention weights
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout == nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Args:
            query: Query tensor (batch, heads, seq_len, d_k)
            key: Key tensor (batch, heads, seq_len, d_k)
            value: Value tensor (batch, heads, seq_len, d_v)
            mask: Optional mask tensor (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
                  True values are masked (not attended to)
        
        Returns:
            output: Attention output (batch, heads, seq_len, d_v)
            attention_weights: Attention weights (batch, heads, seq_len, seq_len)
        """

        d_k = query.size(-1)

        # scores = Q @ K^T / sqrt(d_k)
        # Shape: (batch, heads, seq_len_q, seq_len_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Case where all values are masked
        attention_weights = attention_weights.masked_fill(
            torch.isnan(attention_weights), 0.0
        )

        # Apply dropout 
        attention_weights = self.dropout(attention_weights)

        # Compute output as weighted sum of values
        # output = attention_weights @ value
        output = torch.matmul(attention_weights, value)

        return output, attention_weights
    
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
    where head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0
    ):
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        self.d_v = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        
        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention(dropout)
        
        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform."""
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.xavier_uniform_(self.W_O.weight)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head attention.
        
        Args:
            query: Query tensor (batch, seq_len_q, d_model)
            key: Key tensor (batch, seq_len_k, d_model)
            value: Value tensor (batch, seq_len_v, d_model)
            mask: Optional attention mask
        
        Returns:
            output: Attention output (batch, seq_len_q, d_model)
            attention_weights: Attention weights (batch, n_heads, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)

        # Linear projections
        # Shape: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        Q = self.W_Q(query)
        K = self.W_K(key)
        V = self.W_V(value)

        # Reshape to (batch, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        
        # Apply scaled dot-product attention
        attn_output, attention_weights = self.attention(Q, K, V, mask)

        # Concatenate heads
        # (batch, n_heads, seq_len, d_v) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_O(attn_output)
        
        return output, attention_weights
    
if __name__ == "__main__":
    # Test the attention modules
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    
    # Create random input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test Multi-Head Attention
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.1)
    output, attn_weights = mha(x, x, x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")

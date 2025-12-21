"""
Full Transformer Model for Machine Translation
Combines Encoder and Decoder with Embeddings and Output Projection.

Reference: "Attention Is All You Need" (Vaswani et al., 2017)
"""
import torch
import torch.nn as nn
import math
from typing import Optional

from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from .positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    """
    Full Transformer Model for Sequence-to-Sequence Translation
    
    Components:
        - Source Embedding + Positional Encoding
        - Target Embedding + Positional Encoding
        - Transformer Encoder
        - Transformer Decoder
        - Output Linear Projection
    
    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        d_model: Model dimension
        n_heads: Number of attention heads
        n_encoder_layers: Number of encoder layers
        n_decoder_layers: Number of decoder layers
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        max_seq_length: Maximum sequence length
        share_embeddings: Whether to share src/tgt embeddings
        activation: Activation function ('relu' or 'gelu')
        norm_first: Pre-LN or Post-LN
        pad_idx: Padding token index
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 512,
        share_embeddings: bool = False,
        activation: str = 'relu',
        norm_first: bool = False,
        pad_idx: int = 0,
        use_xavier_init: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.use_xavier_init = use_xavier_init
        
        # Source Embedding
        self.src_embedding = nn.Embedding(
            src_vocab_size, d_model, padding_idx=pad_idx
        )
        
        # Target Embedding (optionally shared with source)
        if share_embeddings and src_vocab_size == tgt_vocab_size:
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = nn.Embedding(
                tgt_vocab_size, d_model, padding_idx=pad_idx
            )
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        
        # Transformer Encoder
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first
        )
        
        # Transformer Decoder
        self.decoder = TransformerDecoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_decoder_layers,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first
        )
        
        # Output Projection (to vocabulary)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Optionally tie output projection weights with target embedding
        # This is a common technique to improve performance
        self.output_projection.weight = self.tgt_embedding.weight
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        if self.use_xavier_init:
            # Xavier initialization for better gradient flow
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        
        # Scale embeddings
        nn.init.normal_(self.src_embedding.weight, mean=0, std=self.d_model ** -0.5)
        if self.tgt_embedding is not self.src_embedding:
            nn.init.normal_(self.tgt_embedding.weight, mean=0, std=self.d_model ** -0.5)
        
        # Zero out padding embedding
        self.src_embedding.weight.data[self.pad_idx].zero_()
        if self.tgt_embedding is not self.src_embedding:
            self.tgt_embedding.weight.data[self.pad_idx].zero_()
    
    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode source sequence.
        
        Args:
            src: Source token indices (batch, src_len)
            src_mask: Source attention mask
            src_key_padding_mask: Source padding mask (batch, src_len)
            
        Returns:
            Encoder output (batch, src_len, d_model)
        """
        # Create padding mask if not provided
        if src_key_padding_mask is None:
            src_key_padding_mask = (src == self.pad_idx)
        
        # Embed and scale
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        
        # Add positional encoding
        src_embedded = self.positional_encoding(src_embedded)
        
        # Encode
        memory = self.encoder(
            src_embedded,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        
        return memory
    
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode target sequence.
        
        Args:
            tgt: Target token indices (batch, tgt_len)
            memory: Encoder output (batch, src_len, d_model)
            tgt_mask: Target causal mask
            memory_mask: Cross-attention mask
            tgt_key_padding_mask: Target padding mask
            memory_key_padding_mask: Source padding mask
            
        Returns:
            Decoder output logits (batch, tgt_len, tgt_vocab_size)
        """
        # Create padding mask if not provided
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = (tgt == self.pad_idx)
        
        # Embed and scale
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        
        # Add positional encoding
        tgt_embedded = self.positional_encoding(tgt_embedded)
        
        # Decode
        output = self.decoder(
            tgt_embedded,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        return logits
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Full forward pass through encoder and decoder.
        
        Args:
            src: Source token indices (batch, src_len)
            tgt: Target token indices (batch, tgt_len)
            src_mask: Source attention mask
            tgt_mask: Target causal mask
            memory_mask: Cross-attention mask
            src_key_padding_mask: Source padding mask
            tgt_key_padding_mask: Target padding mask
            memory_key_padding_mask: Source padding mask for decoder
            
        Returns:
            Output logits (batch, tgt_len, tgt_vocab_size)
        """
        # Create padding masks if not provided
        if src_key_padding_mask is None:
            src_key_padding_mask = (src == self.pad_idx)
        if memory_key_padding_mask is None:
            memory_key_padding_mask = src_key_padding_mask
        
        # Encode
        memory = self.encode(src, src_mask, src_key_padding_mask)
        
        # Decode
        output = self.decode(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask
        )
        
        return output
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config: dict) -> Transformer:
    """
    Create a Transformer model from config dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Transformer model
    """
    return Transformer(
        src_vocab_size=config['vocab']['src_vocab_size'],
        tgt_vocab_size=config['vocab']['tgt_vocab_size'],
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_encoder_layers=config['model']['n_encoder_layers'],
        n_decoder_layers=config['model']['n_decoder_layers'],
        d_ff=config['model']['d_ff'],
        dropout=config['model']['dropout'],
        max_seq_length=config['model']['max_seq_length'],
        pad_idx=0
    )


if __name__ == "__main__":
    # Test the full Transformer
    batch_size = 2
    src_len = 10
    tgt_len = 8
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    
    # Create model
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=512,
        n_heads=8,
        n_encoder_layers=6,
        n_decoder_layers=6,
        d_ff=2048,
        dropout=0.1
    )
    
    # Create random inputs
    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))
    
    # Add padding
    src[0, -2:] = 0
    tgt[0, -3:] = 0
    
    # Forward pass
    output = model(src, tgt)
    
    print(f"Source shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {model.count_parameters():,}")

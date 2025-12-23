"""
Model Embedding Resize Utilities
Utilities for resizing model embeddings when vocabulary is expanded.
"""
import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def resize_token_embeddings(
    model: nn.Module,
    new_vocab_size: int,
    old_vocab_size: Optional[int] = None,
    pad_idx: int = 0
) -> nn.Module:
    """
    Resize model token embeddings to accommodate new vocabulary.
    
    When vocabulary is expanded:
    1. Create new larger embedding layers
    2. Copy old embeddings
    3. Initialize new embeddings (random or mean of existing)
    
    Args:
        model: Transformer model
        new_vocab_size: New vocabulary size
        old_vocab_size: Old vocabulary size (auto-detect if None)
        pad_idx: Padding index
        
    Returns:
        Model with resized embeddings
    """
    # Get current embedding layers
    src_embedding = model.src_embedding.embedding
    tgt_embedding = model.tgt_embedding.embedding
    output_projection = model.output_projection
    
    old_src_size = src_embedding.num_embeddings
    old_tgt_size = tgt_embedding.num_embeddings
    
    if old_vocab_size is None:
        old_vocab_size = old_src_size
    
    # Check if resize is needed
    if old_src_size == new_vocab_size and old_tgt_size == new_vocab_size:
        logger.info("Vocabulary size unchanged, no resize needed")
        return model
    
    logger.info(f"Resizing embeddings:")
    logger.info(f"  Source: {old_src_size} → {new_vocab_size}")
    logger.info(f"  Target: {old_tgt_size} → {new_vocab_size}")
    
    # Resize source embedding
    if old_src_size != new_vocab_size:
        new_src_embedding = resize_embedding_layer(
            src_embedding, new_vocab_size, pad_idx
        )
        model.src_embedding.embedding = new_src_embedding
    
    # Resize target embedding
    if old_tgt_size != new_vocab_size:
        new_tgt_embedding = resize_embedding_layer(
            tgt_embedding, new_vocab_size, pad_idx
        )
        model.tgt_embedding.embedding = new_tgt_embedding
    
    # Resize output projection (tied with target embedding in some models)
    if output_projection.out_features != new_vocab_size:
        new_output_projection = resize_linear_layer(
            output_projection, new_vocab_size
        )
        model.output_projection = new_output_projection
    
    logger.info("✓ Embeddings resized successfully")
    
    return model


def resize_embedding_layer(
    embedding: nn.Embedding,
    new_size: int,
    pad_idx: int = 0
) -> nn.Embedding:
    """
    Resize an embedding layer.
    
    Args:
        embedding: Original embedding layer
        new_size: New vocabulary size
        pad_idx: Padding index
        
    Returns:
        New embedding layer with copied + initialized weights
    """
    old_size = embedding.num_embeddings
    embed_dim = embedding.embedding_dim
    
    # Create new embedding
    new_embedding = nn.Embedding(
        new_size,
        embed_dim,
        padding_idx=pad_idx
    )
    
    # Copy old embeddings
    with torch.no_grad():
        # Copy existing embeddings
        new_embedding.weight[:old_size] = embedding.weight[:old_size]
        
        # Initialize new embeddings
        if new_size > old_size:
            # Use mean of existing embeddings as initialization
            mean_embedding = embedding.weight[:old_size].mean(dim=0)
            std_embedding = embedding.weight[:old_size].std(dim=0).mean()
            
            # Initialize new tokens with slight variation around mean
            for i in range(old_size, new_size):
                new_embedding.weight[i] = mean_embedding + torch.randn(embed_dim) * std_embedding * 0.1
        
        # Ensure padding embedding is zeros
        if pad_idx < new_size:
            new_embedding.weight[pad_idx].fill_(0)
    
    return new_embedding


def resize_linear_layer(
    linear: nn.Linear,
    new_out_features: int
) -> nn.Linear:
    """
    Resize output projection linear layer.
    
    Args:
        linear: Original linear layer
        new_out_features: New output size
        
    Returns:
        New linear layer with copied + initialized weights
    """
    old_out = linear.out_features
    in_features = linear.in_features
    has_bias = linear.bias is not None
    
    # Create new linear layer
    new_linear = nn.Linear(in_features, new_out_features, bias=has_bias)
    
    # Copy old weights
    with torch.no_grad():
        new_linear.weight[:old_out] = linear.weight[:old_out]
        
        if has_bias:
            new_linear.bias[:old_out] = linear.bias[:old_out]
        
        # Initialize new weights
        if new_out_features > old_out:
            # Use Xavier initialization for new weights
            nn.init.xavier_uniform_(new_linear.weight[old_out:])
            if has_bias:
                new_linear.bias[old_out:].fill_(0)
    
    return new_linear


def get_embedding_statistics(model: nn.Module) -> dict:
    """
    Get statistics about model embeddings.
    
    Args:
        model: Transformer model
        
    Returns:
        Dictionary with embedding statistics
    """
    stats = {
        'src_vocab_size': model.src_embedding.num_embeddings,
        'tgt_vocab_size': model.tgt_embedding.num_embeddings,
        'embed_dim': model.src_embedding.embedding_dim,
        'output_vocab_size': model.output_projection.out_features,
    }
    
    # Get embedding layers directly
    src_emb = model.src_embedding
    tgt_emb = model.tgt_embedding
    
    stats['src_params'] = src_emb.weight.numel()
    stats['tgt_params'] = tgt_emb.weight.numel()
    stats['total_embedding_params'] = stats['src_params'] + stats['tgt_params']
    
    return stats


def print_embedding_info(model: nn.Module):
    """Print embedding information."""
    stats = get_embedding_statistics(model)
    
    print("Embedding Information:")
    print("-" * 50)
    print(f"Source vocabulary size: {stats['src_vocab_size']:,}")
    print(f"Target vocabulary size: {stats['tgt_vocab_size']:,}")
    print(f"Embedding dimension: {stats['embed_dim']}")
    print(f"Output projection size: {stats['output_vocab_size']:,}")
    print(f"Total embedding parameters: {stats['total_embedding_params']:,}")
    print("-" * 50)

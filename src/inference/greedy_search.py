"""
Greedy Search Decoding
Simple decoding strategy that selects the most probable token at each step.
"""
import torch
import torch.nn as nn
from typing import Optional


@torch.no_grad()
def greedy_decode(
    model: nn.Module,
    src: torch.Tensor,
    max_length: int = 128,
    bos_idx: int = 2,
    eos_idx: int = 3,
    pad_idx: int = 0
) -> torch.Tensor:
    """
    Greedy decoding for sequence generation.
    
    At each step, select the token with highest probability.
    
    Args:
        model: Transformer model
        src: Source sequence (batch, src_len)
        max_length: Maximum output length
        bos_idx: Beginning of sequence token index
        eos_idx: End of sequence token index
        pad_idx: Padding token index
        
    Returns:
        Generated sequences (batch, output_len)
    """
    model.eval()
    device = src.device
    batch_size = src.size(0)
    
    # Encode source
    memory = model.encode(src)
    
    # Initialize decoder input with BOS token
    decoder_input = torch.full(
        (batch_size, 1),
        bos_idx,
        dtype=torch.long,
        device=device
    )
    
    # Track which sequences are finished
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    # Generate tokens one by one
    for step in range(max_length):
        # Decode
        logits = model.decode(decoder_input, memory)
        
        # Get last token predictions
        next_token_logits = logits[:, -1, :]
        
        # Greedy selection: pick token with highest probability
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        
        # For finished sequences, output padding
        next_token = next_token.masked_fill(finished.unsqueeze(-1), pad_idx)
        
        # Append to decoder input
        decoder_input = torch.cat([decoder_input, next_token], dim=1)
        
        # Check for EOS
        finished = finished | (next_token.squeeze(-1) == eos_idx)
        
        # Stop if all sequences are finished
        if finished.all():
            break
    
    return decoder_input


@torch.no_grad()
def greedy_decode_batch(
    model: nn.Module,
    src_batch: torch.Tensor,
    src_lengths: torch.Tensor,
    max_length: int = 128,
    bos_idx: int = 2,
    eos_idx: int = 3,
    pad_idx: int = 0
) -> torch.Tensor:
    """
    Greedy decoding with variable length source sequences.
    
    Args:
        model: Transformer model
        src_batch: Source sequences (batch, max_src_len)
        src_lengths: Actual lengths of source sequences (batch,)
        max_length: Maximum output length
        bos_idx: BOS token index
        eos_idx: EOS token index
        pad_idx: Padding token index
        
    Returns:
        Generated sequences (batch, output_len)
    """
    model.eval()
    device = src_batch.device
    batch_size = src_batch.size(0)
    
    # Create source padding mask
    src_key_padding_mask = (src_batch == pad_idx)
    
    # Encode source
    memory = model.encode(src_batch, src_key_padding_mask=src_key_padding_mask)
    
    # Initialize
    decoder_input = torch.full(
        (batch_size, 1),
        bos_idx,
        dtype=torch.long,
        device=device
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    for step in range(max_length):
        logits = model.decode(
            decoder_input,
            memory,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        next_token_logits = logits[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        next_token = next_token.masked_fill(finished.unsqueeze(-1), pad_idx)
        
        decoder_input = torch.cat([decoder_input, next_token], dim=1)
        finished = finished | (next_token.squeeze(-1) == eos_idx)
        
        if finished.all():
            break
    
    return decoder_input


if __name__ == "__main__":
    print("Greedy search module loaded successfully")

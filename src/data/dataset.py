"""
Dataset Module
PyTorch Dataset and DataLoader for machine translation.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class TranslationDataset(Dataset):
    """
    PyTorch Dataset for parallel translation data.
    """
    
    def __init__(
        self,
        src_file: str,
        tgt_file: str,
        src_vocab,
        tgt_vocab,
        tokenize_fn: Optional[Callable] = None,
        max_length: int = 512,
        add_bos_eos: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            src_file: Path to source language file
            tgt_file: Path to target language file
            src_vocab: Source vocabulary
            tgt_vocab: Target vocabulary
            tokenize_fn: Tokenization function
            max_length: Maximum sequence length
            add_bos_eos: Whether to add BOS/EOS tokens
        """
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.tokenize_fn = tokenize_fn or (lambda x: x.strip().split())
        self.max_length = max_length
        self.add_bos_eos = add_bos_eos
        
        # Load data
        self.src_data = []
        self.tgt_data = []
        
        self._load_data(src_file, tgt_file)
    
    def _load_data(self, src_file: str, tgt_file: str):
        """Load and process parallel data."""
        with open(src_file, 'r', encoding='utf-8') as f_src, \
             open(tgt_file, 'r', encoding='utf-8') as f_tgt:
            
            for src_line, tgt_line in zip(f_src, f_tgt):
                # Tokenize
                src_tokens = self.tokenize_fn(src_line)
                tgt_tokens = self.tokenize_fn(tgt_line)
                
                # Truncate if necessary
                if len(src_tokens) > self.max_length - 2:  # Account for BOS/EOS
                    src_tokens = src_tokens[:self.max_length - 2]
                if len(tgt_tokens) > self.max_length - 2:
                    tgt_tokens = tgt_tokens[:self.max_length - 2]
                
                # Convert to indices
                src_indices = self.src_vocab.encode(src_tokens)
                tgt_indices = self.tgt_vocab.encode(tgt_tokens)
                
                # Add BOS/EOS tokens
                if self.add_bos_eos:
                    src_indices = [self.src_vocab.bos_idx] + src_indices + [self.src_vocab.eos_idx]
                    tgt_indices = [self.tgt_vocab.bos_idx] + tgt_indices + [self.tgt_vocab.eos_idx]
                
                self.src_data.append(torch.tensor(src_indices, dtype=torch.long))
                self.tgt_data.append(torch.tensor(tgt_indices, dtype=torch.long))
        
        logger.info(f"Loaded {len(self.src_data)} sentence pairs")
    
    def __len__(self) -> int:
        return len(self.src_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.src_data[idx], self.tgt_data[idx]


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_idx: int = 0):
    """
    Collate function for batching with padding.
    
    Args:
        batch: List of (src, tgt) tensor pairs
        pad_idx: Padding index
        
    Returns:
        Tuple of (padded_src, padded_tgt, src_lengths, tgt_lengths)
    """
    src_batch, tgt_batch = zip(*batch)
    
    # Get lengths before padding
    src_lengths = torch.tensor([len(s) for s in src_batch])
    tgt_lengths = torch.tensor([len(t) for t in tgt_batch])
    
    # Pad sequences
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    
    return src_padded, tgt_padded, src_lengths, tgt_lengths


def create_dataloader(
    dataset: TranslationDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pad_idx: int = 0
) -> DataLoader:
    """
    Create a DataLoader for the translation dataset.
    
    Args:
        dataset: TranslationDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pad_idx: Padding index
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, pad_idx),
        pin_memory=True
    )


def create_masks(src: torch.Tensor, tgt: torch.Tensor, pad_idx: int = 0):
    """
    Create attention masks for transformer.
    
    Args:
        src: Source tensor (batch_size, src_len)
        tgt: Target tensor (batch_size, tgt_len)
        pad_idx: Padding index
        
    Returns:
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
    """
    # Padding masks (True where padded)
    src_padding_mask = (src == pad_idx)
    tgt_padding_mask = (tgt == pad_idx)
    
    # Causal mask for decoder (True where should be masked)
    tgt_len = tgt.size(1)
    tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
    tgt_mask = tgt_mask.to(tgt.device)
    
    # Source mask (None for encoder self-attention)
    src_mask = None
    
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


if __name__ == "__main__":
    # This is a placeholder for testing
    print("Dataset module loaded successfully")

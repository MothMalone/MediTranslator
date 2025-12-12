"""
Vocabulary Module
Handles vocabulary building and token-to-index mapping.
"""
from typing import List, Dict, Optional, Iterable
from collections import Counter
import json
import logging

logger = logging.getLogger(__name__)


class Vocabulary:
    """
    Vocabulary class for managing token-to-index mappings.
    """
    
    # Special tokens
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    
    def __init__(
        self,
        min_freq: int = 1,
        max_size: Optional[int] = None,
        special_tokens: Optional[List[str]] = None
    ):
        """
        Initialize vocabulary.
        
        Args:
            min_freq: Minimum frequency for a token to be included
            max_size: Maximum vocabulary size (None for unlimited)
            special_tokens: List of special tokens to add
        """
        self.min_freq = min_freq
        self.max_size = max_size
        
        # Initialize special tokens
        if special_tokens is None:
            special_tokens = [
                self.PAD_TOKEN,
                self.UNK_TOKEN,
                self.BOS_TOKEN,
                self.EOS_TOKEN
            ]
        self.special_tokens = special_tokens
        
        # Token to index and index to token mappings
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        
        # Add special tokens
        for token in self.special_tokens:
            self._add_token(token)
        
        # Store special token indices
        self.pad_idx = self.token2idx.get(self.PAD_TOKEN, 0)
        self.unk_idx = self.token2idx.get(self.UNK_TOKEN, 1)
        self.bos_idx = self.token2idx.get(self.BOS_TOKEN, 2)
        self.eos_idx = self.token2idx.get(self.EOS_TOKEN, 3)
    
    def _add_token(self, token: str) -> int:
        """Add a token to the vocabulary."""
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
        return self.token2idx[token]
    
    def build_from_iterator(self, iterator: Iterable[List[str]]):
        """
        Build vocabulary from an iterator of tokenized sentences.
        
        Args:
            iterator: Iterator yielding lists of tokens
        """
        counter = Counter()
        
        for tokens in iterator:
            counter.update(tokens)
        
        # Filter by frequency and sort
        tokens_with_freq = [
            (token, freq) for token, freq in counter.items()
            if freq >= self.min_freq
        ]
        tokens_with_freq.sort(key=lambda x: (-x[1], x[0]))
        
        # Limit size if specified
        if self.max_size is not None:
            max_tokens = self.max_size - len(self.special_tokens)
            tokens_with_freq = tokens_with_freq[:max_tokens]
        
        # Add tokens to vocabulary
        for token, freq in tokens_with_freq:
            self._add_token(token)
        
        logger.info(f"Built vocabulary with {len(self)} tokens")
    
    def build_from_file(self, file_path: str, tokenize_fn=None):
        """
        Build vocabulary from a text file.
        
        Args:
            file_path: Path to text file
            tokenize_fn: Tokenization function (default: split by whitespace)
        """
        if tokenize_fn is None:
            tokenize_fn = lambda x: x.strip().split()
        
        def file_iterator():
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    yield tokenize_fn(line)
        
        self.build_from_iterator(file_iterator())
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.token2idx)
    
    def __contains__(self, token: str) -> bool:
        """Check if token is in vocabulary."""
        return token in self.token2idx
    
    def encode(self, tokens: List[str]) -> List[int]:
        """
        Convert tokens to indices.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of indices
        """
        return [self.token2idx.get(token, self.unk_idx) for token in tokens]
    
    def decode(self, indices: List[int], skip_special: bool = True) -> List[str]:
        """
        Convert indices to tokens.
        
        Args:
            indices: List of indices
            skip_special: Whether to skip special tokens
            
        Returns:
            List of tokens
        """
        tokens = []
        for idx in indices:
            token = self.idx2token.get(idx, self.UNK_TOKEN)
            if skip_special and token in self.special_tokens:
                continue
            tokens.append(token)
        return tokens
    
    def save(self, file_path: str):
        """
        Save vocabulary to file.
        
        Args:
            file_path: Output file path
        """
        data = {
            'token2idx': self.token2idx,
            'special_tokens': self.special_tokens,
            'min_freq': self.min_freq,
            'max_size': self.max_size
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved vocabulary to {file_path}")
    
    @classmethod
    def load(cls, file_path: str) -> 'Vocabulary':
        """
        Load vocabulary from file.
        
        Args:
            file_path: Input file path
            
        Returns:
            Vocabulary instance
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vocab = cls(
            min_freq=data.get('min_freq', 1),
            max_size=data.get('max_size'),
            special_tokens=data.get('special_tokens')
        )
        
        # Clear and rebuild mappings
        vocab.token2idx = data['token2idx']
        vocab.idx2token = {int(v): k for k, v in vocab.token2idx.items()}
        
        logger.info(f"Loaded vocabulary with {len(vocab)} tokens from {file_path}")
        return vocab


if __name__ == "__main__":
    # Test vocabulary
    vocab = Vocabulary(min_freq=1)
    
    # Build from sample data
    sample_data = [
        ["hello", "world"],
        ["hello", "there"],
        ["world", "is", "beautiful"]
    ]
    vocab.build_from_iterator(sample_data)
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Token to index: {vocab.token2idx}")
    
    # Test encode/decode
    tokens = ["hello", "world", "unknown"]
    indices = vocab.encode(tokens)
    print(f"Encoded: {indices}")
    print(f"Decoded: {vocab.decode(indices)}")

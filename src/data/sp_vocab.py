"""
SentencePiece Vocabulary Wrapper
Provides a unified interface for SentencePiece models compatible with the Vocabulary class.
"""
import os
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    import sentencepiece as spm
    SPM_AVAILABLE = True
except ImportError:
    SPM_AVAILABLE = False
    logger.warning("sentencepiece not available. Install with: pip install sentencepiece")


class SentencePieceVocab:
    """
    SentencePiece vocabulary wrapper that provides the same interface as Vocabulary class.
    
    This ensures consistent special token handling and encoding/decoding.
    """
    
    def __init__(self, model_path: str):
        """
        Load a SentencePiece model.
        
        Args:
            model_path: Path to .model file
        """
        if not SPM_AVAILABLE:
            raise ImportError("sentencepiece is required. Install with: pip install sentencepiece")
        
        self.model_path = model_path
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        
        # Get special token indices (set during training)
        self.pad_idx = self.sp.pad_id()  # Should be 0
        self.unk_idx = self.sp.unk_id()  # Should be 1
        self.bos_idx = self.sp.bos_id()  # Should be 2
        self.eos_idx = self.sp.eos_id()  # Should be 3
        
        logger.info(f"Loaded SentencePiece model: {model_path}")
        logger.info(f"  Vocab size: {len(self)}")
        logger.info(f"  Special tokens: PAD={self.pad_idx}, UNK={self.unk_idx}, BOS={self.bos_idx}, EOS={self.eos_idx}")
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.sp.GetPieceSize()
    
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            add_bos: Add BOS token at start
            add_eos: Add EOS token at end
            
        Returns:
            List of token IDs
        """
        ids = self.sp.EncodeAsIds(text)
        
        if add_bos:
            ids = [self.bos_idx] + ids
        if add_eos:
            ids = ids + [self.eos_idx]
        
        return ids
    
    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
            skip_special: Whether to skip special tokens
            
        Returns:
            Decoded text string
        """
        if skip_special:
            # Filter out special tokens
            special_ids = {self.pad_idx, self.unk_idx, self.bos_idx, self.eos_idx}
            ids = [i for i in ids if i not in special_ids]
        
        return self.sp.DecodeIds(ids)
    
    def encode_as_pieces(self, text: str) -> List[str]:
        """
        Encode text to subword pieces.
        
        Args:
            text: Input text string
            
        Returns:
            List of subword pieces
        """
        return self.sp.EncodeAsPieces(text)
    
    def decode_pieces(self, pieces: List[str]) -> str:
        """
        Decode subword pieces to text.
        
        Args:
            pieces: List of subword pieces
            
        Returns:
            Decoded text string
        """
        return self.sp.DecodePieces(pieces)
    
    def id_to_piece(self, idx: int) -> str:
        """Get piece for an ID."""
        return self.sp.IdToPiece(idx)
    
    def piece_to_id(self, piece: str) -> int:
        """Get ID for a piece."""
        return self.sp.PieceToId(piece)
    
    @classmethod
    def load(cls, model_path: str) -> 'SentencePieceVocab':
        """Load a SentencePiece model (for compatibility with Vocabulary.load())."""
        return cls(model_path)


def load_tokenizer_and_vocab(vocab_dir: str, side: str = "src"):
    """
    Load SentencePiece tokenizer and vocab for a given side (src/tgt).
    Falls back to word-level Vocabulary if SentencePiece not available.
    
    Args:
        vocab_dir: Directory containing vocab files
        side: "src" or "tgt"
        
    Returns:
        Tuple of (vocab, tokenizer) where tokenizer may be None for word-level
    """
    from src.data.vocabulary import Vocabulary
    
    sp_model_path = os.path.join(vocab_dir, f"{side}.model")
    json_vocab_path = os.path.join(vocab_dir, f"{side}_vocab.json")
    
    # Try SentencePiece first
    if os.path.exists(sp_model_path):
        logger.info(f"Loading SentencePiece vocab: {sp_model_path}")
        vocab = SentencePieceVocab(sp_model_path)
        return vocab, vocab  # SentencePieceVocab acts as both vocab and tokenizer
    
    # Fall back to word-level
    elif os.path.exists(json_vocab_path):
        logger.info(f"Loading word-level vocab: {json_vocab_path}")
        vocab = Vocabulary.load(json_vocab_path)
        return vocab, None  # No tokenizer for word-level
    
    else:
        raise FileNotFoundError(f"No vocabulary found in {vocab_dir} for {side}")

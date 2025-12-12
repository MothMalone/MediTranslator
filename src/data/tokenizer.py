"""
Tokenizer Module
Handles text tokenization using various methods.
"""
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class Tokenizer:
    """
    Tokenizer class supporting multiple tokenization methods.
    - Word-level tokenization
    - Character-level tokenization
    - Subword tokenization (BPE via SentencePiece)
    """

    def __init__(
        self,
        method: str = "word",
        model_path: Optional[str] = None,
        vocab_size: int = 32000
    ):
        """
        Initialize the tokenizer.
        
        Args:
            method: Tokenization method ('word', 'char', 'bpe')
            model_path: Path to pretrained tokenizer model (for BPE)
            vocab_size: Vocabulary size for BPE training
        """
        self.method = method
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.sp_model = None
        
        if method == "bpe" and model_path:
            self._load_sentencepiece_model(model_path)

    def _load_sentencepiece_model(self, model_path: str):
        try:
            import sentencepiece as spm
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.Load(model_path)
            logger.info(f"Loaded SentencePiece model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load SentencePiece model: {e}")
            raise
    
    def train_bpe(
        self,
        input_file: str,
        model_prefix: str,
        vocab_size: Optional[int] = None
    ):
        """
        Train a BPE model using SentencePiece.
        
        Args:
            input_file: Path to training data
            model_prefix: Prefix for output model files
            vocab_size: Vocabulary size (uses default if not provided)
        """
        import sentencepiece as spm
        
        vocab_size = vocab_size or self.vocab_size
        
        spm.SentencePieceTrainer.Train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type='bpe',
            character_coverage=0.9995,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece='<pad>',
            unk_piece='<unk>',
            bos_piece='<bos>',
            eos_piece='<eos>'
        )
        
        # Load the trained model
        self.model_path = f"{model_prefix}.model"
        self._load_sentencepiece_model(self.model_path)
        
        logger.info(f"Trained BPE model with vocab size {vocab_size}")

    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a text string.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if self.method == "word":
            return self._word_tokenize(text)
        elif self.method == "char":
            return self._char_tokenize(text)
        elif self.method == "bpe":
            return self._bpe_tokenize(text)
        else:
            raise ValueError(f"Unknown tokenization method: {self.method}")
    
    def _word_tokenize(self, text: str) -> List[str]:
        """Simple word-level tokenization."""
        return text.strip().split()
    
    def _char_tokenize(self, text: str) -> List[str]:
        """Character-level tokenization."""
        return list(text.strip())
    
    def _bpe_tokenize(self, text: str) -> List[str]:
        """BPE tokenization using SentencePiece."""
        if self.sp_model is None:
            raise RuntimeError("SentencePiece model not loaded")
        return self.sp_model.EncodeAsPieces(text)
    
    def detokenize(self, tokens: List[str]) -> str:
        """
        Convert tokens back to text.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Reconstructed text
        """
        if self.method == "word":
            return ' '.join(tokens)
        elif self.method == "char":
            return ''.join(tokens)
        elif self.method == "bpe":
            if self.sp_model is None:
                raise RuntimeError("SentencePiece model not loaded")
            return self.sp_model.DecodePieces(tokens)
        else:
            raise ValueError(f"Unknown tokenization method: {self.method}")
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs (for BPE only).
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        if self.method != "bpe" or self.sp_model is None:
            raise RuntimeError("Encode only available for BPE with loaded model")
        return self.sp_model.EncodeAsIds(text)
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs to text (for BPE only).
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text
        """
        if self.method != "bpe" or self.sp_model is None:
            raise RuntimeError("Decode only available for BPE with loaded model")
        return self.sp_model.DecodeIds(ids)


if __name__ == "__main__":
    # Test word tokenizer
    tokenizer = Tokenizer(method="word")
    text = "Hello world, this is a test."
    tokens = tokenizer.tokenize(text)
    print(f"Word tokens: {tokens}")
    print(f"Detokenized: {tokenizer.detokenize(tokens)}")


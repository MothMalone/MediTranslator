"""
Translator Interface
High-level interface for translating text using trained model.

Supports both word-level and BPE (SentencePiece) vocabularies.
"""
import torch
import torch.nn as nn
from typing import List, Union, Optional
import logging

from .greedy_search import greedy_decode
from .beam_search import beam_search_decode

logger = logging.getLogger(__name__)


def is_sentencepiece_vocab(vocab) -> bool:
    """Check if vocab is a SentencePiece vocabulary."""
    return hasattr(vocab, 'sp') and hasattr(vocab, 'encode')


class Translator:
    """
    High-level translation interface.
    
    Handles:
        - Loading model
        - Tokenization (word-level or BPE)
        - Decoding
        - Detokenization
    
    Args:
        model: Trained Transformer model
        src_vocab: Source vocabulary (Vocabulary or SentencePieceVocab)
        tgt_vocab: Target vocabulary (Vocabulary or SentencePieceVocab)
        tokenizer: Tokenizer (optional, for word-level)
        device: Device to run on
        decoding_method: 'greedy' or 'beam'
        beam_size: Beam size for beam search
        max_length: Maximum output length
        length_penalty: Length penalty for beam search
    """
    
    def __init__(
        self,
        model: nn.Module,
        src_vocab,
        tgt_vocab,
        tokenizer=None,
        device: torch.device = None,
        decoding_method: str = 'beam',
        beam_size: int = 5,
        max_length: int = 128,
        length_penalty: float = 0.6
    ):
        self.model = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.tokenizer = tokenizer
        self.device = device or torch.device('cpu')
        self.decoding_method = decoding_method
        self.beam_size = beam_size
        self.max_length = max_length
        self.length_penalty = length_penalty
        
        # Detect if using BPE
        self.use_bpe = is_sentencepiece_vocab(src_vocab)
        if self.use_bpe:
            logger.info("Translator using BPE (SentencePiece) tokenization")
        else:
            logger.info("Translator using word-level tokenization")
        
        self.model.to(self.device)
        self.model.eval()
    
    def translate(
        self,
        text: Union[str, List[str]],
        return_tokens: bool = False
    ) -> Union[str, List[str]]:
        """
        Translate text.
        
        Args:
            text: Input text or list of texts
            return_tokens: Whether to return tokens instead of text
            
        Returns:
            Translated text or list of texts
        """
        # Handle single string
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False
        
        # Tokenize
        src_tokens_list = self._tokenize(texts)
        
        # Convert to tensor
        src_tensor = self._prepare_input(src_tokens_list)
        
        # Decode
        if self.decoding_method == 'greedy':
            output_tokens = self._greedy_translate(src_tensor)
        elif self.decoding_method == 'beam':
            output_tokens = self._beam_translate(src_tensor)
        else:
            raise ValueError(f"Unknown decoding method: {self.decoding_method}")
        
        # Return tokens if requested
        if return_tokens:
            if single_input:
                return output_tokens[0]
            return output_tokens
        
        # Detokenize
        translations = self._detokenize(output_tokens)
        
        if single_input:
            return translations[0]
        return translations
    
    def _tokenize(self, texts: List[str]) -> List[List[str]]:
        """Tokenize input texts (only used for word-level)."""
        if self.use_bpe:
            # For BPE, we don't tokenize to strings - handled in _prepare_input
            return texts  # Return raw texts
        elif self.tokenizer:
            return [self.tokenizer.tokenize(text) for text in texts]
        else:
            return [text.strip().split() for text in texts]
    
    def _prepare_input(self, tokens_or_texts) -> torch.Tensor:
        """Convert tokens/texts to tensor with padding."""
        encoded = []
        max_len = 0
        
        if self.use_bpe:
            # BPE: Use SentencePiece to encode directly from text
            for text in tokens_or_texts:
                # SentencePieceVocab.encode returns token IDs
                indices = self.src_vocab.encode(text, add_bos=True, add_eos=True)
                encoded.append(indices)
                max_len = max(max_len, len(indices))
        else:
            # Word-level: Convert token strings to indices
            for tokens in tokens_or_texts:
                # Add BOS/EOS
                indices = [self.src_vocab.bos_idx]
                indices.extend(self.src_vocab.encode(tokens))
                indices.append(self.src_vocab.eos_idx)
                
                encoded.append(indices)
                max_len = max(max_len, len(indices))
        
        # Pad sequences
        padded = []
        for indices in encoded:
            padding_length = max_len - len(indices)
            padded.append(indices + [self.src_vocab.pad_idx] * padding_length)
        
        return torch.tensor(padded, dtype=torch.long, device=self.device)
    
    @torch.no_grad()
    def _greedy_translate(self, src: torch.Tensor) -> List[List[int]]:
        """Translate using greedy decoding."""
        output = greedy_decode(
            self.model,
            src,
            max_length=self.max_length,
            bos_idx=self.tgt_vocab.bos_idx,
            eos_idx=self.tgt_vocab.eos_idx,
            pad_idx=self.tgt_vocab.pad_idx
        )
        
        # Convert to list of lists
        results = []
        for seq in output:
            tokens = seq.tolist()
            # Remove special tokens
            tokens = self._remove_special_tokens(tokens)
            results.append(tokens)
        
        return results
    
    @torch.no_grad()
    def _beam_translate(self, src: torch.Tensor) -> List[List[int]]:
        """Translate using beam search."""
        output = beam_search_decode(
            self.model,
            src,
            beam_size=self.beam_size,
            max_length=self.max_length,
            bos_idx=self.tgt_vocab.bos_idx,
            eos_idx=self.tgt_vocab.eos_idx,
            pad_idx=self.tgt_vocab.pad_idx,
            length_penalty=self.length_penalty
        )
        
        # Remove special tokens
        results = []
        for tokens in output:
            tokens = self._remove_special_tokens(tokens)
            results.append(tokens)
        
        return results
    
    def _remove_special_tokens(self, tokens: List[int]) -> List[int]:
        """Remove BOS, EOS, and PAD tokens."""
        special_indices = {
            self.tgt_vocab.bos_idx,
            self.tgt_vocab.eos_idx,
            self.tgt_vocab.pad_idx
        }
        return [t for t in tokens if t not in special_indices]
    
    def _detokenize(self, tokens_list: List[List[int]]) -> List[str]:
        """Convert token indices to text."""
        results = []
        
        for tokens in tokens_list:
            if self.use_bpe:
                # BPE: Use SentencePiece decode - it handles subword merging automatically
                text = self.tgt_vocab.decode(tokens, skip_special=True)
            else:
                # Word-level: Decode indices to tokens, then join
                text_tokens = self.tgt_vocab.decode(tokens, skip_special=True)
                
                # Join tokens
                if self.tokenizer:
                    text = self.tokenizer.detokenize(text_tokens)
                else:
                    text = ' '.join(text_tokens)
            
            results.append(text)
        
        return results
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        src_vocab_path: str,
        tgt_vocab_path: str,
        device: torch.device = None,
        **kwargs
    ) -> 'Translator':
        """
        Load translator from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            src_vocab_path: Path to source vocabulary
            tgt_vocab_path: Path to target vocabulary
            device: Device to load model on
            **kwargs: Additional arguments for Translator
            
        Returns:
            Translator instance
        """
        from ..data.vocabulary import Vocabulary
        from ..models.transformer import Transformer
        
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load vocabularies
        src_vocab = Vocabulary.load(src_vocab_path)
        tgt_vocab = Vocabulary.load(tgt_vocab_path)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint['config']
        
        # Create model
        model = Transformer(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            **config['model']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return cls(
            model=model,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            device=device,
            **kwargs
        )


if __name__ == "__main__":
    print("Translator module loaded successfully")

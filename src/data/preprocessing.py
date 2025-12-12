"""
Data Preprocessing Module
Handles data cleaning, normalization, and preparation for training.
"""
import re
import unicodedata
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Data preprocessor for machine translation datasets.
    Handles cleaning, normalization, and filtering of text data.
    """
    
    def __init__(
        self,
        lowercase: bool = False,
        remove_punctuation: bool = False,
        normalize_unicode: bool = True,
        min_length: int = 1,
        max_length: int = 512,
        max_ratio: float = 3.0
    ):
        """
        Initialize the preprocessor.
        
        Args:
            lowercase: Whether to convert text to lowercase
            remove_punctuation: Whether to remove punctuation
            normalize_unicode: Whether to normalize unicode characters
            min_length: Minimum sentence length (in tokens)
            max_length: Maximum sentence length (in tokens)
            max_ratio: Maximum ratio between source and target lengths
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.normalize_unicode = normalize_unicode
        self.min_length = min_length
        self.max_length = max_length
        self.max_ratio = max_ratio
    
    def clean_text(self, text: str) -> str:
        """
        Clean a single text string.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Strip whitespace
        text = text.strip()
        
        # Normalize unicode
        if self.normalize_unicode:
            text = unicodedata.normalize('NFC', text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def filter_pair(
        self,
        src: str,
        tgt: str,
        src_tokens: Optional[List[str]] = None,
        tgt_tokens: Optional[List[str]] = None
    ) -> bool:
        """
        Check if a sentence pair should be kept.
        
        Args:
            src: Source sentence
            tgt: Target sentence
            src_tokens: Pre-tokenized source (optional)
            tgt_tokens: Pre-tokenized target (optional)
            
        Returns:
            True if the pair should be kept, False otherwise
        """
        # Use provided tokens or split by whitespace
        if src_tokens is None:
            src_tokens = src.split()
        if tgt_tokens is None:
            tgt_tokens = tgt.split()
        
        src_len = len(src_tokens)
        tgt_len = len(tgt_tokens)
        
        # Check minimum length
        if src_len < self.min_length or tgt_len < self.min_length:
            return False
        
        # Check maximum length
        if src_len > self.max_length or tgt_len > self.max_length:
            return False
        
        # Check length ratio
        if src_len > 0 and tgt_len > 0:
            ratio = max(src_len / tgt_len, tgt_len / src_len)
            if ratio > self.max_ratio:
                return False
        
        return True
    
    def process_file(
        self,
        src_path: str,
        tgt_path: str,
        output_src_path: str,
        output_tgt_path: str
    ) -> Tuple[int, int]:
        """
        Process parallel files and save cleaned data.
        
        Args:
            src_path: Path to source file
            tgt_path: Path to target file
            output_src_path: Path to output source file
            output_tgt_path: Path to output target file
            
        Returns:
            Tuple of (original_count, filtered_count)
        """
        original_count = 0
        kept_count = 0
        
        with open(src_path, 'r', encoding='utf-8') as src_file, \
             open(tgt_path, 'r', encoding='utf-8') as tgt_file, \
             open(output_src_path, 'w', encoding='utf-8') as out_src, \
             open(output_tgt_path, 'w', encoding='utf-8') as out_tgt:
            
            for src_line, tgt_line in zip(src_file, tgt_file):
                original_count += 1
                
                # Clean text
                src_clean = self.clean_text(src_line)
                tgt_clean = self.clean_text(tgt_line)
                
                # Filter
                if self.filter_pair(src_clean, tgt_clean):
                    out_src.write(src_clean + '\n')
                    out_tgt.write(tgt_clean + '\n')
                    kept_count += 1
        
        logger.info(f"Processed {original_count} pairs, kept {kept_count} "
                   f"({kept_count/original_count*100:.1f}%)")
        
        return original_count, kept_count
    
    def get_statistics(
        self,
        src_path: str,
        tgt_path: str
    ) -> dict:
        """
        Get statistics about the dataset.
        
        Args:
            src_path: Path to source file
            tgt_path: Path to target file
            
        Returns:
            Dictionary with statistics
        """
        src_lengths = []
        tgt_lengths = []
        
        with open(src_path, 'r', encoding='utf-8') as src_file, \
             open(tgt_path, 'r', encoding='utf-8') as tgt_file:
            
            for src_line, tgt_line in zip(src_file, tgt_file):
                src_lengths.append(len(src_line.strip().split()))
                tgt_lengths.append(len(tgt_line.strip().split()))
        
        import numpy as np
        
        stats = {
            'num_pairs': len(src_lengths),
            'src_avg_length': np.mean(src_lengths),
            'src_max_length': np.max(src_lengths),
            'src_min_length': np.min(src_lengths),
            'tgt_avg_length': np.mean(tgt_lengths),
            'tgt_max_length': np.max(tgt_lengths),
            'tgt_min_length': np.min(tgt_lengths),
        }
        
        return stats


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor(
        normalize_unicode=True,
        min_length=1,
        max_length=128
    )
    
    # Test cleaning
    test_text = "  Hello   World!  "
    print(f"Original: '{test_text}'")
    print(f"Cleaned: '{preprocessor.clean_text(test_text)}'")

"""
BLEU Score Calculation
Uses sacrebleu - the industry-standard BLEU implementation.

sacrebleu provides:
- Accurate, standardized BLEU computation
- Compatible with WMT/official evaluation
- Handles tokenization edge cases correctly
- Much faster than custom implementation
"""
from typing import List
import logging

# Use sacrebleu for accurate BLEU scoring
try:
    from sacrebleu.metrics import BLEU
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    logging.warning("sacrebleu not available. Install with: pip install sacrebleu")

logger = logging.getLogger(__name__)


def corpus_bleu(
    hypotheses: List[List[str]],
    references: List[List[List[str]]],
    max_n: int = 4,
    smooth: bool = False
) -> float:
    """
    Calculate corpus-level BLEU score using sacrebleu.
    
    This is the recommended way to compute BLEU for machine translation.
    Uses the official sacrebleu implementation which is:
    - More accurate than custom implementations
    - Compatible with WMT evaluation
    - Handles edge cases correctly
    
    Args:
        hypotheses: List of hypothesis token lists
            Example: [['I', 'love', 'python'], ['hello', 'world']]
        references: List of reference token lists (each hyp can have multiple refs)
            Example: [[['I', 'love', 'python']], [['hello', 'world']]]
        max_n: Maximum n-gram order (default: 4 for BLEU-4)
        smooth: Use smoothing for zero counts
        
    Returns:
        BLEU score as a float (0-1 range)
        
    Note:
        sacrebleu expects strings, so we join tokens with spaces
    """
    if not SACREBLEU_AVAILABLE:
        logger.error("sacrebleu not installed. Cannot compute BLEU score.")
        return 0.0
    
    if not hypotheses or not references:
        logger.warning("Empty hypotheses or references")
        return 0.0
    
    # Convert token lists to strings (sacrebleu expects strings)
    hyp_strings = [' '.join(tokens) for tokens in hypotheses]
    
    # sacrebleu expects references as List[List[str]] where inner lists are multiple refs per hypothesis
    # Our format: references[i] = [[ref1_tokens], [ref2_tokens], ...]
    # Need: refs_transposed[ref_idx][hyp_idx] = ref_string
    
    # Get number of references per hypothesis (assume all have same number)
    num_refs = len(references[0]) if references else 0
    
    # Transpose and convert to strings
    ref_strings = []
    for ref_idx in range(num_refs):
        ref_list = []
        for hyp_idx in range(len(references)):
            tokens = references[hyp_idx][ref_idx]
            ref_list.append(' '.join(tokens))
        ref_strings.append(ref_list)
    
    # Compute BLEU
    bleu = BLEU(max_ngram_order=max_n, smooth_method='exp' if smooth else 'none')
    result = bleu.corpus_score(hyp_strings, ref_strings)
    
    # sacrebleu returns score in 0-100 range, convert to 0-1
    return result.score / 100.0


def sentence_bleu(
    hypothesis: List[str],
    references: List[List[str]],
    max_n: int = 4,
    smooth: bool = False
) -> float:
    """
    Calculate sentence-level BLEU score using sacrebleu.
    
    Args:
        hypothesis: Hypothesis token list
        references: List of reference token lists
        max_n: Maximum n-gram order
        smooth: Use smoothing
        
    Returns:
        BLEU score (0-1 range)
    """
    # Use corpus_bleu with single sentence
    return corpus_bleu([hypothesis], [references], max_n=max_n, smooth=smooth)


# Backward compatibility alias
def compute_bleu(hypotheses: List[List[str]], references: List[List[List[str]]], 
                 max_n: int = 4) -> float:
    """Alias for corpus_bleu for backward compatibility."""
    return corpus_bleu(hypotheses, references, max_n=max_n)


class BLEUCalculator:
    """
    BLEU Calculator wrapper for sacrebleu.
    
    Provides a convenient interface for calculating BLEU scores from strings.
    Uses sacrebleu which is case-insensitive by default (standard MT evaluation).
    """
    
    def __init__(self, lowercase: bool = True, tokenize: str = '13a'):
        """
        Initialize BLEU calculator.
        
        Args:
            lowercase: Lowercase text before scoring (default: True, standard practice)
            tokenize: Tokenization method ('13a', 'intl', 'zh', 'none')
                      '13a' = standard Moses tokenizer (recommended)
        """
        if not SACREBLEU_AVAILABLE:
            raise ImportError("sacrebleu required. Install with: pip install sacrebleu")
        
        self.bleu = BLEU(lowercase=lowercase, tokenize=tokenize)
        logger.info(f"BLEUCalculator initialized (lowercase={lowercase}, tokenize={tokenize})")
    
    def calculate(self, hypotheses: List[str], references: List[str]) -> dict:
        """
        Calculate BLEU score from string lists.
        
        Args:
            hypotheses: List of hypothesis strings
            references: List of reference strings (one per hypothesis)
            
        Returns:
            Dictionary with:
                - bleu: BLEU score (0-100 range, standard MT reporting)
                - precisions: List of n-gram precisions [1-gram, 2-gram, 3-gram, 4-gram]
                - bp: Brevity penalty (0-1)
                - ratio: Length ratio (hypothesis / reference)
                - hyp_len: Total hypothesis length
                - ref_len: Total reference length
        """
        if len(hypotheses) != len(references):
            raise ValueError(f"Mismatch: {len(hypotheses)} hypotheses vs {len(references)} references")
        
        # sacrebleu expects references as List[List[str]]
        # where each inner list contains all reference translations for one source
        refs_transposed = [[ref] for ref in references]
        refs_transposed = list(zip(*refs_transposed))  # Transpose to [[ref1, ref2, ...]]
        
        # Compute corpus BLEU
        result = self.bleu.corpus_score(hypotheses, refs_transposed)
        
        return {
            'bleu': result.score,  # 0-100 range
            'precisions': result.precisions,
            'bp': result.bp,
            'ratio': result.sys_len / result.ref_len if result.ref_len > 0 else 0.0,
            'hyp_len': result.sys_len,
            'ref_len': result.ref_len
        }
    
    def calculate_from_files(self, hypothesis_file: str, reference_file: str) -> dict:
        """
        Calculate BLEU from text files.
        
        Args:
            hypothesis_file: Path to hypothesis file (one sentence per line)
            reference_file: Path to reference file (one sentence per line)
            
        Returns:
            Dictionary with BLEU metrics
        """
        with open(hypothesis_file, 'r', encoding='utf-8') as f:
            hypotheses = [line.strip() for line in f]
        
        with open(reference_file, 'r', encoding='utf-8') as f:
            references = [line.strip() for line in f]
        
        return self.calculate(hypotheses, references)


if __name__ == "__main__":
    # Test BLEU calculation
    print("Testing BLEU with sacrebleu...")
    
    # Example 1: Perfect match
    hyp = [['the', 'cat', 'is', 'on', 'the', 'mat']]
    ref = [[['the', 'cat', 'is', 'on', 'the', 'mat']]]
    score = corpus_bleu(hyp, ref)
    print(f"Perfect match BLEU: {score:.4f} (expected: 1.0)")
    
    # Example 2: Partial match
    hyp = [['the', 'cat', 'sat', 'on', 'the', 'mat']]
    ref = [[['the', 'cat', 'is', 'on', 'the', 'mat']]]
    score = corpus_bleu(hyp, ref)
    print(f"Partial match BLEU: {score:.4f}")
    
    # Example 3: Multiple sentences
    hyp = [
        ['the', 'cat', 'is', 'sleeping'],
        ['I', 'love', 'python']
    ]
    ref = [
        [['the', 'cat', 'is', 'sleeping']],
        [['I', 'love', 'python']]
    ]
    score = corpus_bleu(hyp, ref)
    print(f"Corpus BLEU: {score:.4f}")

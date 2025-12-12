"""
BLEU Score Calculation
Implements BLEU (Bilingual Evaluation Understudy) metric for machine translation.
"""
import math
from collections import Counter
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def get_ngrams(tokens: List[str], n: int) -> Counter:
    """
    Extract n-grams from a list of tokens.
    
    Args:
        tokens: List of tokens
        n: N-gram order
        
    Returns:
        Counter of n-grams
    """
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams.append(ngram)
    return Counter(ngrams)


def modified_precision(
    hypothesis: List[str],
    references: List[List[str]],
    n: int
) -> Tuple[int, int]:
    """
    Calculate modified precision for n-grams.
    
    Clips each n-gram count by its maximum reference count.
    
    Args:
        hypothesis: Hypothesis tokens
        references: List of reference token lists
        n: N-gram order
        
    Returns:
        Tuple of (clipped count, total count)
    """
    # Get hypothesis n-grams
    hyp_ngrams = get_ngrams(hypothesis, n)
    
    if not hyp_ngrams:
        return 0, 0
    
    # Get maximum reference n-gram counts
    max_ref_counts = Counter()
    for reference in references:
        ref_ngrams = get_ngrams(reference, n)
        for ngram, count in ref_ngrams.items():
            max_ref_counts[ngram] = max(max_ref_counts[ngram], count)
    
    # Calculate clipped count
    clipped_count = 0
    total_count = 0
    
    for ngram, count in hyp_ngrams.items():
        clipped_count += min(count, max_ref_counts[ngram])
        total_count += count
    
    return clipped_count, total_count


def brevity_penalty(
    hypothesis_length: int,
    reference_lengths: List[int]
) -> float:
    """
    Calculate brevity penalty.
    
    Penalizes hypotheses shorter than references.
    
    Args:
        hypothesis_length: Length of hypothesis
        reference_lengths: Lengths of references
        
    Returns:
        Brevity penalty factor (0-1)
    """
    if hypothesis_length == 0:
        return 0.0
    
    # Find closest reference length
    closest_length = min(
        reference_lengths,
        key=lambda ref_len: (abs(ref_len - hypothesis_length), ref_len)
    )
    
    if hypothesis_length >= closest_length:
        return 1.0
    else:
        return math.exp(1 - closest_length / hypothesis_length)


def sentence_bleu(
    hypothesis: List[str],
    references: List[List[str]],
    max_n: int = 4,
    weights: Optional[Tuple[float, ...]] = None
) -> float:
    """
    Calculate sentence-level BLEU score.
    
    Args:
        hypothesis: Hypothesis tokens
        references: List of reference token lists
        max_n: Maximum n-gram order
        weights: Weights for each n-gram order
        
    Returns:
        BLEU score (0-1)
    """
    if weights is None:
        weights = tuple([1.0 / max_n] * max_n)
    
    # Calculate modified precision for each n-gram order
    precisions = []
    for n in range(1, max_n + 1):
        clipped, total = modified_precision(hypothesis, references, n)
        if total == 0:
            precision = 0.0
        else:
            precision = clipped / total
        precisions.append(precision)
    
    # Check for zero precisions
    if 0 in precisions:
        return 0.0
    
    # Calculate geometric mean of precisions
    log_precision_sum = sum(
        w * math.log(p) for w, p in zip(weights, precisions)
    )
    
    # Calculate brevity penalty
    bp = brevity_penalty(
        len(hypothesis),
        [len(ref) for ref in references]
    )
    
    # Calculate final BLEU
    bleu = bp * math.exp(log_precision_sum)
    
    return bleu


def corpus_bleu(
    hypotheses: List[List[str]],
    references: List[List[List[str]]],
    max_n: int = 4,
    weights: Optional[Tuple[float, ...]] = None
) -> float:
    """
    Calculate corpus-level BLEU score.
    
    Args:
        hypotheses: List of hypothesis token lists
        references: List of reference lists (each hypothesis can have multiple references)
        max_n: Maximum n-gram order
        weights: Weights for each n-gram order
        
    Returns:
        BLEU score (0-1)
    """
    if weights is None:
        weights = tuple([1.0 / max_n] * max_n)
    
    # Accumulate counts across corpus
    total_clipped = [0] * max_n
    total_count = [0] * max_n
    total_hyp_length = 0
    total_ref_length = 0
    
    for hyp, refs in zip(hypotheses, references):
        # Accumulate n-gram counts
        for n in range(1, max_n + 1):
            clipped, count = modified_precision(hyp, refs, n)
            total_clipped[n-1] += clipped
            total_count[n-1] += count
        
        # Accumulate lengths for brevity penalty
        total_hyp_length += len(hyp)
        
        # Find closest reference length
        ref_lengths = [len(ref) for ref in refs]
        closest_ref_length = min(
            ref_lengths,
            key=lambda ref_len: (abs(ref_len - len(hyp)), ref_len)
        )
        total_ref_length += closest_ref_length
    
    # Calculate corpus-level precisions
    precisions = []
    for n in range(max_n):
        if total_count[n] == 0:
            precision = 0.0
        else:
            precision = total_clipped[n] / total_count[n]
        precisions.append(precision)
    
    # Check for zero precisions (add smoothing)
    if 0 in precisions:
        # Use smoothing: add small count
        return 0.0
    
    # Calculate geometric mean
    log_precision_sum = sum(
        w * math.log(p) for w, p in zip(weights, precisions)
    )
    
    # Calculate brevity penalty
    if total_hyp_length >= total_ref_length:
        bp = 1.0
    else:
        bp = math.exp(1 - total_ref_length / total_hyp_length)
    
    # Calculate final BLEU
    bleu = bp * math.exp(log_precision_sum)
    
    return bleu


def compute_bleu(
    hypotheses: List[str],
    references: List[str],
    tokenize: bool = True
) -> float:
    """
    Convenience function to compute BLEU from strings.
    
    Args:
        hypotheses: List of hypothesis strings
        references: List of reference strings
        tokenize: Whether to tokenize by whitespace
        
    Returns:
        BLEU score (0-100)
    """
    if tokenize:
        hyp_tokens = [hyp.strip().split() for hyp in hypotheses]
        ref_tokens = [[ref.strip().split()] for ref in references]
    else:
        hyp_tokens = hypotheses
        ref_tokens = [[ref] for ref in references]
    
    bleu = corpus_bleu(hyp_tokens, ref_tokens)
    
    return bleu * 100


class BLEUCalculator:
    """
    BLEU Calculator class with support for sacrebleu.
    
    Can use either custom implementation or sacrebleu library.
    """
    
    def __init__(self, use_sacrebleu: bool = True):
        """
        Initialize BLEU calculator.
        
        Args:
            use_sacrebleu: Whether to use sacrebleu library
        """
        self.use_sacrebleu = use_sacrebleu
        
        if use_sacrebleu:
            try:
                import sacrebleu
                self.sacrebleu = sacrebleu
                logger.info("Using sacrebleu for BLEU calculation")
            except ImportError:
                logger.warning("sacrebleu not available, using custom implementation")
                self.use_sacrebleu = False
    
    def calculate(
        self,
        hypotheses: List[str],
        references: List[str]
    ) -> dict:
        """
        Calculate BLEU score.
        
        Args:
            hypotheses: List of hypothesis strings
            references: List of reference strings
            
        Returns:
            Dictionary with BLEU score and other metrics
        """
        if self.use_sacrebleu:
            bleu = self.sacrebleu.corpus_bleu(hypotheses, [references])
            return {
                'bleu': bleu.score,
                'precisions': bleu.precisions,
                'bp': bleu.bp,
                'ratio': bleu.sys_len / bleu.ref_len,
                'hyp_len': bleu.sys_len,
                'ref_len': bleu.ref_len
            }
        else:
            bleu = compute_bleu(hypotheses, references)
            return {
                'bleu': bleu
            }
    
    def calculate_from_files(
        self,
        hypothesis_file: str,
        reference_file: str
    ) -> dict:
        """
        Calculate BLEU from files.
        
        Args:
            hypothesis_file: Path to hypothesis file
            reference_file: Path to reference file
            
        Returns:
            Dictionary with BLEU score
        """
        with open(hypothesis_file, 'r', encoding='utf-8') as f:
            hypotheses = [line.strip() for line in f]
        
        with open(reference_file, 'r', encoding='utf-8') as f:
            references = [line.strip() for line in f]
        
        return self.calculate(hypotheses, references)


if __name__ == "__main__":
    # Test BLEU calculation
    hypotheses = [
        "the cat sat on the mat",
        "there is a cat on the mat"
    ]
    references = [
        "the cat is on the mat",
        "there is a cat on the mat"
    ]
    
    # Test custom implementation
    bleu = compute_bleu(hypotheses, references)
    print(f"Custom BLEU: {bleu:.2f}")
    
    # Test with BLEUCalculator
    calculator = BLEUCalculator(use_sacrebleu=False)
    result = calculator.calculate(hypotheses, references)
    print(f"BLEUCalculator: {result}")

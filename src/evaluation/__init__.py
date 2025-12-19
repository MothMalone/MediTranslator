"""
Evaluation Module
"""
from .bleu import compute_bleu, corpus_bleu, sentence_bleu

__all__ = [
    'compute_bleu',
    'corpus_bleu',
    'sentence_bleu'
]

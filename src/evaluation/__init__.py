"""
Evaluation Module
"""
from .bleu import compute_bleu, BLEUCalculator

__all__ = [
    'compute_bleu',
    'BLEUCalculator'
]

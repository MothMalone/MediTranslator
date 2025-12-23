"""
Vocabulary Expansion for Domain Adaptation
Expand BPE vocabulary with domain-specific terms from training data.

This helps when the base vocabulary doesn't cover specialized terms well.

Usage:
    python scripts/expand_vocab.py --config experiments/medical_vocab_expanded_en2vi/config.yaml
"""
import argparse
import os
import sys
from collections import Counter
from typing import Set, List, Tuple
import sentencepiece as spm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.data.sp_vocab import SentencePieceVocab


def extract_oov_words(
    text_file: str,
    sp_model: spm.SentencePieceProcessor,
    min_freq: int = 2
) -> Counter:
    """
    Extract out-of-vocabulary (OOV) words from text file.
    
    Args:
        text_file: Path to text file
        sp_model: SentencePiece model
        min_freq: Minimum frequency threshold
        
    Returns:
        Counter of words and their frequencies
    """
    print(f"Scanning {text_file}...")
    
    word_counter = Counter()
    oov_word_counter = Counter()
    
    with open(text_file, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            word_counter.update(words)
    
    print(f"Found {len(word_counter)} unique words")
    
    # Check which words are poorly represented (split into too many pieces)
    for word, freq in word_counter.items():
        if freq < min_freq:
            continue
            
        # Encode word with SentencePiece
        pieces = sp_model.encode(word, out_type=str)
        
        # If word is split into many pieces, it might benefit from being in vocab
        if len(pieces) > 2:  # Split into 3+ pieces
            oov_word_counter[word] = freq
    
    return oov_word_counter


def select_top_candidates(
    src_oov: Counter,
    tgt_oov: Counter,
    max_new_src: int,
    max_new_tgt: int
) -> Tuple[List[str], List[str]]:
    """
    Select top candidates for vocabulary expansion.
    
    Args:
        src_oov: Source OOV words with frequencies
        tgt_oov: Target OOV words with frequencies
        max_new_src: Maximum new source words
        max_new_tgt: Maximum new target words
        
    Returns:
        (src_new_words, tgt_new_words)
    """
    # Sort by frequency and take top N
    src_new = [word for word, _ in src_oov.most_common(max_new_src)]
    tgt_new = [word for word, _ in tgt_oov.most_common(max_new_tgt)]
    
    return src_new, tgt_new


def expand_sentencepiece_vocab(
    base_model_path: str,
    new_words: List[str],
    output_dir: str,
    prefix: str
):
    """
    Expand SentencePiece vocabulary with new words.
    
    This creates a new vocab file with additional entries.
    
    Args:
        base_model_path: Path to base .model file
        new_words: List of new words to add
        output_dir: Output directory
        prefix: Prefix for output files (src/tgt)
    """
    print(f"Expanding {prefix} vocabulary...")
    
    # Load base model
    sp = spm.SentencePieceProcessor()
    sp.load(base_model_path)
    
    base_vocab_size = sp.vocab_size()
    
    # Create expanded vocab file
    base_vocab_path = base_model_path.replace('.model', '.vocab')
    expanded_vocab_path = os.path.join(output_dir, f'{prefix}.vocab')
    
    # Copy base vocab
    with open(base_vocab_path, 'r', encoding='utf-8') as f_in:
        base_vocab_lines = f_in.readlines()
    
    # Add new words with scores (use score slightly lower than minimum)
    # Get minimum score from base vocab
    min_score = float('inf')
    for line in base_vocab_lines:
        if '\t' in line:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                score = float(parts[1])
                min_score = min(min_score, score)
    
    new_vocab_lines = base_vocab_lines.copy()
    
    # Add new words with scores slightly below minimum
    for i, word in enumerate(new_words):
        score = min_score - (i + 1) * 0.1
        # Add BPE prefix
        piece = f"▁{word}" if not word.startswith("▁") else word
        new_vocab_lines.append(f"{piece}\t{score}\n")
    
    # Write expanded vocab
    with open(expanded_vocab_path, 'w', encoding='utf-8') as f:
        f.writelines(new_vocab_lines)
    
    # Copy model file (NOTE: model file still uses base vocab internally)
    # The .vocab file is what's actually used for encoding/decoding
    import shutil
    expanded_model_path = os.path.join(output_dir, f'{prefix}.model')
    shutil.copy(base_model_path, expanded_model_path)
    
    print(f"✓ Expanded vocab: {base_vocab_size} → {base_vocab_size + len(new_words)}")
    print(f"  Vocab file: {expanded_vocab_path}")
    print(f"  Model file: {expanded_model_path}")
    print(f"  NOTE: .vocab file is used by the model, new tokens are available!")
    
    return base_vocab_size + len(new_words)


def main():
    parser = argparse.ArgumentParser(description="Expand vocabulary with domain-specific terms")
    parser.add_argument('--config', type=str, required=True, help='Config file')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Check if vocab expansion is enabled
    vocab_exp_config = config.get('vocab_expansion', {})
    if not vocab_exp_config.get('enabled', False):
        print("Vocabulary expansion is disabled in config.")
        print("Set vocab_expansion.enabled = true to enable it.")
        return
    
    print("=" * 80)
    print("VOCABULARY EXPANSION")
    print("=" * 80)
    
    # Get paths
    data_config = config['data']
    paths_config = config['paths']
    
    base_vocab_dir = paths_config['vocab_dir']
    expanded_vocab_dir = paths_config['expanded_vocab_dir']
    
    train_src = data_config['train_src']
    train_tgt = data_config['train_tgt']
    
    max_new_src = vocab_exp_config.get('max_new_words_src', 5000)
    max_new_tgt = vocab_exp_config.get('max_new_words_tgt', 5000)
    min_freq = vocab_exp_config.get('min_freq', 2)
    
    print(f"Base vocab: {base_vocab_dir}")
    print(f"Expanded vocab: {expanded_vocab_dir}")
    print(f"Training data: {train_src}, {train_tgt}")
    print(f"Max new words: src={max_new_src}, tgt={max_new_tgt}")
    print(f"Min frequency: {min_freq}")
    print()
    
    # Create output directory
    os.makedirs(expanded_vocab_dir, exist_ok=True)
    
    # Load base SentencePiece models
    print("Loading base SentencePiece models...")
    src_sp_path = os.path.join(base_vocab_dir, 'src.model')
    tgt_sp_path = os.path.join(base_vocab_dir, 'tgt.model')
    
    src_sp = spm.SentencePieceProcessor()
    src_sp.load(src_sp_path)
    
    tgt_sp = spm.SentencePieceProcessor()
    tgt_sp.load(tgt_sp_path)
    
    print(f"✓ Base vocab sizes: src={src_sp.vocab_size()}, tgt={tgt_sp.vocab_size()}")
    print()
    
    # Extract OOV words
    print("Step 1: Extracting OOV/poorly-represented words...")
    print("-" * 80)
    
    src_oov = extract_oov_words(train_src, src_sp, min_freq)
    tgt_oov = extract_oov_words(train_tgt, tgt_sp, min_freq)
    
    print(f"\nFound {len(src_oov)} source candidates")
    print(f"Found {len(tgt_oov)} target candidates")
    print()
    
    # Show top candidates
    print("Top 20 source candidates:")
    for word, freq in src_oov.most_common(20):
        pieces = src_sp.encode(word, out_type=str)
        print(f"  {word:20s} (freq={freq:5d}, pieces={len(pieces)}: {' '.join(pieces[:3])}...)")
    
    print("\nTop 20 target candidates:")
    for word, freq in tgt_oov.most_common(20):
        pieces = tgt_sp.encode(word, out_type=str)
        print(f"  {word:20s} (freq={freq:5d}, pieces={len(pieces)}: {' '.join(pieces[:3])}...)")
    print()
    
    # Select top candidates
    print("Step 2: Selecting top candidates...")
    print("-" * 80)
    
    src_new_words, tgt_new_words = select_top_candidates(
        src_oov, tgt_oov, max_new_src, max_new_tgt
    )
    
    print(f"Selected {len(src_new_words)} source words")
    print(f"Selected {len(tgt_new_words)} target words")
    print()
    
    # Expand vocabularies
    print("Step 3: Expanding vocabularies...")
    print("-" * 80)
    
    src_new_size = expand_sentencepiece_vocab(
        src_sp_path, src_new_words, expanded_vocab_dir, 'src'
    )
    
    tgt_new_size = expand_sentencepiece_vocab(
        tgt_sp_path, tgt_new_words, expanded_vocab_dir, 'tgt'
    )
    
    print()
    print("=" * 80)
    print("✅ VOCABULARY EXPANSION COMPLETED")
    print("=" * 80)
    print(f"Source vocabulary: {src_sp.vocab_size()} → {src_new_size} (+{len(src_new_words)})")
    print(f"Target vocabulary: {tgt_sp.vocab_size()} → {tgt_new_size} (+{len(tgt_new_words)})")
    print(f"\nExpanded vocabularies saved to: {expanded_vocab_dir}")
    print("\nNext steps:")
    print("1. Update config to use expanded vocab:")
    print(f"   vocab_dir: '{expanded_vocab_dir}'")
    print("2. Run fine-tuning with expanded vocabulary")
    print("=" * 80)


if __name__ == "__main__":
    main()

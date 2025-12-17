"""
Train SentencePiece BPE Models for Machine Translation
This script MUST be run before training with BPE tokenization.

Usage:
    python scripts/train_bpe.py --config experiments/v3_vi2en/config.yaml
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sentencepiece as spm
from src.utils.config import load_config


def train_sentencepiece(
    input_file: str,
    model_prefix: str,
    vocab_size: int = 32000,
    model_type: str = "bpe",
    character_coverage: float = 0.9995,
    pad_id: int = 0,
    unk_id: int = 1,
    bos_id: int = 2,
    eos_id: int = 3
):
    """
    Train a SentencePiece model.
    
    Args:
        input_file: Path to training text file
        model_prefix: Output model prefix (will create .model and .vocab files)
        vocab_size: Target vocabulary size
        model_type: 'bpe' or 'unigram'
        character_coverage: Character coverage for training
        pad_id, unk_id, bos_id, eos_id: Special token IDs
    """
    print(f"Training SentencePiece model: {model_prefix}")
    print(f"  Input: {input_file}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Model type: {model_type}")
    print(f"  Character coverage: {character_coverage}")
    
    # Train the model
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        pad_id=pad_id,
        unk_id=unk_id,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        # Additional settings for better quality
        num_threads=os.cpu_count(),
        train_extremely_large_corpus=False,
        max_sentence_length=4192,
        shuffle_input_sentence=True,
        # Normalization
        normalization_rule_name="nmt_nfkc_cf",
        # Split digits
        split_digits=True,
        # Byte fallback for unknown characters
        byte_fallback=True,
    )
    
    print(f"  ✓ Model saved to {model_prefix}.model")
    print(f"  ✓ Vocab saved to {model_prefix}.vocab")
    
    # Verify the model
    sp = spm.SentencePieceProcessor()
    sp.Load(f"{model_prefix}.model")
    print(f"  ✓ Verified: vocab_size={sp.GetPieceSize()}")
    print(f"  ✓ Special tokens: PAD={sp.pad_id()}, UNK={sp.unk_id()}, BOS={sp.bos_id()}, EOS={sp.eos_id()}")
    
    return sp


def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece BPE models")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file"
    )
    parser.add_argument(
        "--src-vocab-size",
        type=int,
        default=None,
        help="Source vocab size (overrides config)"
    )
    parser.add_argument(
        "--tgt-vocab-size",
        type=int,
        default=None,
        help="Target vocab size (overrides config)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="bpe",
        choices=["bpe", "unigram"],
        help="SentencePiece model type"
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Get paths
    vocab_dir = config['paths']['vocab_dir']
    os.makedirs(vocab_dir, exist_ok=True)
    
    train_src = config['data']['train_src']
    train_tgt = config['data']['train_tgt']
    
    # Get vocab sizes
    vocab_config = config.get('vocab', {})
    src_vocab_size = args.src_vocab_size or vocab_config.get('src_vocab_size', 32000)
    tgt_vocab_size = args.tgt_vocab_size or vocab_config.get('tgt_vocab_size', 32000)
    
    # Determine character coverage based on language
    # Vietnamese and other Asian languages need higher coverage
    src_lang = config['data'].get('src_lang', 'en')
    tgt_lang = config['data'].get('tgt_lang', 'vi')
    
    src_coverage = 0.9999 if src_lang in ['vi', 'zh', 'ja', 'ko', 'th'] else 0.9995
    tgt_coverage = 0.9999 if tgt_lang in ['vi', 'zh', 'ja', 'ko', 'th'] else 0.9995
    
    print("=" * 60)
    print("Training SentencePiece BPE Models")
    print("=" * 60)
    print(f"Source: {src_lang} -> {train_src}")
    print(f"Target: {tgt_lang} -> {train_tgt}")
    print(f"Output directory: {vocab_dir}")
    print("=" * 60)
    
    # Train source model
    print("\n[1/2] Training source language model...")
    src_model_prefix = os.path.join(vocab_dir, "src")
    train_sentencepiece(
        input_file=train_src,
        model_prefix=src_model_prefix,
        vocab_size=src_vocab_size,
        model_type=args.model_type,
        character_coverage=src_coverage
    )
    
    # Train target model
    print("\n[2/2] Training target language model...")
    tgt_model_prefix = os.path.join(vocab_dir, "tgt")
    train_sentencepiece(
        input_file=train_tgt,
        model_prefix=tgt_model_prefix,
        vocab_size=tgt_vocab_size,
        model_type=args.model_type,
        character_coverage=tgt_coverage
    )
    
    print("\n" + "=" * 60)
    print("✓ BPE training completed!")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  {src_model_prefix}.model")
    print(f"  {src_model_prefix}.vocab")
    print(f"  {tgt_model_prefix}.model")
    print(f"  {tgt_model_prefix}.vocab")
    print(f"\nYou can now run training with BPE tokenization:")
    print(f"  python scripts/train.py --config {args.config}")


if __name__ == "__main__":
    main()

"""
Main Training Script
Train the Transformer model for machine translation.

Supports both word-level and BPE tokenization.
For BPE, run train_bpe.py first to create SentencePiece models.
"""
import argparse
import os
import sys
import logging

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.helpers import set_seed, get_device, get_model_summary
from src.data.vocabulary import Vocabulary
from src.data.dataset import TranslationDataset, create_dataloader
from src.data.sp_vocab import SentencePieceVocab
from src.models.transformer import Transformer
from src.training.trainer import Trainer
from typing import Tuple, Optional


def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer MT model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)"
    )
    return parser.parse_args()


def build_vocabulary(config: dict, logger: logging.Logger):
    """Build or load vocabulary (word-level or BPE)."""
    vocab_config = config['vocab']
    vocab_dir = config['paths']['vocab_dir']
    os.makedirs(vocab_dir, exist_ok=True)
    
    use_bpe = vocab_config.get('tokenization') == 'bpe'
    
    if use_bpe:
        # BPE: Load SentencePiece models
        src_model_path = os.path.join(vocab_dir, 'src.model')
        tgt_model_path = os.path.join(vocab_dir, 'tgt.model')
        
        if not os.path.exists(src_model_path) or not os.path.exists(tgt_model_path):
            logger.error("=" * 60)
            logger.error("BPE tokenization requested but SentencePiece models not found!")
            logger.error(f"Expected: {src_model_path}")
            logger.error(f"Expected: {tgt_model_path}")
            logger.error("")
            logger.error("Please run train_bpe.py first:")
            logger.error(f"  python scripts/train_bpe.py --config {config.get('_config_path', 'your_config.yaml')}")
            logger.error("=" * 60)
            raise FileNotFoundError(f"SentencePiece models not found in {vocab_dir}")
        
        logger.info("Loading SentencePiece BPE vocabularies...")
        src_vocab = SentencePieceVocab(src_model_path)
        tgt_vocab = SentencePieceVocab(tgt_model_path)
        
        logger.info(f"Source vocabulary size: {len(src_vocab)} (BPE)")
        logger.info(f"Target vocabulary size: {len(tgt_vocab)} (BPE)")
        
    else:
        # Word-level: Build or load JSON vocabularies
        src_vocab_path = os.path.join(vocab_dir, 'src_vocab.json')
        tgt_vocab_path = os.path.join(vocab_dir, 'tgt_vocab.json')
        
        if os.path.exists(src_vocab_path) and os.path.exists(tgt_vocab_path):
            logger.info("Loading existing word-level vocabularies...")
            src_vocab = Vocabulary.load(src_vocab_path)
            tgt_vocab = Vocabulary.load(tgt_vocab_path)
        else:
            logger.info("Building word-level vocabularies from data...")
            
            # Build source vocabulary
            src_vocab = Vocabulary(
                min_freq=vocab_config.get('min_freq', 2),
                max_size=vocab_config.get('src_vocab_size', 32000)
            )
            src_vocab.build_from_file(config['data']['train_src'])
            src_vocab.save(src_vocab_path)
            
            # Build target vocabulary
            tgt_vocab = Vocabulary(
                min_freq=vocab_config.get('min_freq', 2),
                max_size=vocab_config.get('tgt_vocab_size', 32000)
            )
            tgt_vocab.build_from_file(config['data']['train_tgt'])
            tgt_vocab.save(tgt_vocab_path)
        
        logger.info(f"Source vocabulary size: {len(src_vocab)} (word-level)")
        logger.info(f"Target vocabulary size: {len(tgt_vocab)} (word-level)")
    
    return src_vocab, tgt_vocab


def create_datasets(config: dict, src_vocab, tgt_vocab, logger: logging.Logger):
    """Create train and validation datasets."""
    import os
    from torch.utils.data import random_split
    data_config = config['data']
    vocab_config = config.get('vocab', {})
    use_bpe = vocab_config.get('tokenization') == 'bpe'

    max_len = data_config.get('max_seq_length', 128)

    def make_ds(src_path, tgt_path):
        if use_bpe:
            # For BPE: SentencePieceVocab handles both tokenization AND vocabulary
            # Pass the vocab as the tokenizer (it has encode method)
            return TranslationDataset(
                src_file=src_path,
                tgt_file=tgt_path,
                src_vocab=src_vocab,
                tgt_vocab=tgt_vocab,
                src_tokenizer=src_vocab,  # SentencePieceVocab acts as tokenizer
                tgt_tokenizer=tgt_vocab,  # SentencePieceVocab acts as tokenizer
                max_length=max_len
            )
        else:
            # Word-level: no external tokenizer
            return TranslationDataset(
                src_file=src_path,
                tgt_file=tgt_path,
                src_vocab=src_vocab,
                tgt_vocab=tgt_vocab,
                src_tokenizer=None,
                tgt_tokenizer=None,
                max_length=max_len
            )

    # Check if validation files exist
    val_src_exists = os.path.exists(data_config.get('val_src', ''))
    val_tgt_exists = os.path.exists(data_config.get('val_tgt', ''))

    if val_src_exists and val_tgt_exists:
        logger.info("Creating training dataset...")
        train_dataset = make_ds(data_config['train_src'], data_config['train_tgt'])

        logger.info("Creating validation dataset...")
        val_dataset = make_ds(data_config['val_src'], data_config['val_tgt'])
    else:
        logger.info("Validation files not found. Splitting training data...")
        logger.info("Creating full training dataset...")
        full_dataset = make_ds(data_config['train_src'], data_config['train_tgt'])

        # Split: 90% train, 10% validation
        val_split = data_config.get('val_split', 0.1)
        total_size = len(full_dataset)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size

        logger.info(f"Splitting {total_size} samples: {train_size} train, {val_size} val")
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(config.get('seed', 42))
        )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    log_dir = config['paths']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(
        "train",
        log_file=os.path.join(log_dir, "train.log")
    )
    
    logger.info("=" * 60)
    logger.info("Starting training")
    logger.info("=" * 60)
    
    # Set seed for reproducibility
    seed = config.get('seed', 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")
    
    # Get device
    device = get_device(args.device or config.get('device'))
    logger.info(f"Using device: {device}")
    
    # Build vocabularies
    src_vocab, tgt_vocab = build_vocabulary(config, logger)
    
    # Update config with actual vocab sizes
    config['vocab']['src_vocab_size'] = len(src_vocab)
    config['vocab']['tgt_vocab_size'] = len(tgt_vocab)
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(config, src_vocab, tgt_vocab, logger)
    
    # Create data loaders
    train_config = config['training']
    train_loader = create_dataloader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        pad_idx=src_vocab.pad_idx
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        pad_idx=src_vocab.pad_idx
    )
    
    # Create model
    model_config = config['model']
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=model_config['d_model'],
        n_heads=model_config['n_heads'],
        n_encoder_layers=model_config['n_encoder_layers'],
        n_decoder_layers=model_config['n_decoder_layers'],
        d_ff=model_config['d_ff'],
        dropout=model_config['dropout'],
        max_seq_length=model_config['max_seq_length'],
        pad_idx=src_vocab.pad_idx
    )
    
    logger.info(get_model_summary(model))
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        tgt_vocab=tgt_vocab
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()

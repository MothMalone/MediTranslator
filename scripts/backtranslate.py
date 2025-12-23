"""
Back-Translation Script for Data Augmentation

Generates synthetic training data by:
1. Taking monolingual English data
2. Translating English → Vietnamese using a trained En→Vi model
3. Pairing synthetic Vietnamese with original English

This augments the original training data and improves model robustness.

Usage:
    python scripts/backtranslate.py --config experiments/iwslt_v4_vi2en_lora/config.yaml \
        --model experiments/iwslt_v2_en2vi/checkpoints/best_model.pt \
        --input data/monolingual_en.txt \
        --output data/backtranslation/synthetic_train
"""

import argparse
import os
import sys
import torch
import logging
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.helpers import set_seed, get_device
from src.models.transformer import Transformer
from src.data.sp_vocab import SentencePieceVocab
from src.inference.beam_search import BeamSearch
import sentencepiece as spm


def load_model_and_vocab(checkpoint_path: str, device: str, logger: logging.Logger) -> Tuple[Transformer, SentencePieceVocab, SentencePieceVocab]:
    """Load model and vocabularies from checkpoint."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct model from config
    config = checkpoint['config']
    model_config = config['model']
    
    model = Transformer(
        src_vocab_size=checkpoint['src_vocab_size'],
        tgt_vocab_size=checkpoint['tgt_vocab_size'],
        d_model=model_config['d_model'],
        n_heads=model_config['n_heads'],
        n_encoder_layers=model_config['n_encoder_layers'],
        n_decoder_layers=model_config['n_decoder_layers'],
        d_ff=model_config['d_ff'],
        dropout=model_config.get('dropout', 0.1),
        max_seq_length=model_config['max_seq_length']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load vocabularies
    vocab_dir = config['paths']['vocab_dir']
    src_vocab = SentencePieceVocab(os.path.join(vocab_dir, 'src.model'))
    tgt_vocab = SentencePieceVocab(os.path.join(vocab_dir, 'tgt.model'))
    
    logger.info(f"Source vocab size: {len(src_vocab)}")
    logger.info(f"Target vocab size: {len(tgt_vocab)}")
    
    return model, src_vocab, tgt_vocab


def backtranslate_batch(
    texts: List[str],
    model: Transformer,
    src_vocab: SentencePieceVocab,
    tgt_vocab: SentencePieceVocab,
    device: str,
    beam_size: int = 5,
    max_length: int = 128
) -> List[str]:
    """
    Translate a batch of texts using the model.
    
    Args:
        texts: List of source language texts
        model: Transformer model
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        device: Device to use
        beam_size: Beam search size
        max_length: Maximum translation length
    
    Returns:
        List of translations
    """
    # Tokenize
    tokenized = [src_vocab.encode(text) for text in texts]
    
    # Find max length in batch
    max_src_len = max(len(tokens) for tokens in tokenized)
    
    # Pad batch
    batch_tokens = []
    attention_masks = []
    for tokens in tokenized:
        padded = tokens + [src_vocab.pad_id()] * (max_src_len - len(tokens))
        mask = [1] * len(tokens) + [0] * (max_src_len - len(tokens))
        batch_tokens.append(padded)
        attention_masks.append(mask)
    
    # Convert to tensors
    src_ids = torch.tensor(batch_tokens, dtype=torch.long, device=device)
    src_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)
    
    # Encode
    with torch.no_grad():
        encoder_output = model.encoder(src_ids, src_mask)
    
    # Beam search decode
    beam_searcher = BeamSearch(
        model=model,
        src_ids=src_ids,
        src_mask=src_mask,
        encoder_output=encoder_output,
        beam_size=beam_size,
        max_length=max_length,
        device=device,
        vocab=tgt_vocab
    )
    
    translations = beam_searcher.search()
    
    # Decode to text
    results = []
    for tokens in translations:
        # Remove EOS and pad tokens
        tokens = [t for t in tokens if t not in [tgt_vocab.eos_id(), tgt_vocab.pad_id()]]
        text = tgt_vocab.decode(tokens)
        results.append(text)
    
    return results


def backtranslate_file(
    input_file: str,
    output_prefix: str,
    model: Transformer,
    src_vocab: SentencePieceVocab,
    tgt_vocab: SentencePieceVocab,
    device: str,
    batch_size: int = 32,
    beam_size: int = 5,
    logger: logging.Logger = None
) -> None:
    """
    Back-translate a file of monolingual data.
    
    Args:
        input_file: Path to monolingual source file (English)
        output_prefix: Prefix for output files (will create .src and .tgt)
        model: Translator model (En→Vi in this case)
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        device: Device to use
        batch_size: Batch size for translation
        beam_size: Beam search size
        logger: Logger instance
    """
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Read input file
    logger.info(f"Reading monolingual data from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Read {len(lines)} lines")
    
    # Back-translate in batches
    src_output = []
    tgt_output = []
    
    for i in tqdm(range(0, len(lines), batch_size), desc="Back-translating"):
        batch = lines[i:i+batch_size]
        
        try:
            translations = backtranslate_batch(
                batch,
                model,
                src_vocab,
                tgt_vocab,
                device,
                beam_size=beam_size
            )
            
            # Store: synthetic source (original target language text)
            # Store: synthetic target (original source language text)
            for original, translated in zip(batch, translations):
                if translated.strip():  # Skip empty translations
                    src_output.append(translated)  # Synthetic Vietnamese
                    tgt_output.append(original)     # Original English
        
        except Exception as e:
            logger.warning(f"Error translating batch {i//batch_size}: {e}")
            continue
    
    # Write output files
    os.makedirs(os.path.dirname(output_prefix) if os.path.dirname(output_prefix) else '.', exist_ok=True)
    
    src_path = f"{output_prefix}.vi.txt"
    tgt_path = f"{output_prefix}.en.txt"
    
    logger.info(f"Writing synthetic source to {src_path}")
    with open(src_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(src_output) + '\n')
    
    logger.info(f"Writing synthetic target to {tgt_path}")
    with open(tgt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tgt_output) + '\n')
    
    logger.info(f"Back-translation complete: {len(src_output)} pairs generated")
    logger.info(f"  Source: {src_path}")
    logger.info(f"  Target: {tgt_path}")


def main():
    parser = argparse.ArgumentParser(description="Back-translate data for augmentation")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to checkpoint of translation model (En→Vi)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to monolingual English file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/backtranslation/synthetic_train",
        help="Output prefix for synthetic data"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for translation"
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam search size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    args = parser.parse_args()
    
    # Setup logger
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    logger = setup_logger(
        "backtranslate",
        log_file=os.path.join(os.path.dirname(args.output), 'backtranslate.log')
    )
    
    device = get_device(args.device)
    set_seed(42)
    
    # Load model and vocab
    model, src_vocab, tgt_vocab = load_model_and_vocab(args.model, device, logger)
    
    # Back-translate
    backtranslate_file(
        args.input,
        args.output,
        model,
        src_vocab,
        tgt_vocab,
        device,
        batch_size=args.batch_size,
        beam_size=args.beam_size,
        logger=logger
    )


if __name__ == "__main__":
    main()

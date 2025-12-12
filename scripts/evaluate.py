"""
Evaluation Script
Evaluate trained model on test set.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.helpers import get_device
from src.data.vocabulary import Vocabulary
from src.data.dataset import TranslationDataset, create_dataloader
from src.models.transformer import Transformer
from src.inference.translator import Translator
from src.evaluation.bleu import BLEUCalculator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Transformer MT model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--test_src",
        type=str,
        default=None,
        help="Path to test source file (overrides config)"
    )
    parser.add_argument(
        "--test_tgt",
        type=str,
        default=None,
        help="Path to test target/reference file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.txt",
        help="Path to output file for predictions"
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help="Beam size for beam search"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum output length"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use"
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, config: dict, device: torch.device):
    """Load model from checkpoint."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load vocabularies
    vocab_dir = config['paths']['vocab_dir']
    src_vocab = Vocabulary.load(os.path.join(vocab_dir, 'src_vocab.json'))
    tgt_vocab = Vocabulary.load(os.path.join(vocab_dir, 'tgt_vocab.json'))
    
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
        pad_idx=0
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, src_vocab, tgt_vocab


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logger("evaluate")
    
    logger.info("=" * 60)
    logger.info("Starting evaluation")
    logger.info("=" * 60)
    
    # Get device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model, src_vocab, tgt_vocab = load_model(args.checkpoint, config, device)
    
    # Create translator
    translator = Translator(
        model=model,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        device=device,
        decoding_method='beam',
        beam_size=args.beam_size,
        max_length=args.max_length
    )
    
    # Load test data
    test_src = args.test_src or config['data'].get('test_src')
    test_tgt = args.test_tgt or config['data'].get('test_tgt')
    
    if not test_src:
        logger.error("No test source file specified")
        return
    
    logger.info(f"Loading test data from {test_src}")
    
    with open(test_src, 'r', encoding='utf-8') as f:
        source_sentences = [line.strip() for line in f]
    
    references = None
    if test_tgt and os.path.exists(test_tgt):
        with open(test_tgt, 'r', encoding='utf-8') as f:
            references = [line.strip() for line in f]
    
    logger.info(f"Translating {len(source_sentences)} sentences...")
    
    # Translate
    predictions = []
    for sentence in tqdm(source_sentences, desc="Translating"):
        translation = translator.translate(sentence)
        predictions.append(translation)
    
    # Save predictions
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(pred + '\n')
    
    logger.info(f"Saved predictions to {args.output}")
    
    # Calculate BLEU if references available
    if references:
        logger.info("Calculating BLEU score...")
        
        bleu_calculator = BLEUCalculator()
        result = bleu_calculator.calculate(predictions, references)
        
        logger.info("=" * 40)
        logger.info("Evaluation Results")
        logger.info("=" * 40)
        logger.info(f"BLEU Score: {result['bleu']:.2f}")
        
        if 'precisions' in result:
            for i, p in enumerate(result['precisions'], 1):
                logger.info(f"  {i}-gram precision: {p:.1f}")
            logger.info(f"  Brevity Penalty: {result['bp']:.3f}")
            logger.info(f"  Length Ratio: {result['ratio']:.3f}")
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()

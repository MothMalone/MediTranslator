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
from src.data.sp_vocab import SentencePieceVocab
from src.data.dataset import TranslationDataset, create_dataloader
from src.models.transformer import Transformer
from src.inference.translator import Translator
from src.evaluation.bleu import BLEUCalculator

try:
    from evaluate import load as load_metric
    HF_EVALUATE_AVAILABLE = True
except ImportError:
    HF_EVALUATE_AVAILABLE = False
    load_metric = None


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
        "--metrics",
        nargs='+',
        default=['bleu'],
        choices=['bleu', 'ter', 'meteor', 'all'],
        help="Metrics to compute (default: bleu)"
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
    
    # Load vocabularies - check for expanded_vocab_dir first (vocab expansion models), then vocab_dir
    vocab_dir = config['paths'].get('expanded_vocab_dir') or config['paths']['vocab_dir']
    vocab_config = config.get('vocab', {})
    use_bpe = vocab_config.get('tokenization') == 'bpe'
    
    if use_bpe:
        # BPE: Load SentencePiece models
        src_model_path = os.path.join(vocab_dir, 'src.model')
        tgt_model_path = os.path.join(vocab_dir, 'tgt.model')
        
        if os.path.exists(src_model_path) and os.path.exists(tgt_model_path):
            src_vocab = SentencePieceVocab(src_model_path)
            tgt_vocab = SentencePieceVocab(tgt_model_path)
        else:
            raise FileNotFoundError(f"SentencePiece models not found in {vocab_dir}")
    else:
        # Word-level vocabularies
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
    
    # Calculate metrics if references available
    if references:
        logger.info("=" * 60)
        logger.info("Evaluation Results")
        logger.info("=" * 60)
        
        # Determine which metrics to compute
        metrics_to_compute = args.metrics
        if 'all' in metrics_to_compute:
            metrics_to_compute = ['bleu', 'ter', 'meteor']
        
        results = {}
        
        # BLEU
        if 'bleu' in metrics_to_compute:
            logger.info("\nCalculating BLEU score...")
            bleu_calculator = BLEUCalculator()
            bleu_result = bleu_calculator.calculate(predictions, references)
            results['bleu'] = bleu_result
            
            logger.info(f"BLEU Score: {bleu_result['bleu']:.2f}")
            if 'precisions' in bleu_result:
                for i, p in enumerate(bleu_result['precisions'], 1):
                    logger.info(f"  {i}-gram precision: {p:.1f}")
                logger.info(f"  Brevity Penalty: {bleu_result['bp']:.3f}")
                logger.info(f"  Length Ratio: {bleu_result['ratio']:.3f}")
        
        # TER (Translation Edit Rate)
        if 'ter' in metrics_to_compute:
            if HF_EVALUATE_AVAILABLE:
                logger.info("\nCalculating TER score...")
                try:
                    ter = load_metric('ter')
                    # TER expects list of references for each prediction
                    ter_result = ter.compute(
                        predictions=predictions,
                        references=[[ref] for ref in references]
                    )
                    results['ter'] = ter_result['score']
                    logger.info(f"TER Score: {ter_result['score']:.2f}")
                except Exception as e:
                    logger.warning(f"Failed to calculate TER: {e}")
            else:
                logger.warning("TER metric requires 'evaluate' library. Install with: pip install evaluate")
        
        # METEOR
        if 'meteor' in metrics_to_compute:
            if HF_EVALUATE_AVAILABLE:
                logger.info("\nCalculating METEOR score...")
                try:
                    meteor = load_metric('meteor')
                    meteor_result = meteor.compute(
                        predictions=predictions,
                        references=references
                    )
                    results['meteor'] = meteor_result['meteor']
                    logger.info(f"METEOR Score: {meteor_result['meteor']:.4f}")
                except Exception as e:
                    logger.warning(f"Failed to calculate METEOR: {e}")
            else:
                logger.warning("METEOR metric requires 'evaluate' library. Install with: pip install evaluate")
        
        # Save results to file
        results_file = args.output.replace('.txt', '_results.txt')
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Evaluation Results\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model: {args.checkpoint}\n")
            f.write(f"Test source: {test_src}\n")
            f.write(f"Test target: {test_tgt}\n")
            f.write(f"Predictions: {args.output}\n\n")
            
            if 'bleu' in results:
                f.write(f"BLEU Score: {results['bleu']['bleu']:.2f}\n")
                if 'precisions' in results['bleu']:
                    for i, p in enumerate(results['bleu']['precisions'], 1):
                        f.write(f"  {i}-gram precision: {p:.1f}\n")
                    f.write(f"  Brevity Penalty: {results['bleu']['bp']:.3f}\n")
                    f.write(f"  Length Ratio: {results['bleu']['ratio']:.3f}\n")
                f.write("\n")
            
            if 'ter' in results:
                f.write(f"TER Score: {results['ter']:.2f}\n\n")
            
            if 'meteor' in results:
                f.write(f"METEOR Score: {results['meteor']:.4f}\n\n")
        
        logger.info(f"\nResults saved to {results_file}")
        logger.info("=" * 60)
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()

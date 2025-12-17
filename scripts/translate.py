"""
Translation Script
Translate text using trained model.

Supports both word-level (JSON) and BPE (SentencePiece) vocabularies.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.config import load_config
from src.utils.helpers import get_device
from src.data.vocabulary import Vocabulary
from src.data.sp_vocab import SentencePieceVocab
from src.models.transformer import Transformer
from src.inference.translator import Translator


def load_vocabs(vocab_dir: str, tokenization: str = "word"):
    """
    Load vocabularies based on tokenization type.
    
    Args:
        vocab_dir: Directory containing vocabulary files
        tokenization: "bpe" for SentencePiece, "word" for JSON vocabularies
        
    Returns:
        (src_vocab, tgt_vocab) tuple
    """
    if tokenization == "bpe":
        # Load SentencePiece models
        src_sp_path = os.path.join(vocab_dir, 'src.model')
        tgt_sp_path = os.path.join(vocab_dir, 'tgt.model')
        
        if os.path.exists(src_sp_path) and os.path.exists(tgt_sp_path):
            print(f"Loading BPE (SentencePiece) vocabularies from {vocab_dir}")
            src_vocab = SentencePieceVocab(src_sp_path)
            tgt_vocab = SentencePieceVocab(tgt_sp_path)
            return src_vocab, tgt_vocab
        else:
            raise FileNotFoundError(
                f"BPE tokenization configured but SentencePiece models not found in {vocab_dir}.\n"
                f"Expected: {src_sp_path} and {tgt_sp_path}\n"
                f"Run: python scripts/train_bpe.py --config <your_config.yaml>"
            )
    else:
        # Load word-level JSON vocabularies
        src_path = os.path.join(vocab_dir, 'src_vocab.json')
        tgt_path = os.path.join(vocab_dir, 'tgt_vocab.json')
        
        if os.path.exists(src_path) and os.path.exists(tgt_path):
            print(f"Loading word-level vocabularies from {vocab_dir}")
            src_vocab = Vocabulary.load(src_path)
            tgt_vocab = Vocabulary.load(tgt_path)
            return src_vocab, tgt_vocab
        else:
            raise FileNotFoundError(
                f"Word-level vocabularies not found in {vocab_dir}.\n"
                f"Expected: {src_path} and {tgt_path}"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Translate text")
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
        "--input",
        type=str,
        default=None,
        help="Input file to translate (or use --text)"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to translate (for single sentence)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for translations"
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help="Beam size for beam search"
    )
    parser.add_argument(
        "--decoding",
        type=str,
        choices=['greedy', 'beam'],
        default='beam',
        help="Decoding method"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode"
    )
    return parser.parse_args()


def load_translator(checkpoint_path: str, config: dict, args):
    """Load translator from checkpoint."""
    device = get_device(args.device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Determine tokenization type
    tokenization = config.get('vocab', {}).get('tokenization', 'word')
    
    # Load vocabularies - check for expanded_vocab_dir first (vocab expansion models), then vocab_dir
    vocab_dir = config['paths'].get('expanded_vocab_dir') or config['paths']['vocab_dir']
    src_vocab, tgt_vocab = load_vocabs(vocab_dir, tokenization)
    
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
    
    # Check if checkpoint has LoRA weights
    state_dict = checkpoint['model_state_dict']
    has_lora = any('lora' in key for key in state_dict.keys())
    
    if has_lora:
        print("Detected LoRA checkpoint, applying LoRA to model...")
        # Apply LoRA with same config as training
        lora_config = config.get('lora', {})
        model, _ = apply_lora_to_model(
            model,
            target_modules=lora_config.get('target_modules', ['query', 'value']),
            rank=lora_config.get('rank', 8),
            alpha=lora_config.get('alpha', 16),
            dropout=0.0  # No dropout during inference
        )
        print("LoRA applied, loading weights...")
    
    model.load_state_dict(state_dict)
    
    # Merge LoRA weights for faster inference
    if has_lora:
        print("Merging LoRA weights for inference...")
        merge_lora_weights(model)
        print("LoRA weights merged!")
    
    # Create translator
    translator = Translator(
        model=model,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        device=device,
        decoding_method=args.decoding,
        beam_size=args.beam_size
    )
    
    return translator


def interactive_mode(translator):
    """Run translator in interactive mode."""
    print("=" * 50)
    print("Interactive Translation Mode")
    print("Enter text to translate (or 'quit' to exit)")
    print("=" * 50)
    
    while True:
        try:
            text = input("\nSource: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not text:
                continue
            
            translation = translator.translate(text)
            print(f"Translation: {translation}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load translator
    print(f"Loading model from {args.checkpoint}...")
    translator = load_translator(args.checkpoint, config, args)
    print("Model loaded!")
    
    # Interactive mode
    if args.interactive:
        interactive_mode(translator)
        return
    
    # Single text translation
    if args.text:
        translation = translator.translate(args.text)
        print(f"Source: {args.text}")
        print(f"Translation: {translation}")
        return
    
    # File translation
    if args.input:
        print(f"Translating file: {args.input}")
        
        with open(args.input, 'r', encoding='utf-8') as f:
            source_lines = [line.strip() for line in f]
        
        translations = []
        for line in source_lines:
            translation = translator.translate(line)
            translations.append(translation)
        
        # Output
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                for trans in translations:
                    f.write(trans + '\n')
            print(f"Saved translations to {args.output}")
        else:
            for src, tgt in zip(source_lines, translations):
                print(f"Source: {src}")
                print(f"Translation: {tgt}")
                print()
        
        return
    
    # No input specified, show help
    print("Please specify --text, --input, or --interactive")


if __name__ == "__main__":
    main()

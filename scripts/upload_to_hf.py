"""
Upload trained models to HuggingFace Hub
"""
import argparse
import os
import sys
import shutil
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import HfApi, create_repo
from src.utils.config import load_config


def create_model_card(config: dict, experiment_name: str, bleu_score: float = None) -> str:
    """Create a model card for HuggingFace."""
    version_info = config.get('version', {})
    model_config = config.get('model', {})
    train_config = config.get('training', {})
    data_config = config.get('data', {})
    
    card = f"""---
language:
- vi
- en
tags:
- translation
- transformer
- seq2seq
license: mit
datasets:
- iwslt2015
metrics:
- bleu
---

# {version_info.get('name', experiment_name)} - Vietnamese-English Translation

## Model Description

{version_info.get('description', 'Transformer model for Vietnamese to English translation')}

This model is trained from scratch using the Transformer architecture for machine translation.

### Model Details

- **Language pair**: Vietnamese → English
- **Architecture**: Transformer (Encoder-Decoder)
- **Parameters**: 
  - d_model: {model_config.get('d_model', 512)}
  - n_heads: {model_config.get('n_heads', 8)}
  - n_encoder_layers: {model_config.get('n_encoder_layers', 6)}
  - n_decoder_layers: {model_config.get('n_decoder_layers', 6)}
  - d_ff: {model_config.get('d_ff', 2048)}
  - dropout: {model_config.get('dropout', 0.1)}

### Training Details

- **Optimizer**: {train_config.get('optimizer', 'adamw').upper()}
- **Learning Rate**: {train_config.get('learning_rate', 0.0001)}
- **Batch Size**: {train_config.get('batch_size', 32)}
- **Label Smoothing**: {train_config.get('label_smoothing', 0.0)}
- **Scheduler**: {train_config.get('scheduler', 'warmup')}
- **Dataset**: IWSLT 2015 Vi-En

### Performance

"""
    
    if bleu_score:
        card += f"- **BLEU Score**: {bleu_score:.2f}\n"
    
    card += f"""
### Improvements

"""
    
    improvements = version_info.get('improvements', [])
    if improvements:
        for imp in improvements:
            card += f"- {imp}\n"
    
    card += """
## Usage

```python
# Load model and translate
from src.models.transformer import Transformer
from src.inference.translator import Translator
from src.data.vocabulary import Vocabulary
import torch

# Load vocabularies
src_vocab = Vocabulary.load('src_vocab.json')
tgt_vocab = Vocabulary.load('tgt_vocab.json')

# Load model
model = Transformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    d_model=512,
    n_heads=8,
    n_encoder_layers=6,
    n_decoder_layers=6,
    d_ff=2048,
    dropout=0.1,
    max_seq_length=512,
    pad_idx=0
)

checkpoint = torch.load('best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Create translator
translator = Translator(
    model=model,
    src_vocab=src_vocab,
    tgt_vocab=tgt_vocab,
    device='cuda',
    decoding_method='beam',
    beam_size=5
)

# Translate
vietnamese_text = "Xin chào, bạn khỏe không?"
translation = translator.translate(vietnamese_text)
print(translation)
```

## Training Data

- **Dataset**: IWSLT 2015 Vietnamese-English parallel corpus
- **Training pairs**: ~500,000 sentence pairs
- **Validation pairs**: ~50,000 sentence pairs
- **Test pairs**: ~3,000 sentence pairs

## Limitations

- Trained specifically for Vietnamese to English translation
- Performance may vary on out-of-domain text
- Medical/technical domains may require fine-tuning

## Citation

```bibtex
@misc{nlp-transformer-mt,
  author = {MothMalone},
  title = {Transformer Machine Translation Vi-En},
  year = {2025},
  publisher = {HuggingFace},
  howpublished = {\\url{https://huggingface.co/MothMalone}}
}
```
"""
    
    return card


def prepare_model_repo(experiment_path: str, repo_name: str, bleu_score: float = None):
    """Prepare model repository for upload."""
    
    # Create temporary directory
    temp_dir = f"temp_upload_{repo_name}"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Load config
    config_path = os.path.join(experiment_path, 'config.yaml')
    config = load_config(config_path)
    
    # Copy model checkpoint
    checkpoint_src = os.path.join(experiment_path, 'checkpoints', 'best_model.pt')
    if not os.path.exists(checkpoint_src):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_src}")
    
    shutil.copy(checkpoint_src, os.path.join(temp_dir, 'best_model.pt'))
    print(f"✓ Copied checkpoint")
    
    # Copy config
    shutil.copy(config_path, os.path.join(temp_dir, 'config.yaml'))
    print(f"✓ Copied config")
    
    # Copy vocabularies
    vocab_dir = config['paths'].get('vocab_dir', 'data/vocab')
    src_vocab = os.path.join(vocab_dir, 'src_vocab.json')
    tgt_vocab = os.path.join(vocab_dir, 'tgt_vocab.json')
    
    if os.path.exists(src_vocab):
        shutil.copy(src_vocab, os.path.join(temp_dir, 'src_vocab.json'))
        print(f"✓ Copied source vocabulary")
    
    if os.path.exists(tgt_vocab):
        shutil.copy(tgt_vocab, os.path.join(temp_dir, 'tgt_vocab.json'))
        print(f"✓ Copied target vocabulary")
    
    # Create model card
    model_card = create_model_card(config, repo_name, bleu_score)
    with open(os.path.join(temp_dir, 'README.md'), 'w') as f:
        f.write(model_card)
    print(f"✓ Created model card")
    
    # Copy training metrics if available
    metrics_path = os.path.join(experiment_path, 'logs', 'metrics.json')
    if os.path.exists(metrics_path):
        shutil.copy(metrics_path, os.path.join(temp_dir, 'training_metrics.json'))
        print(f"✓ Copied training metrics")
    
    # Create requirements.txt
    requirements = """torch>=2.0.0
numpy>=1.21.0
pyyaml>=6.0
tqdm>=4.65.0
"""
    with open(os.path.join(temp_dir, 'requirements.txt'), 'w') as f:
        f.write(requirements)
    print(f"✓ Created requirements.txt")
    
    return temp_dir


def upload_to_huggingface(temp_dir: str, repo_name: str, username: str, token: str):
    """Upload model to HuggingFace Hub."""
    
    api = HfApi()
    
    # Create repository
    repo_id = f"{username}/{repo_name}"
    
    try:
        create_repo(
            repo_id=repo_id,
            token=token,
            exist_ok=True,
            private=False
        )
        print(f"✓ Created/verified repository: {repo_id}")
    except Exception as e:
        print(f"⚠ Repository creation: {e}")
    
    # Upload all files
    try:
        api.upload_folder(
            folder_path=temp_dir,
            repo_id=repo_id,
            token=token,
            commit_message=f"Upload {repo_name} model"
        )
        print(f"✓ Uploaded model to https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"✗ Upload failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Path to experiment directory (e.g., experiments/v1_baseline)"
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        required=True,
        help="Repository name on HuggingFace (e.g., transformer-vi-en-v1)"
    )
    parser.add_argument(
        "--username",
        type=str,
        default="MothMalone",
        help="HuggingFace username"
    )
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="HuggingFace API token"
    )
    parser.add_argument(
        "--bleu",
        type=float,
        default=None,
        help="BLEU score to include in model card"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove temporary directory after upload"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Uploading {args.experiment} to HuggingFace")
    print("=" * 60)
    
    # Prepare repository
    print("\n1. Preparing model repository...")
    temp_dir = prepare_model_repo(args.experiment, args.repo_name, args.bleu)
    
    # Upload to HuggingFace
    print(f"\n2. Uploading to HuggingFace ({args.username}/{args.repo_name})...")
    upload_to_huggingface(temp_dir, args.repo_name, args.username, args.token)
    
    # Cleanup
    if args.cleanup:
        print("\n3. Cleaning up...")
        shutil.rmtree(temp_dir)
        print(f"✓ Removed temporary directory")
    else:
        print(f"\n✓ Temporary files kept in: {temp_dir}")
    
    print("\n" + "=" * 60)
    print(f"✅ Model uploaded successfully!")
    print(f"🔗 https://huggingface.co/{args.username}/{args.repo_name}")
    print("=" * 60)


if __name__ == "__main__":
    main()

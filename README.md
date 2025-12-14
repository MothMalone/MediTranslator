# MediTranslator

A Vietnamese-English neural machine translation system built from scratch using Transformer architecture. This project implements the classic "Attention is All You Need" paper and applies it to both general and medical domain translation.


## Quick Start

```bash
# Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train a model
python scripts/train.py --config experiments/v1_baseline/config.yaml. # change to or tweak the desired config inside experiments

# Translate something
python scripts/translate.py --checkpoint checkpoints/best_model.pt --input "Xin chào"
```

## Project Structure

```
MainProblem/
├── data/
│   ├── raw/           # Training data (IWSLT dataset)
│   └── vocab/         # Vocabulary files for source/target languages
│
├── src/
│   ├── data/          # Dataset loading, tokenization, vocab building
│   ├── models/        # Transformer implementation
│   │   ├── attention.py          # Multi-head attention
│   │   ├── encoder.py            # Encoder stack
│   │   ├── decoder.py            # Decoder stack
│   │   ├── transformer.py        # Full model
│   │   ├── lora.py               # LoRA (Low-Rank Adaptation)
│   │   └── positional_encoding.py
│   ├── training/      # Training loop, loss, metrics
│   ├── inference/     # Beam search, greedy decoding
│   ├── evaluation/    # BLEU score calculation
│   └── utils/         # Config, logging, helpers
│
├── experiments/       # Different model configurations
│   ├── v1_baseline/   # Standard Transformer Base (Vi→En)
│   ├── v2_en2vi/      # Reverse direction (En→Vi)
│   ├── v2_improved/   # With label smoothing & beam search
│   ├── v3_en2vi/      # Larger model (Transformer Big)
│   ├── v3_vi2en/      # Optimized Vi→En
│   ├── medical_en2vi/ # Medical domain fine-tuning (En→Vi)
│   └── medical_vi2en/ # Medical domain fine-tuning (Vi→En)
│
└── scripts/
    ├── train.py              # Training script (from scratch)
    ├── train_lora.py         # LoRA fine-tuning (parameter-efficient)
    ├── finetune_medical.py   # Full fine-tuning with advanced techniques
    ├── evaluate.py           # Calculate BLEU scores
    ├── translate.py          # Interactive translation
    └── calculate_bleu.py     # Standalone BLEU calculator
```

## Model Configurations

The project includes several experiment configs to test different setups:

### Base Models
- **v1_baseline**: Standard Transformer Base (512d, 6 layers) with greedy decoding
- **v2_improved**: Adds label smoothing and beam search
- **v3**: Transformer Big (1024d, 16 heads) with larger vocabulary and BPE tokenization

### Medical Domain Fine-tuning
- **medical_en2vi**: Medical En→Vi with LoRA fine-tuning (rank-16, all attention + FFN)
- **medical_vi2en**: Medical Vi→En with LoRA fine-tuning (rank-16, all attention + FFN)

All configs are in `experiments/*/config.yaml` and can be easily modified.

## Training

### Training from Scratch

Each experiment has its own config file. Training saves checkpoints and logs automatically:

```bash
# Train baseline model
python scripts/train.py --config experiments/v1_baseline/config.yaml

# Resume from checkpoint
python scripts/train.py --config experiments/v1_baseline/config.yaml --resume checkpoints/checkpoint_epoch_10.pt
```

### Fine-tuning on Medical Domain

For medical domain adaptation, use the specialized fine-tuning scripts:

#### LoRA Fine-tuning (Recommended - Parameter Efficient)

LoRA (Low-Rank Adaptation) allows efficient fine-tuning by only training low-rank adapter matrices:

```bash
# Fine-tune medical En→Vi with LoRA
python scripts/train_lora.py --config experiments/medical_en2vi/config.yaml

# Fine-tune medical Vi→En with LoRA
python scripts/train_lora.py --config experiments/medical_vi2en/config.yaml
```

**LoRA Benefits:**
- 🚀 **Efficiency**: Only ~2-5% of parameters are trainable
- 💾 **Storage**: Save only LoRA weights (~10MB vs ~500MB)
- ⚡ **Speed**: Faster training with lower memory usage
- 🎯 **Performance**: Competitive with full fine-tuning

**LoRA Configuration** (in config.yaml):
```yaml
lora:
  enabled: true
  rank: 16              # Low-rank dimension (higher = more capacity)
  alpha: 32             # Scaling factor (typically 2x rank)
  dropout: 0.05         # LoRA dropout for regularization
  target_modules:       # Which layers to adapt
    - "query"           # Query projections
    - "key"             # Key projections
    - "value"           # Value projections
    - "output"          # Output projections
    - "fc1"             # First feedforward layer
    - "fc2"             # Second feedforward layer
```

#### Full Fine-tuning (Advanced Techniques)

For maximum performance with Tier 1 + Tier 2 optimizations:

```bash
# Full fine-tuning with discriminative LR, SWA, cosine annealing
python scripts/finetune_medical.py --config experiments/medical_en2vi/config.yaml
python scripts/finetune_medical.py --config experiments/medical_vi2en/config.yaml
```

**Advanced Features:**
- **Discriminative Learning Rates**: Different LR for embeddings/encoder/decoder
- **Cosine Annealing**: Smooth LR decay with warmup
- **Stochastic Weight Averaging (SWA)**: Averaging checkpoints for better generalization
- **Label Smoothing**: Prevents overconfidence on training data

**Fine-tuning Configuration** (in config.yaml):
```yaml
training:
  # Load pretrained base model
  resume_from: "experiments/v2_en2vi/checkpoints/best_model.pt"
  
  # Fine-tuning hyperparameters
  batch_size: 16
  epochs: 25
  learning_rate: 0.00005  # Lower LR for fine-tuning (5e-5)
  
  # Discriminative learning rates (Tier 2)
  use_discriminative_lr: true
  discriminative_lr_groups:
    embeddings: 0.5       # 2.5e-5
    encoder: 0.7          # 3.5e-5
    decoder: 1.0          # 5e-5
  
  # Scheduler with warmup
  scheduler: "cosine_warmup"
  warmup_steps: 1000
  min_lr: 1.0e-6
  
  # Stochastic Weight Averaging (Tier 2)
  use_swa: true
  swa_start_epoch: 20
  swa_lr: 0.00002
  
  # Label smoothing
  label_smoothing: 0.1
```

Training logs go to TensorBoard and optionally Weights & Biases (set your API key in the config).

## Evaluation

Calculate BLEU scores on test data:

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```

Or evaluate specific files:

```bash
python scripts/calculate_bleu.py --reference data/raw/public_test.en.txt --hypothesis predictions.txt
```

## Implementation Details

The Transformer is implemented following the original paper:
- Scaled dot-product attention with multi-head mechanism
- Sinusoidal positional encodings
- Layer normalization and residual connections
- Label smoothing for better generalization
- Beam search decoding for inference

The code is modular so you can swap out components easily (e.g., different attention mechanisms or decoding strategies).

## Data

The project uses parallel Vietnamese-English text. Training data is in `data/raw/`:
- `train.vi.txt` / `train.en.txt` - Training pairs
- `public_test.vi.txt` / `public_test.en.txt` - Test set

Vocabulary is built using BPE or word-level tokenization depending on the config.

## Medical Domain Fine-tuning (VLSP 2025)

The project includes specialized medical domain fine-tuning capabilities for the VLSP 2025 shared task.

### Medical Dataset

The medical dataset contains 500K parallel sentences from medical literature:
- **Domain**: Medical terminology, clinical notes, healthcare documents
- **Size**: ~500K training pairs, ~25K validation (5% split)
- **Languages**: Vietnamese ↔ English
- **Source**: VLSP Medical Translation Dataset

### Fine-tuning Approach

Two fine-tuning strategies are provided:

#### 1. **LoRA Fine-tuning** (Recommended)
- Parameter-efficient: Only 2-5% parameters trained
- Fast convergence: 25 epochs with early stopping
- Rank-16 adapters on all attention + feedforward layers
- Optimal for limited compute resources

#### 2. **Full Fine-tuning** (Maximum Performance)
- Complete model adaptation with advanced techniques
- Discriminative learning rates per layer group
- Stochastic Weight Averaging for robust checkpoints
- Cosine annealing with linear warmup

### Quick Start - Medical Fine-tuning

```bash
# 1. Ensure you have a pretrained base model
#    (e.g., experiments/v2_en2vi/checkpoints/best_model.pt)

# 2. Update the config paths in experiments/medical_*/config.yaml:
#    - resume_from: path to your base model
#    - train_src/train_tgt: path to medical dataset
#    - vocab_dir: path to base model vocabulary

# 3. Run LoRA fine-tuning (recommended)
python scripts/train_lora.py --config experiments/medical_en2vi/config.yaml

# OR run full fine-tuning with advanced techniques
python scripts/finetune_medical.py --config experiments/medical_en2vi/config.yaml

# 4. Evaluate on medical test set
python scripts/evaluate.py \
    --checkpoint experiments/medical_en2vi/checkpoints/best_model.pt \
    --test-src ../MedicalDataset_VLSP/public_test.en.txt \
    --test-tgt ../MedicalDataset_VLSP/public_test.vi.txt
```

### Medical Domain Optimizations

The medical configs include domain-specific optimizations:

- **Longer sequences**: 256 tokens (vs 128 for general domain)
- **Medical-tuned vocabulary**: Reuses general vocab with domain adaptation
- **Lower learning rate**: 5e-5 for stable fine-tuning
- **Extended training**: 25 epochs with patience-8 early stopping
- **Domain-specific inference**: Beam size 5, length penalty 0.6

### Results

Expected performance on medical test set:
- **LoRA fine-tuning**: ~35-40 BLEU (En→Vi), ~38-42 BLEU (Vi→En)
- **Full fine-tuning**: ~40-45 BLEU (En→Vi), ~42-47 BLEU (Vi→En)

*Results depend on base model quality and dataset characteristics.*

## Requirements

Full list in `requirements.txt`.

## Notes

- Training from scratch takes time - use a GPU if possible
- Start with v1_baseline to get familiar with the setup
- Beam search is slower but gives better translation quality
- Check experiment configs before training to adjust batch size for your hardware

## References

- Vaswani et al., "Attention Is All You Need" (2017)
- The Annotated Transformer (Harvard NLP)
- IWSLT Dataset

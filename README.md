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
│   └── v3_vi2en/      # Optimized Vi→En
│
└── scripts/
    ├── train.py       # Training script
    ├── evaluate.py    # Calculate BLEU scores
    └── translate.py   # Interactive translation
```

## Model Configurations

The project includes several experiment configs to test different setups:

- **v1_baseline**: Standard Transformer Base (512d, 6 layers) with greedy decoding
- **v2_improved**: Adds label smoothing and beam search
- **v3**: Transformer Big (1024d, 16 heads) with larger vocabulary and BPE tokenization

All configs are in `experiments/*/config.yaml` and can be easily modified.

## Training

Each experiment has its own config file. Training saves checkpoints and logs automatically:

```bash
# Train baseline model
python scripts/train.py --config experiments/v1_baseline/config.yaml

# Resume from checkpoint
python scripts/train.py --config experiments/v1_baseline/config.yaml --resume checkpoints/checkpoint_epoch_10.pt
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

## Medical Domain (VLSP 2025)

The same architecture can be fine-tuned for medical translation. Medical data preprocessing and domain adaptation code will be added for the VLSP shared task.

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

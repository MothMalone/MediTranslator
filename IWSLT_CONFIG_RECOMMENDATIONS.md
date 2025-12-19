# IWSLT Config Analysis & Recommendations

## Dataset Overview: IWSLT'15 English-Vietnamese

**Available files in `data/raw_IWSLT'15_en-vi/`:**

- ✅ `train.{en,vi}.txt` - **133,318 sentence pairs** (main training data)
- ✅ `tst2012.{en,vi}.txt` - **1,554 sentence pairs** (validation)
- ✅ `tst2013.{en,vi}.txt` - **1,269 sentence pairs** (test)
- ❌ `val.{en,vi}.txt` - **NOT AVAILABLE** (referenced in v1 and v3 configs)

**Dataset characteristics:**

- Small to medium-sized dataset (~133K pairs)
- TED talk transcriptions (conversational, varied topics)
- Official test sets available (tst2012, tst2013)
- No dedicated validation split - must use test set or split from training

---

## Config Comparison

| Feature              | V1 (Baseline)   | V2 (Optimized) ✅ | V3 (Large)          |
| -------------------- | --------------- | ----------------- | ------------------- |
| **Model Size**       | Base (512-dim)  | Base (512-dim)    | Big (1024-dim)      |
| **Vocab Size**       | 40K             | **16K BPE**       | 50K BPE             |
| **Batch Size**       | 32              | **64**            | 8 (accum=8, eff=64) |
| **Learning Rate**    | 0.0001          | **0.001**         | 0.0001              |
| **Epochs**           | 5               | **30**            | 5                   |
| **Scheduler**        | warmup          | **cosine**        | cosine              |
| **Optimizer**        | adam            | **adamw**         | adamw               |
| **Label Smoothing**  | 0.0             | **0.1**           | 0.1                 |
| **Mixed Precision**  | ❌              | **✅**            | ✅                  |
| **Early Stopping**   | ❌              | **12 epochs**     | 10 epochs           |
| **Validation Split** | ❌ Missing file | ✅ tst2012        | ❌ Missing file     |
| **Beam Size**        | 1 (greedy)      | **5**             | 10                  |

---

## Issues Found

### ❌ V1 Config Issues:

1. **Missing validation files**: References `val.en.txt` and `val.vi.txt` that don't exist
2. **Too few epochs**: Only 5 epochs for 133K samples is insufficient
3. **No optimizations**: No label smoothing, basic Adam, greedy search only
4. **Large vocab**: 40K word-level vocab is too big for 133K pairs

### ✅ V2 Config: Best Configuration

- Uses correct validation file (`tst2012.en.txt`)
- Appropriate vocab size (16K BPE)
- Optimized hyperparameters based on successful notebook
- Good training duration (30 epochs)

### ⚠️ V3 Config Issues:

1. **Missing validation files**: References non-existent `val.en.txt`
2. **Model too large**: 1024-dim model with 16 heads is overkill for 133K pairs
3. **Risk of overfitting**: Large model on small dataset
4. **Too few epochs**: Only 5 epochs insufficient for large model
5. **Huge vocab**: 50K BPE tokens excessive for this dataset size

---

## Recommendations

### 🥇 **RECOMMENDED: Use V2 Config (with minor fixes)**

**Why V2 is best:**

- ✅ Correct validation split (tst2012)
- ✅ Optimal vocab size (16K BPE)
- ✅ Modern training techniques (AdamW, cosine schedule, mixed precision)
- ✅ Appropriate model size for dataset
- ✅ Good training duration (30 epochs)
- ✅ Proven hyperparameters (based on BLEU 0.2579 notebook)

**Suggested improvements for V2:**

```yaml
# Keep all V2 settings, just verify these:
data:
  val_src: "data/raw_IWSLT'15_en-vi/tst2012.en.txt" # ✅ Correct
  val_tgt: "data/raw_IWSLT'15_en-vi/tst2012.vi.txt" # ✅ Correct
  test_src: "data/raw_IWSLT'15_en-vi/tst2013.en.txt" # ✅ Correct for final eval

training:
  epochs: 30 # Good for 133K samples
  batch_size: 64
  learning_rate: 0.001
  scheduler: "cosine"
  warmup_steps: 6000
  early_stopping_patience: 12
```

---

### 🥈 **Alternative: Modified V1 Config**

If you want a faster baseline:

**Required fixes:**

```yaml
data:
  # Fix validation split
  val_src: "data/raw_IWSLT'15_en-vi/tst2012.en.txt"
  val_tgt: "data/raw_IWSLT'15_en-vi/tst2012.vi.txt"

vocab:
  src_vocab_size: 16000 # Reduce from 40K
  tgt_vocab_size: 16000
  tokenization: "bpe" # Add BPE

training:
  epochs: 20 # Increase from 5
  batch_size: 64 # Increase from 32
  learning_rate: 0.0005 # Increase slightly
```

---

### ⚠️ **NOT RECOMMENDED: V3 Config**

**Problems:**

- Model too large for dataset size → overfitting risk
- Missing validation files
- Excessive vocab size
- Very small batch size (8) even with accumulation
- Only 5 epochs for large model

**When to use V3:**

- You have much more data (>1M pairs)
- You're doing transfer learning from a pretrained model
- You have significant GPU memory (16GB+ VRAM)

---

## Quick Start Commands

### For V2 (Recommended):

```bash
# 1. Verify config is correct
cat experiments/iwslt_v2_en2vi/config.yaml | grep -A 3 "val_src"

# 2. Install requirements
pip install sacrebleu

# 3. Train
python scripts/train.py --config experiments/iwslt_v2_en2vi/config.yaml

# 4. Expected results
# - Training time: ~8-12 hours on single GPU
# - Expected BLEU on tst2013: 0.25-0.27 (25-27 BLEU points)
# - Best checkpoint: Around epoch 20-25
```

### For Vi→En direction:

```bash
# Use corresponding Vi→En config
python scripts/train.py --config experiments/iwslt_v2_vi2en/config.yaml
```

---

## Training Data Statistics

**IWSLT'15 En-Vi Dataset:**

- Training: 133,318 pairs
- Validation: 1,554 pairs (tst2012)
- Test: 1,269 pairs (tst2013)
- Total: 136,141 sentence pairs

**Estimated training time (V2 config):**

- Batch size: 64
- Epochs: 30
- Steps per epoch: ~2,083 (133,318 / 64)
- Total steps: ~62,490
- Time per step: ~0.5s (on RTX 3090)
- **Total time: ~8-10 hours**

---

## Expected Performance

### V2 Config (Recommended):

- **BLEU on tst2013: 25.0-27.0**
- Training loss: ~2.0-2.5
- Validation loss: ~2.5-3.0
- Convergence: Epoch 20-25

### V1 Config (Baseline):

- **BLEU on tst2013: 18.0-22.0**
- Lower quality due to suboptimal hyperparameters

### V3 Config (Large):

- **BLEU on tst2013: 20.0-24.0** (worse due to overfitting)
- Risk of overfitting after epoch 3-4

---

## Fixes Required

### Priority 1: Fix V1 Config

```yaml
# File: experiments/iwslt_v1_en2vi/config.yaml
data:
  val_src: "data/raw_IWSLT'15_en-vi/tst2012.en.txt" # Change from val.en.txt
  val_tgt: "data/raw_IWSLT'15_en-vi/tst2012.vi.txt" # Change from val.vi.txt
```

### Priority 2: Fix V3 Config

```yaml
# File: experiments/iwslt_v3_en2vi/config.yaml
data:
  val_src: "data/raw_IWSLT'15_en-vi/tst2012.en.txt" # Change from val.en.txt
  val_tgt: "data/raw_IWSLT'15_en-vi/tst2012.vi.txt" # Change from val.vi.txt

# Optional: Reduce model size to match dataset
model:
  d_model: 512 # Reduce from 1024
  n_heads: 8 # Reduce from 16
  d_ff: 2048 # Reduce from 4096

training:
  epochs: 30 # Increase from 5
  batch_size: 32 # Increase from 8
```

---

## Summary

✅ **Use V2 config** - Already optimized and correct
⚠️ **Fix V1 validation paths** - Change `val.*.txt` → `tst2012.*.txt`
⚠️ **Fix V3 validation paths** - Change `val.*.txt` → `tst2012.*.txt`
❌ **Don't use V3 as-is** - Model too large for dataset size

**Best practice:**

1. Start with **V2 config** (no changes needed)
2. Monitor training on WandB
3. Use early stopping to prevent overfitting
4. Evaluate on tst2013 for final score

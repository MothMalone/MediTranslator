"""
Medical Domain Fine-tuning with LoRA
Simple and clean LoRA fine-tuning script for medical translation.

Usage:
    python scripts/finetune_lora.py --config experiments/medical_vocab_expanded_en2vi/config.yaml
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.transformer import Transformer
from src.models.lora import apply_lora_to_model, mark_only_lora_as_trainable, merge_lora_weights
from src.data.sp_vocab import SentencePieceVocab
from src.data.vocabulary import Vocabulary
from src.data.dataset import TranslationDataset, create_dataloader
from src.training.loss import LabelSmoothingLoss
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.helpers import set_seed, get_device
from src.utils.embedding_resize import resize_token_embeddings, print_embedding_info


def load_pretrained_with_lora(config, device, logger):
    """Load pretrained model and apply LoRA."""
    logger.info("=" * 80)
    logger.info("STEP 1: Loading pretrained model")
    logger.info("=" * 80)
    
    # Load pretrained checkpoint for model initialization
    checkpoint_path = config['training']['resume_from']
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Auto-detect if this is a resume (checkpoint has training state)
    has_training_state = all(key in checkpoint for key in ['optimizer_state_dict', 'scheduler_state_dict', 'epoch', 'global_step'])
    
    if has_training_state:
        logger.info("🔄 RESUME MODE: Checkpoint contains training state")
        resume_checkpoint = checkpoint
    else:
        logger.info("🆕 FRESH TRAINING: Starting from pretrained model")
        resume_checkpoint = None
    
    # Load vocabularies
    vocab_config = config.get('vocab', {})
    vocab_expansion_config = config.get('vocab_expansion', {})
    use_bpe = vocab_config.get('tokenization') == 'bpe'
    
    # Use expanded vocab if enabled and available
    if vocab_expansion_config.get('enabled', False) and 'expanded_vocab_dir' in config['paths']:
        vocab_dir = config['paths']['expanded_vocab_dir']
        logger.info(f"Using expanded vocabulary from: {vocab_dir}")
    else:
        vocab_dir = config['paths']['vocab_dir']
        logger.info(f"Using base vocabulary from: {vocab_dir}")
    
    if use_bpe:
        logger.info(f"Loading SentencePiece vocabularies from: {vocab_dir}")
        
        # Load SentencePiece models (for tokenization)
        src_vocab = SentencePieceVocab(os.path.join(vocab_dir, 'src.model'))
        tgt_vocab = SentencePieceVocab(os.path.join(vocab_dir, 'tgt.model'))
        
        # Read vocab size directly from .vocab files (overrides .model size if expanded)
        src_vocab_file = os.path.join(vocab_dir, 'src.vocab')
        tgt_vocab_file = os.path.join(vocab_dir, 'tgt.vocab')
        
        actual_src_vocab_size = len(src_vocab)
        actual_tgt_vocab_size = len(tgt_vocab)
        
        # If .vocab files exist, use their size (for expanded vocabs)
        if os.path.exists(src_vocab_file):
            with open(src_vocab_file, 'r', encoding='utf-8') as f:
                actual_src_vocab_size = sum(1 for _ in f)
            logger.info(f"  ✓ Source vocab from .vocab file: {actual_src_vocab_size} tokens")
        
        if os.path.exists(tgt_vocab_file):
            with open(tgt_vocab_file, 'r', encoding='utf-8') as f:
                actual_tgt_vocab_size = sum(1 for _ in f)
            logger.info(f"  ✓ Target vocab from .vocab file: {actual_tgt_vocab_size} tokens")
        
        logger.info(f"✓ Loaded BPE vocabularies: src={actual_src_vocab_size}, tgt={actual_tgt_vocab_size}")
    else:
        src_vocab = Vocabulary.load(os.path.join(vocab_dir, 'src_vocab.json'))
        tgt_vocab = Vocabulary.load(os.path.join(vocab_dir, 'tgt_vocab.json'))
        actual_src_vocab_size = len(src_vocab)
        actual_tgt_vocab_size = len(tgt_vocab)
        logger.info(f"✓ Loaded vocabularies: src={actual_src_vocab_size}, tgt={actual_tgt_vocab_size}")
    
    # Create model with ACTUAL vocab size (from .vocab files if expanded)
    model_config = config['model']
    model = Transformer(
        src_vocab_size=actual_src_vocab_size,
        tgt_vocab_size=actual_tgt_vocab_size,
        d_model=model_config['d_model'],
        n_heads=model_config['n_heads'],
        n_encoder_layers=model_config['n_encoder_layers'],
        n_decoder_layers=model_config['n_decoder_layers'],
        d_ff=model_config['d_ff'],
        dropout=model_config['dropout'],
        max_seq_length=model_config['max_seq_length'],
        pad_idx=src_vocab.pad_idx,
        use_xavier_init=model_config.get('use_xavier_init', True)
    )
    
    # Load pretrained weights
    checkpoint_state = checkpoint['model_state_dict']
    pretrained_src_vocab_size = None
    pretrained_tgt_vocab_size = None
    
    # Detect pretrained vocab sizes from checkpoint
    # Model structure: src_embedding.weight and tgt_embedding.weight (not .embedding.weight)
    for key in checkpoint_state.keys():
        if key == 'src_embedding.weight':
            pretrained_src_vocab_size = checkpoint_state[key].shape[0]
            logger.info(f"Found src embedding: {key}, size: {pretrained_src_vocab_size}")
        if key == 'tgt_embedding.weight':
            pretrained_tgt_vocab_size = checkpoint_state[key].shape[0]
            logger.info(f"Found tgt embedding: {key}, size: {pretrained_tgt_vocab_size}")
    
    # If not found, assume current vocab size (no expansion)
    if pretrained_src_vocab_size is None:
        logger.warning("Could not detect pretrained src vocab size from checkpoint, assuming current size")
        pretrained_src_vocab_size = actual_src_vocab_size
    if pretrained_tgt_vocab_size is None:
        logger.warning("Could not detect pretrained tgt vocab size from checkpoint, assuming current size")
        pretrained_tgt_vocab_size = actual_tgt_vocab_size
    
    # Check if vocabulary has been expanded (use ACTUAL vocab size from files)
    vocab_expanded = (actual_src_vocab_size > pretrained_src_vocab_size or 
                      actual_tgt_vocab_size > pretrained_tgt_vocab_size)
    
    # Check if checkpoint has LoRA weights (indicates resume training)
    has_lora_weights = any('lora' in key.lower() for key in checkpoint_state.keys())
    
    if has_lora_weights:
        logger.info("=" * 80)
        logger.info("🔄 RESUME MODE: Checkpoint contains LoRA weights")
        logger.info("=" * 80)
        # When resuming, vocab should already be expanded in checkpoint
        # Don't re-expand or resize embeddings
        if vocab_expanded:
            logger.warning("⚠️  Vocab size mismatch detected in resume mode!")
            logger.warning(f"Current: src={actual_src_vocab_size}, tgt={actual_tgt_vocab_size}")
            logger.warning(f"Checkpoint: src={pretrained_src_vocab_size}, tgt={pretrained_tgt_vocab_size}")
            logger.warning("This might indicate wrong vocab files. Continuing with checkpoint vocab...")
            vocab_expanded = False  # Don't resize when resuming
    else:
        logger.info("=" * 80)
        logger.info("🆕 FRESH TRAINING: Loading pretrained model")
        if vocab_expanded:
            src_expansion = actual_src_vocab_size - pretrained_src_vocab_size
            tgt_expansion = actual_tgt_vocab_size - pretrained_tgt_vocab_size
            logger.info(f"📈 Vocab expansion detected:")
            if src_expansion > 0:
                logger.info(f"   Source: {pretrained_src_vocab_size} → {actual_src_vocab_size} (+{src_expansion})")
            if tgt_expansion > 0:
                logger.info(f"   Target: {pretrained_tgt_vocab_size} → {actual_tgt_vocab_size} (+{tgt_expansion})")
        logger.info("=" * 80)
    
    # Step 1: Apply LoRA architecture FIRST (if enabled)
    lora_config = config.get('lora', {})
    should_apply_lora = lora_config.get('enabled', True)
    
    if should_apply_lora:
        logger.info("\n📐 Applying LoRA architecture...")
        model, lora_count = apply_lora_to_model(
            model,
            target_modules=lora_config.get('target_modules', ['W_Q', 'W_V']),
            rank=lora_config.get('rank', 8),
            alpha=lora_config.get('alpha', 16),
            dropout=lora_config.get('dropout', 0.0)
        )
        logger.info(f"✓ Applied LoRA to {lora_count} layers")
        
        if lora_count == 0:
            logger.warning("WARNING: No LoRA layers applied! Trying fallback patterns...")
            model, lora_count = apply_lora_to_model(
                model,
                target_modules=['q_proj', 'v_proj', 'query', 'value'],
                rank=lora_config.get('rank', 8),
                alpha=lora_config.get('alpha', 16),
                dropout=lora_config.get('dropout', 0.0)
            )
            logger.info(f"✓ Applied LoRA to {lora_count} layers (fallback)")
    
    # Step 2: Load weights based on scenario
    if vocab_expanded and not has_lora_weights:
        # Fresh training with vocab expansion
        logger.info("\n📦 Loading pretrained weights + expanding vocabulary...")
        
        # Load weights manually - skip embeddings that have size mismatch
        model_state = model.state_dict()
        embedding_keys = ['src_embedding.weight', 'tgt_embedding.weight', 
                         'output_projection.weight', 'output_projection.bias']
        
        # Load non-embedding weights first
        filtered_checkpoint = {k: v for k, v in checkpoint_state.items() 
                              if k not in embedding_keys}
        missing_keys, unexpected_keys = model.load_state_dict(filtered_checkpoint, strict=False)
        logger.info("✓ Loaded pretrained weights (non-embedding layers)")
        
        # Manually copy embedding weights (old vocab size only)
        logger.info("Copying embedding weights for original vocabulary...")
        with torch.no_grad():
            if 'src_embedding.weight' in checkpoint_state:
                old_src_emb = checkpoint_state['src_embedding.weight']
                model.src_embedding.weight[:pretrained_src_vocab_size] = old_src_emb
                logger.info(f"  ✓ Copied src_embedding: {pretrained_src_vocab_size}/{actual_src_vocab_size} tokens")
            
            if 'tgt_embedding.weight' in checkpoint_state:
                old_tgt_emb = checkpoint_state['tgt_embedding.weight']
                model.tgt_embedding.weight[:pretrained_tgt_vocab_size] = old_tgt_emb
                logger.info(f"  ✓ Copied tgt_embedding: {pretrained_tgt_vocab_size}/{actual_tgt_vocab_size} tokens")
            
            if 'output_projection.weight' in checkpoint_state:
                old_out_weight = checkpoint_state['output_projection.weight']
                model.output_projection.weight[:pretrained_tgt_vocab_size] = old_out_weight
                logger.info(f"  ✓ Copied output_projection.weight: {pretrained_tgt_vocab_size}/{actual_tgt_vocab_size} tokens")
            
            if 'output_projection.bias' in checkpoint_state:
                old_out_bias = checkpoint_state['output_projection.bias']
                model.output_projection.bias[:pretrained_tgt_vocab_size] = old_out_bias
                logger.info(f"  ✓ Copied output_projection.bias: {pretrained_tgt_vocab_size}/{actual_tgt_vocab_size} tokens")
        
        logger.info("✓ Embeddings expanded successfully (new tokens initialized randomly)")
        
        # Print embedding info
        print_embedding_info(model)
        print_embedding_info(model)
    else:
        # Normal loading (vocab sizes match)
        logger.info("\n📦 Loading weights...")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint_state, strict=False)
        
        if has_lora_weights:
            logger.info("✓ Loaded model + LoRA weights (resume training)")
        else:
            logger.info("✓ Loaded pretrained weights (fresh training)")
            
        # Explain missing/unexpected keys
        if missing_keys:
            # Count LoRA-related missing keys
            lora_missing = [k for k in missing_keys if 'lora' in k.lower()]
            if lora_missing:
                logger.info(f"  → Missing {len(lora_missing)} LoRA weights (will be initialized randomly)")
            other_missing = len(missing_keys) - len(lora_missing)
            if other_missing > 0:
                logger.warning(f"  ⚠️  Missing {other_missing} non-LoRA keys")
        
        if unexpected_keys:
            logger.info(f"  → Unexpected keys: {len(unexpected_keys)} (ignored)")
    
    # Step 3: Freeze base model parameters
    if should_apply_lora:
        mark_only_lora_as_trainable(model)
        logger.info("✓ Froze base model parameters (only LoRA trainable)")
        
        # Unfreeze new embeddings if vocab was expanded
        if vocab_expanded:
            logger.info("\n📌 Unfreezing new vocabulary embeddings...")
            
            # Unfreeze entire embedding layers (new tokens need gradients)
            # PyTorch doesn't support partial parameter freezing, so we unfreeze the whole layer
            # Old embeddings won't update much due to low learning rate for embeddings
            unfrozen_count = 0
            
            if actual_src_vocab_size > pretrained_src_vocab_size:
                model.src_embedding.weight.requires_grad = True
                new_tokens = actual_src_vocab_size - pretrained_src_vocab_size
                logger.info(f"  ✓ Unfroze src_embedding (includes {new_tokens} new tokens)")
                unfrozen_count += model.src_embedding.weight.numel()
            
            if actual_tgt_vocab_size > pretrained_tgt_vocab_size:
                model.tgt_embedding.weight.requires_grad = True
                new_tokens = actual_tgt_vocab_size - pretrained_tgt_vocab_size
                logger.info(f"  ✓ Unfroze tgt_embedding (includes {new_tokens} new tokens)")
                unfrozen_count += model.tgt_embedding.weight.numel()
                
                # Also unfreeze output projection (shares vocab with target)
                model.output_projection.weight.requires_grad = True
                model.output_projection.bias.requires_grad = True
                logger.info(f"  ✓ Unfroze output_projection (includes {new_tokens} new tokens)")
                unfrozen_count += model.output_projection.weight.numel() + model.output_projection.bias.numel()
            
            logger.info(f"  → Total unfrozen embedding params: {unfrozen_count:,}")
            logger.info("  ⚠️  Note: Use low LR for embeddings (e.g., 0.5x base LR) to avoid catastrophic forgetting")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    model.to(device)
    return model, src_vocab, tgt_vocab, resume_checkpoint


def create_dataloaders(config, src_vocab, tgt_vocab, logger):
    """Create train and validation dataloaders."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Preparing medical dataset")
    logger.info("=" * 80)
    
    data_config = config['data']
    train_config = config['training']
    vocab_config = config.get('vocab', {})
    use_bpe = vocab_config.get('tokenization') == 'bpe'
    
    # Read training data
    with open(data_config['train_src'], 'r', encoding='utf-8') as f:
        train_src_lines = f.readlines()
    with open(data_config['train_tgt'], 'r', encoding='utf-8') as f:
        train_tgt_lines = f.readlines()
    
    logger.info(f"Loaded {len(train_src_lines):,} training samples")
    
    # Split train/val
    val_split = data_config.get('val_split', 0.01)
    split_idx = int(len(train_src_lines) * (1 - val_split))
    
    # Create temp files
    os.makedirs('data/temp', exist_ok=True)
    
    with open('data/temp/train_src.txt', 'w', encoding='utf-8') as f:
        f.writelines(train_src_lines[:split_idx])
    with open('data/temp/train_tgt.txt', 'w', encoding='utf-8') as f:
        f.writelines(train_tgt_lines[:split_idx])
    with open('data/temp/val_src.txt', 'w', encoding='utf-8') as f:
        f.writelines(train_src_lines[split_idx:])
    with open('data/temp/val_tgt.txt', 'w', encoding='utf-8') as f:
        f.writelines(train_tgt_lines[split_idx:])
    
    logger.info(f"✓ Data split: {split_idx:,} train, {len(train_src_lines)-split_idx:,} val")
    
    # Create datasets
    train_dataset = TranslationDataset(
        src_file='data/temp/train_src.txt',
        tgt_file='data/temp/train_tgt.txt',
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        src_tokenizer=src_vocab if use_bpe else None,
        tgt_tokenizer=tgt_vocab if use_bpe else None,
        max_length=data_config.get('max_seq_length', 256)
    )
    
    val_dataset = TranslationDataset(
        src_file='data/temp/val_src.txt',
        tgt_file='data/temp/val_tgt.txt',
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        src_tokenizer=src_vocab if use_bpe else None,
        tgt_tokenizer=tgt_vocab if use_bpe else None,
        max_length=data_config.get('max_seq_length', 256)
    )
    
    logger.info(f"✓ Train dataset: {len(train_dataset):,} samples")
    logger.info(f"✓ Val dataset: {len(val_dataset):,} samples")
    
    # Create dataloaders
    batch_size = train_config.get('batch_size', 32)
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pad_idx=src_vocab.pad_idx
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pad_idx=src_vocab.pad_idx
    )
    
    logger.info(f"✓ Train batches: {len(train_loader):,}")
    logger.info(f"✓ Val batches: {len(val_loader):,}")
    
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, config, device, tgt_vocab, logger, resume_checkpoint=None):
    """Training loop with validation and SWA."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Training with LoRA + SWA")
    logger.info("=" * 80)
    
    train_config = config['training']
    
    # Check if resuming training
    is_resuming = resume_checkpoint is not None
    if is_resuming:
        logger.info("📂 RESUMING TRAINING from checkpoint")
        start_epoch = resume_checkpoint.get('epoch', 0)
        start_step = resume_checkpoint.get('global_step', 0)
        best_val_loss_resumed = resume_checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"  Starting from epoch {start_epoch + 1}, step {start_step}")
        logger.info(f"  Best val loss so far: {best_val_loss_resumed:.4f}")
    else:
        logger.info("🆕 STARTING FRESH TRAINING")
        start_epoch = 0
        start_step = 0
        best_val_loss_resumed = float('inf')
    
    # Initialize wandb
    logger.info("\n" + "=" * 80)
    logger.info("Initializing Weights & Biases...")
    logger.info("=" * 80)
    
    # Check both training.use_wandb and wandb.enabled
    use_wandb = train_config.get('use_wandb', False) or config.get('wandb', {}).get('enabled', False)
    wandb_id = config.get('wandb', {}).get('id', None)
    
    if use_wandb:
        try:
            import wandb
            wandb_kwargs = dict(
                project=config.get('wandb', {}).get('project', 'nlp-medical-mt'),
                name=config.get('wandb', {}).get('name', 'lora_finetune'),
                config=config,
                tags=config.get('wandb', {}).get('tags', [])
            )
            if wandb_id is not None:
                wandb_kwargs['id'] = wandb_id
                wandb_kwargs['resume'] = 'must'
            wandb.init(**wandb_kwargs)
            logger.info(f"✓ Wandb initialized (id={wandb_id})")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
    else:
        logger.info("✗ Wandb disabled")
    
    # Optimizer with discriminative learning rates
    learning_rate = train_config.get('learning_rate', 1e-4)
    
    # Check if discriminative LR is enabled
    use_discriminative_lr = train_config.get('use_discriminative_lr', False)
    
    if use_discriminative_lr:
        logger.info("Creating optimizer with discriminative learning rates...")
        
        # Get LR multipliers for different layer groups
        lr_groups = train_config.get('discriminative_lr_groups', {
            'embeddings': 0.5,
            'encoder': 0.7,
            'decoder': 1.0
        })
        
        # Separate parameters by layer group
        embedding_params = []
        encoder_params = []
        decoder_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            if 'embedding' in name.lower():
                embedding_params.append(param)
            elif 'encoder' in name.lower():
                encoder_params.append(param)
            elif 'decoder' in name.lower():
                decoder_params.append(param)
            else:
                other_params.append(param)
        
        # Create parameter groups with different learning rates
        param_groups = []
        
        if embedding_params:
            emb_lr = learning_rate * lr_groups.get('embeddings', 0.5)
            param_groups.append({
                'params': embedding_params,
                'lr': emb_lr,
                'name': 'embeddings'
            })
            logger.info(f"  - Embeddings: {len(embedding_params)} params, lr={emb_lr:.2e}")
        
        if encoder_params:
            enc_lr = learning_rate * lr_groups.get('encoder', 0.7)
            param_groups.append({
                'params': encoder_params,
                'lr': enc_lr,
                'name': 'encoder'
            })
            logger.info(f"  - Encoder: {len(encoder_params)} params, lr={enc_lr:.2e}")
        
        if decoder_params:
            dec_lr = learning_rate * lr_groups.get('decoder', 1.0)
            param_groups.append({
                'params': decoder_params,
                'lr': dec_lr,
                'name': 'decoder'
            })
            logger.info(f"  - Decoder: {len(decoder_params)} params, lr={dec_lr:.2e}")
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': learning_rate,
                'name': 'other'
            })
            logger.info(f"  - Other: {len(other_params)} params, lr={learning_rate:.2e}")
        
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=train_config.get('weight_decay', 0.01),
            betas=train_config.get('betas', [0.9, 0.999])
        )
        logger.info(f"✓ Optimizer: AdamW with discriminative LR (base_lr={learning_rate:.2e})")
    else:
        # Standard optimizer with single learning rate
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=train_config.get('weight_decay', 0.01),
            betas=train_config.get('betas', [0.9, 0.999])
        )
        logger.info(f"✓ Optimizer: AdamW (lr={learning_rate})")
    
    # Load optimizer state if resuming
    if is_resuming and 'optimizer_state_dict' in resume_checkpoint:
        try:
            # Check if parameter groups match
            saved_groups = len(resume_checkpoint['optimizer_state_dict']['param_groups'])
            current_groups = len(optimizer.param_groups)
            
            if saved_groups == current_groups:
                optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
                logger.info("✓ Loaded optimizer state from checkpoint")
            else:
                logger.warning(f"⚠️  Optimizer structure mismatch!")
                logger.warning(f"   Checkpoint has {saved_groups} param groups, current has {current_groups}")
                logger.warning(f"   Skipping optimizer state - will use fresh optimizer")
                logger.warning(f"   This happens when discriminative LR config changed")
        except Exception as e:
            logger.warning(f"⚠️  Failed to load optimizer state: {e}")
            logger.warning(f"   Continuing with fresh optimizer")
    
    # SWA setup
    # Support both flat config (use_swa: true) and nested config (swa.enabled: true)
    swa_enabled = train_config.get('use_swa', False)
    if not swa_enabled:
        swa_config = train_config.get('swa', {})
        swa_enabled = swa_config.get('enabled', False)
    
    if swa_enabled:
        swa_model = AveragedModel(model)
        # Try flat config first, then nested
        swa_start_epoch = train_config.get('swa_start_epoch')
        swa_lr = train_config.get('swa_lr')
        
        if swa_start_epoch is None or swa_lr is None:
            swa_config = train_config.get('swa', {})
            swa_start_epoch = swa_config.get('start_epoch', 2)
            swa_lr = swa_config.get('lr', learning_rate * 0.1)
        
        logger.info(f"✓ SWA enabled: start_epoch={swa_start_epoch}, swa_lr={swa_lr}")
    else:
        swa_model = None
        logger.info("✗ SWA disabled")
    
    # Scheduler
    total_steps = len(train_loader) * train_config.get('epochs', 5)
    warmup_steps = train_config.get('warmup_steps', 500)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress))))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    logger.info(f"✓ Scheduler: Cosine with warmup (warmup_steps={warmup_steps})")
    
    # Load scheduler state if resuming (and optimizer was loaded successfully)
    if is_resuming and 'scheduler_state_dict' in resume_checkpoint:
        try:
            scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
            logger.info("✓ Loaded scheduler state from checkpoint")
        except Exception as e:
            logger.warning(f"⚠️  Failed to load scheduler state: {e}")
            logger.warning(f"   Continuing with fresh scheduler")
    
    # SWA scheduler (will be used after SWA starts)
    if swa_enabled:
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
    else:
        swa_scheduler = None
    
    # Loss
    criterion = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        padding_idx=tgt_vocab.pad_idx,
        smoothing=train_config.get('label_smoothing', 0.1)
    )
    
    # Training settings
    epochs = train_config.get('epochs', 5)
    grad_accum_steps = train_config.get('gradient_accumulation_steps', 1)
    max_grad_norm = train_config.get('max_grad_norm', 1.0)
    eval_every = train_config.get('eval_every', 500)
    patience = train_config.get('early_stopping_patience', 5)
    
    logger.info(f"✓ Epochs: {epochs}")
    logger.info(f"✓ Gradient accumulation: {grad_accum_steps}")
    logger.info(f"✓ Eval every: {eval_every} steps")
    logger.info(f"✓ Early stopping patience: {patience}")
    
    # Training state
    best_val_loss = best_val_loss_resumed if is_resuming else float('inf')
    patience_counter = 0
    global_step = start_step
    checkpoint_dir = config['paths']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    swa_active = False
    
    logger.info("\n" + "=" * 80)
    if is_resuming:
        logger.info(f"Resuming training from epoch {start_epoch + 1}...")
    else:
        logger.info("Starting training...")
    logger.info("=" * 80 + "\n")
    
    for epoch in range(start_epoch, epochs):
        # Activate SWA if configured
        if swa_enabled and not swa_active and epoch >= swa_start_epoch:
            swa_active = True
            logger.info(f"\n{'='*80}")
            logger.info(f"🔄 SWA ACTIVATED at epoch {epoch+1}")
            logger.info(f"{'='*80}\n")
        
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}{'[SWA]' if swa_active else ''}")
        
        for batch_idx, (src, tgt, src_lengths, tgt_lengths) in enumerate(progress_bar):
            src = src.to(device)
            tgt = tgt.to(device)
            
            # Teacher forcing
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Forward pass
            output = model(src, tgt_input)
            loss = criterion(
                output.contiguous().view(-1, output.size(-1)),
                tgt_output.contiguous().view(-1)
            )
            loss = loss / grad_accum_steps
            
            # Backward pass
            loss.backward()
            
            # Optimizer step with gradient accumulation
            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                
                # Update SWA model if active
                if swa_active and swa_model is not None:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()
                else:
                    scheduler.step()
                
                optimizer.zero_grad()
                global_step += 1
            
            epoch_loss += loss.item() * grad_accum_steps
            
            # Update progress bar
            if swa_active and swa_scheduler is not None:
                current_lr = swa_scheduler.get_last_lr()[0] if hasattr(swa_scheduler, 'get_last_lr') else swa_lr
            else:
                current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f"{loss.item() * grad_accum_steps:.4f}",
                'lr': f"{current_lr:.2e}",
                'swa': '✓' if swa_active else '✗'
            })
            
            # Log to wandb
            if global_step % train_config.get('log_every', 50) == 0:
                use_wandb_logging = train_config.get('use_wandb', False) or config.get('wandb', {}).get('enabled', False)
                if use_wandb_logging:
                    try:
                        import wandb
                        wandb.log({
                            'train/loss': loss.item() * grad_accum_steps,
                            'train/lr': current_lr,
                            'train/step': global_step,
                            'train/swa_active': swa_active
                        })
                    except:
                        pass
            
            # Validation
            if global_step > 0 and global_step % eval_every == 0:
                val_loss = validate(model, val_loader, criterion, device)
                logger.info(f"Step {global_step}: Train Loss = {loss.item() * grad_accum_steps:.4f}, Val Loss = {val_loss:.4f}, LR = {current_lr:.2e}")
                
                # Log validation to wandb
                use_wandb_logging = train_config.get('use_wandb', False) or config.get('wandb', {}).get('enabled', False)
                if use_wandb_logging:
                    try:
                        import wandb
                        wandb.log({
                            'val/loss': val_loss,
                            'val/step': global_step
                        })
                    except:
                        pass
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch,
                        'global_step': global_step,
                        'best_val_loss': best_val_loss,
                        'config': config
                    }, os.path.join(checkpoint_dir, 'best_model.pt'))
                    logger.info(f"✓ Saved best model (val_loss={val_loss:.4f})")
                else:
                    patience_counter += 1
                    logger.info(f"No improvement ({patience_counter}/{patience})")
                
                model.train()
                
                # Early stopping
                if patience_counter >= patience:
                    logger.info(f"\nEarly stopping triggered at step {global_step}")
                    break
        
        # End of epoch
        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"\nEpoch {epoch+1}/{epochs} completed - Avg Loss: {avg_loss:.4f}\n")
        
        if patience_counter >= patience:
            break
    
    # Finalize SWA
    if swa_active and swa_model is not None:
        logger.info("\n" + "=" * 80)
        logger.info("Finalizing SWA...")
        logger.info("=" * 80)
        
        # Update batch normalization if present
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        logger.info("✓ Updated batch normalization statistics")
        
        # Evaluate SWA model
        swa_val_loss = validate(swa_model, val_loader, criterion, device)
        logger.info(f"✓ SWA model validation loss: {swa_val_loss:.4f}")
        
        # Use SWA model if it's better
        if swa_val_loss < best_val_loss:
            logger.info(f"✓ SWA model is better! ({swa_val_loss:.4f} < {best_val_loss:.4f})")
            model.load_state_dict(swa_model.module.state_dict())
            best_val_loss = swa_val_loss
            
            # Save SWA model as best
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config,
                'swa_applied': True
            }, os.path.join(checkpoint_dir, 'best_model.pt'))
            logger.info("✓ Saved SWA model as best model")
        else:
            logger.info(f"✗ Regular model is better ({best_val_loss:.4f} < {swa_val_loss:.4f})")
    
    logger.info("\n" + "=" * 80)
    logger.info("Training completed!")
    logger.info("=" * 80)
    
    return best_val_loss


def validate(model, val_loader, criterion, device):
    """Validation loop."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt, src_lengths, tgt_lengths in val_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            output = model(src, tgt_input)
            loss = criterion(
                output.contiguous().view(-1, output.size(-1)),
                tgt_output.contiguous().view(-1)
            )
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def save_final_model(model, config, logger):
    """Merge LoRA weights and save final model."""
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Saving final model")
    logger.info("=" * 80)
    
    checkpoint_dir = config['paths']['checkpoint_dir']
    
    # Merge LoRA weights (this replaces LinearWithLoRA with nn.Linear)
    model = merge_lora_weights(model)
    logger.info("✓ Merged LoRA weights into base model")
    
    # Save merged model
    final_path = os.path.join(checkpoint_dir, 'final_model_merged.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, final_path)
    logger.info(f"✓ Saved merged model: {final_path}")
    
    return final_path


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for medical translation")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup directories
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(config['paths']['log_dir'], 'finetune_lora.log')
    logger = setup_logger("lora_finetune", log_file=log_file)
    
    logger.info("=" * 80)
    logger.info("MEDICAL LORA FINE-TUNING")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Get device
    device = get_device(config.get('device', 'cuda'))
    logger.info(f"Device: {device}\n")
    
    # Load model with LoRA
    model, src_vocab, tgt_vocab, resume_checkpoint = load_pretrained_with_lora(config, device, logger)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config, src_vocab, tgt_vocab, logger)
    
    # Train
    best_val_loss = train_model(model, train_loader, val_loader, config, device, tgt_vocab, logger, resume_checkpoint)
    
    # Save final model
    final_path = save_final_model(model, config, logger)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ FINE-TUNING COMPLETED!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Final model saved: {final_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

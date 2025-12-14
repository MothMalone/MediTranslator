"""
Medical Domain Fine-tuning Script with Advanced Techniques
Implements:
- Tier 1: LoRA, Lower LR, Cosine Annealing, Longer Training
- Tier 2: Discriminative LR, Stochastic Weight Averaging (SWA)

Usage:
    python scripts/finetune_medical.py --config experiments/medical_vi2en/config.yaml
    python scripts/finetune_medical.py --config experiments/medical_en2vi/config.yaml
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR
import logging
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.transformer import Transformer
from src.models.lora import apply_lora_to_model, get_lora_parameters, merge_lora_weights
from src.data.vocabulary import Vocabulary
from src.data.dataset import TranslationDataset, create_dataloader
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.helpers import set_seed, get_device


class DiscriminativeLROptimizer:
    """Optimizer with discriminative learning rates for different layer groups."""
    
    def __init__(self, model, config, base_lr):
        self.model = model
        self.config = config
        self.base_lr = base_lr
        
        # Group parameters
        param_groups = self._create_param_groups()
        
        # Create optimizer
        opt_config = config['training']
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=base_lr,
            weight_decay=opt_config.get('weight_decay', 0.01),
            betas=opt_config.get('betas', [0.9, 0.999]),
            eps=opt_config.get('eps', 1e-8)
        )
    
    def _create_param_groups(self):
        """Create parameter groups with different learning rates."""
        train_config = self.config['training']
        
        if not train_config.get('use_discriminative_lr', False):
            # No discriminative LR, return all LoRA params with base LR
            return [{'params': get_lora_parameters(self.model), 'lr': self.base_lr}]
        
        lr_groups = train_config.get('discriminative_lr_groups', {
            'embeddings': 0.5,
            'encoder': 0.7,
            'decoder': 1.0
        })
        
        # Collect parameters by group
        embedding_params = []
        encoder_params = []
        decoder_params = []
        
        for name, module in self.model.named_modules():
            if 'lora' in name.lower():
                # This is a LoRA parameter
                if 'embedding' in name.lower():
                    embedding_params.extend(module.parameters())
                elif 'encoder' in name.lower():
                    encoder_params.extend(module.parameters())
                elif 'decoder' in name.lower():
                    decoder_params.extend(module.parameters())
        
        param_groups = []
        
        if embedding_params:
            param_groups.append({
                'params': embedding_params,
                'lr': self.base_lr * lr_groups.get('embeddings', 0.5),
                'name': 'embeddings'
            })
            logging.info(f"Embedding LR: {self.base_lr * lr_groups.get('embeddings', 0.5):.2e}")
        
        if encoder_params:
            param_groups.append({
                'params': encoder_params,
                'lr': self.base_lr * lr_groups.get('encoder', 0.7),
                'name': 'encoder'
            })
            logging.info(f"Encoder LR: {self.base_lr * lr_groups.get('encoder', 0.7):.2e}")
        
        if decoder_params:
            param_groups.append({
                'params': decoder_params,
                'lr': self.base_lr * lr_groups.get('decoder', 1.0),
                'name': 'decoder'
            })
            logging.info(f"Decoder LR: {self.base_lr * lr_groups.get('decoder', 1.0):.2e}")
        
        # If no params were collected, fall back to all LoRA params
        if not param_groups:
            param_groups = [{'params': get_lora_parameters(self.model), 'lr': self.base_lr}]
        
        return param_groups
    
    def get_optimizer(self):
        return self.optimizer


class CosineWarmupScheduler:
    """Cosine annealing scheduler with linear warmup."""
    
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = 0
        
        # Store initial LRs for each param group
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self):
        """Update learning rate."""
        self.current_step += 1
        
        for i, (base_lr, param_group) in enumerate(zip(self.base_lrs, self.optimizer.param_groups)):
            if self.current_step < self.warmup_steps:
                # Linear warmup
                lr = base_lr * self.current_step / self.warmup_steps
            else:
                # Cosine annealing
                progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                lr = self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
                lr = lr.item() if torch.is_tensor(lr) else lr
            
            param_group['lr'] = lr
    
    def get_last_lr(self):
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]


def load_pretrained_model_with_lora(config, device):
    """Load pretrained model and apply LoRA."""
    logger = logging.getLogger(__name__)
    
    # Load checkpoint
    checkpoint_path = config['training']['resume_from']
    logger.info(f"Loading pretrained checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load vocabularies
    vocab_dir = config['paths']['vocab_dir']
    src_vocab = Vocabulary.load(os.path.join(vocab_dir, 'src_vocab.json'))
    tgt_vocab = Vocabulary.load(os.path.join(vocab_dir, 'tgt_vocab.json'))
    
    logger.info(f"Loaded vocabularies: src={len(src_vocab)}, tgt={len(tgt_vocab)}")
    
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
    
    # Load pretrained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("✓ Loaded pretrained weights")
    
    # Apply LoRA
    lora_config = config.get('lora', {})
    if lora_config.get('enabled', False):
        logger.info("\nApplying LoRA...")
        model, lora_count = apply_lora_to_model(
            model,
            target_modules=lora_config.get('target_modules', ['query', 'value']),
            rank=lora_config.get('rank', 8),
            alpha=lora_config.get('alpha', 16),
            dropout=lora_config.get('dropout', 0.0)
        )
        logger.info(f"✓ Applied LoRA to {lora_count} layers")
        
        # Count trainable vs total parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    model.to(device)
    return model, src_vocab, tgt_vocab


def create_data_loaders(config, src_vocab, tgt_vocab):
    """Create training and validation data loaders."""
    logger = logging.getLogger(__name__)
    data_config = config['data']
    train_config = config['training']
    
    # Read training data
    with open(data_config['train_src'], 'r', encoding='utf-8') as f:
        train_src_lines = f.readlines()
    with open(data_config['train_tgt'], 'r', encoding='utf-8') as f:
        train_tgt_lines = f.readlines()
    
    # Create train/val split
    val_split = data_config.get('val_split', 0.05)
    split_idx = int(len(train_src_lines) * (1 - val_split))
    
    # Save splits to temp files
    os.makedirs('data/temp', exist_ok=True)
    
    with open('data/temp/train_src.txt', 'w', encoding='utf-8') as f:
        f.writelines(train_src_lines[:split_idx])
    with open('data/temp/train_tgt.txt', 'w', encoding='utf-8') as f:
        f.writelines(train_tgt_lines[:split_idx])
    with open('data/temp/val_src.txt', 'w', encoding='utf-8') as f:
        f.writelines(train_src_lines[split_idx:])
    with open('data/temp/val_tgt.txt', 'w', encoding='utf-8') as f:
        f.writelines(train_tgt_lines[split_idx:])
    
    logger.info(f"Data split: {split_idx:,} train, {len(train_src_lines)-split_idx:,} val")
    
    # Create datasets
    train_dataset = TranslationDataset(
        src_file='data/temp/train_src.txt',
        tgt_file='data/temp/train_tgt.txt',
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_length=data_config.get('max_seq_length', 256)
    )
    
    val_dataset = TranslationDataset(
        src_file='data/temp/val_src.txt',
        tgt_file='data/temp/val_tgt.txt',
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_length=data_config.get('max_seq_length', 256)
    )
    
    # Create test dataset if available
    test_dataset = None
    if 'test_src' in data_config and os.path.exists(data_config['test_src']):
        test_dataset = TranslationDataset(
            src_file=data_config['test_src'],
            tgt_file=data_config['test_tgt'],
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            max_length=data_config.get('max_seq_length', 256)
        )
        logger.info(f"Test set: {len(test_dataset):,} samples")
    
    # Create data loaders
    batch_size = train_config.get('batch_size', 16)
    
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
    
    test_loader = None
    if test_dataset:
        test_loader = create_dataloader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pad_idx=src_vocab.pad_idx
        )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="Medical domain fine-tuning with advanced techniques")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup directories
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(config['paths']['log_dir'], 'finetune_medical.log')
    logger = setup_logger("medical_finetune", log_file=log_file)
    
    logger.info("=" * 80)
    logger.info("MEDICAL DOMAIN FINE-TUNING")
    logger.info("Tier 1 + Tier 2 Optimizations")
    logger.info("=" * 80)
    
    # Set seed
    set_seed(config.get('seed', 42))
    device = get_device(config.get('device', 'cuda'))
    logger.info(f"Device: {device}")
    
    # Load model with LoRA
    logger.info("\n" + "=" * 80)
    logger.info("Loading pretrained model and applying LoRA...")
    logger.info("=" * 80)
    model, src_vocab, tgt_vocab = load_pretrained_model_with_lora(config, device)
    
    # Create data loaders
    logger.info("\n" + "=" * 80)
    logger.info("Preparing medical dataset...")
    logger.info("=" * 80)
    train_loader, val_loader, test_loader = create_data_loaders(config, src_vocab, tgt_vocab)
    
    # Create optimizer with discriminative learning rates
    logger.info("\n" + "=" * 80)
    logger.info("Setting up optimizer with discriminative learning rates...")
    logger.info("=" * 80)
    base_lr = config['training'].get('learning_rate', 5e-5)
    discriminative_opt = DiscriminativeLROptimizer(model, config, base_lr)
    optimizer = discriminative_opt.get_optimizer()
    
    # Create scheduler
    train_config = config['training']
    total_steps = len(train_loader) * train_config.get('epochs', 25)
    warmup_steps = train_config.get('warmup_steps', 1000)
    
    if train_config.get('scheduler', 'cosine_warmup') == 'cosine_warmup':
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=train_config.get('min_lr', 1e-6)
        )
        logger.info(f"✓ Cosine warmup scheduler (warmup={warmup_steps}, total={total_steps})")
    else:
        scheduler = None
    
    # Setup SWA (Stochastic Weight Averaging)
    swa_model = None
    swa_scheduler = None
    if train_config.get('use_swa', False):
        swa_model = AveragedModel(model)
        swa_start = train_config.get('swa_start_epoch', 20)
        swa_lr = train_config.get('swa_lr', 2e-5)
        logger.info(f"✓ SWA enabled (start_epoch={swa_start}, lr={swa_lr})")
    
    # Initialize wandb
    logger.info("\n" + "=" * 80)
    logger.info("Initializing Weights & Biases...")
    logger.info("=" * 80)
    
    if train_config.get('use_wandb', False):
        try:
            import wandb
            wandb.init(
                project=config['wandb'].get('project', 'nlp-medical-mt'),
                name=config['wandb'].get('name', 'medical_finetune'),
                config=config,
                tags=config['wandb'].get('tags', [])
            )
            logger.info("✓ Wandb initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
    
    # Training loop
    logger.info("\n" + "=" * 80)
    logger.info("Starting fine-tuning...")
    logger.info("=" * 80)
    
    from src.training.loss import LabelSmoothingLoss
    criterion = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        padding_idx=tgt_vocab.pad_idx,
        smoothing=train_config.get('label_smoothing', 0.1)
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0
    
    for epoch in range(train_config['epochs']):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_config['epochs']}")
        
        for batch_idx, (src, tgt, src_lengths, tgt_lengths) in enumerate(progress_bar):
            src = src.to(device)
            tgt = tgt.to(device)
            
            # Split target into input and output (teacher forcing)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Forward pass
            output = model(src, tgt_input)
            loss = criterion(
                output.contiguous().view(-1, output.size(-1)), 
                tgt_output.contiguous().view(-1)
            )
            loss = loss / train_config['gradient_accumulation_steps']
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % train_config['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config['max_grad_norm'])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Update SWA model
                if swa_model is not None and epoch >= train_config.get('swa_start_epoch', 20):
                    swa_model.update_parameters(model)
            
            epoch_loss += loss.item() * train_config['gradient_accumulation_steps']
            
            # Logging
            if global_step % train_config.get('log_every', 50) == 0:
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{current_lr:.2e}"
                })
                
                if train_config.get('use_wandb', False):
                    try:
                        wandb.log({
                            'train/loss': loss.item() * train_config['gradient_accumulation_steps'],
                            'train/lr': current_lr,
                            'train/step': global_step
                        })
                    except:
                        pass
            
            # Validation
            if global_step % train_config.get('eval_every', 250) == 0:
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for val_src, val_tgt, val_src_len, val_tgt_len in val_loader:
                        val_src = val_src.to(device)
                        val_tgt = val_tgt.to(device)
                        
                        # Split target
                        val_tgt_input = val_tgt[:, :-1]
                        val_tgt_output = val_tgt[:, 1:]
                        
                        val_output = model(val_src, val_tgt_input)
                        val_loss += criterion(
                            val_output.contiguous().view(-1, val_output.size(-1)),
                            val_tgt_output.contiguous().view(-1)
                        ).item()
                
                val_loss /= len(val_loader)
                logger.info(f"Step {global_step}: Val Loss = {val_loss:.4f}")
                
                if train_config.get('use_wandb', False):
                    try:
                        wandb.log({'val/loss': val_loss, 'val/step': global_step})
                    except:
                        pass
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'global_step': global_step,
                        'best_val_loss': best_val_loss,
                        'config': config
                    }, os.path.join(config['paths']['checkpoint_dir'], 'best_model.pt'))
                    logger.info(f"✓ Saved best model (val_loss={val_loss:.4f})")
                else:
                    patience_counter += 1
                
                model.train()
                
                # Early stopping
                if patience_counter >= train_config.get('early_stopping_patience', 8):
                    logger.info(f"Early stopping triggered at step {global_step}")
                    break
        
        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{train_config['epochs']} - Avg Loss: {avg_epoch_loss:.4f}")
        
        if patience_counter >= train_config.get('early_stopping_patience', 8):
            break
    
    # Merge LoRA weights and save final model
    logger.info("\n" + "=" * 80)
    logger.info("Merging LoRA weights...")
    logger.info("=" * 80)
    merge_lora_weights(model)
    
    final_checkpoint = os.path.join(config['paths']['checkpoint_dir'], 'final_model_merged.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, final_checkpoint)
    logger.info(f"✓ Saved merged model: {final_checkpoint}")
    
    # Save SWA model if used
    if swa_model is not None:
        swa_checkpoint = config['paths'].get('swa_checkpoint', 'swa_model.pt')
        torch.save({
            'model_state_dict': swa_model.module.state_dict(),
            'config': config
        }, swa_checkpoint)
        logger.info(f"✓ Saved SWA model: {swa_checkpoint}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Medical fine-tuning completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

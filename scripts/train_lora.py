"""
Training script with LoRA support for efficient fine-tuning.
Uses parameter-efficient LoRA adapters on attention layers.
"""
import argparse
import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.helpers import set_seed, get_device
from src.data.vocabulary import Vocabulary
from src.data.dataset import TranslationDataset, create_dataloader
from src.training.trainer import Trainer
from src.models.transformer import Transformer


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for efficient fine-tuning."""
    
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA weights
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scaling


def inject_lora(model, rank=8, alpha=16, target_modules=['query', 'value']):
    """Inject LoRA layers into transformer attention modules."""
    lora_params = 0
    
    for name, module in model.named_modules():
        # Target attention projection layers
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Add LoRA adapter
                lora = LoRALayer(module.in_features, module.out_features, rank, alpha)
                
                # Freeze original layer
                module.weight.requires_grad = False
                if module.bias is not None:
                    module.bias.requires_grad = False
                
                # Attach LoRA
                module.lora_adapter = lora
                
                # Override forward
                original_forward = module.forward
                def new_forward(x, orig_fwd=original_forward, lora=lora):
                    return orig_fwd(x) + lora(x)
                module.forward = new_forward
                
                lora_params += sum(p.numel() for p in lora.parameters())
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"LoRA injection complete:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  LoRA parameters: {lora_params:,}")
    print(f"  Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Transformer with LoRA")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup
    set_seed(config.get('seed', 42))
    device = get_device(config.get('device', 'cuda'))
    
    # Setup logger
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    logger = setup_logger("train_lora", log_file=os.path.join(config['paths']['log_dir'], 'train.log'))
    
    logger.info("=" * 80)
    logger.info("Training with LoRA")
    logger.info("=" * 80)
    
    # Load vocabularies
    vocab_dir = config['paths']['vocab_dir']
    src_vocab = Vocabulary.load(os.path.join(vocab_dir, 'src_vocab.json'))
    tgt_vocab = Vocabulary.load(os.path.join(vocab_dir, 'tgt_vocab.json'))
    
    logger.info(f"Loaded vocabularies:")
    logger.info(f"  Source: {len(src_vocab):,} tokens")
    logger.info(f"  Target: {len(tgt_vocab):,} tokens")
    
    # Create datasets
    data_config = config['data']
    train_dataset = TranslationDataset(
        src_file=data_config['train_src'],
        tgt_file=data_config['train_tgt'],
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_length=data_config.get('max_seq_length', 128)
    )
    
    val_dataset = TranslationDataset(
        src_file=data_config.get('val_src') or data_config['train_src'],
        tgt_file=data_config.get('val_tgt') or data_config['train_tgt'],
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_length=data_config.get('max_seq_length', 128)
    )
    
    logger.info(f"Datasets:")
    logger.info(f"  Train: {len(train_dataset):,} samples")
    logger.info(f"  Val: {len(val_dataset):,} samples")
    
    # Create dataloaders
    train_config = config['training']
    train_loader = create_dataloader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        pad_idx=src_vocab.pad_idx
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        pad_idx=src_vocab.pad_idx
    )
    
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
    
    # Load pretrained weights if specified
    if train_config.get('resume_from'):
        logger.info(f"Loading pretrained weights from {train_config['resume_from']}")
        checkpoint = torch.load(train_config['resume_from'], map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Pretrained weights loaded")
    
    # Inject LoRA
    lora_config = config.get('lora', {})
    if lora_config.get('enabled', False):
        logger.info("Injecting LoRA adapters...")
        model = inject_lora(
            model,
            rank=lora_config.get('rank', 8),
            alpha=lora_config.get('alpha', 16),
            target_modules=lora_config.get('target_modules', ['query', 'value'])
        )
    
    model = model.to(device)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        tgt_vocab=tgt_vocab
    )
    
    # Train
    logger.info("\nStarting training...")
    trainer.train()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

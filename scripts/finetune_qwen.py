"""
VLSP 2025 Medical Translation - Complete Pipeline
Preprocessing → Data Splitting → Fine-tuning Qwen2.5-3B → Evaluation

This is an isolated, production-ready script for the VLSP 2025 shared task.
"""
import argparse
import os
import sys
import json
import torch
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import List, Tuple, Dict

from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import evaluate

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Data paths
    'data': {
        'train_src': '../MedicalDataset_VLSP/train.vi.txt',
        'train_tgt': '../MedicalDataset_VLSP/train.en.txt',
        'test_src': '../MedicalDataset_VLSP/public_test.vi.txt',
        'test_tgt': '../MedicalDataset_VLSP/public_test.en.txt',
        'src_lang': 'vi',
        'tgt_lang': 'en',
        'val_split': 0.05,  # 5% for validation
        'max_length': 512,
        'preprocessing': {
            'lowercase': False,  # Keep original case for medical terms
            'remove_extra_spaces': True,
            'min_length': 3,  # Min tokens
            'max_length': 500,  # Max tokens
        }
    },
    
    # Model configuration
    'model': {
        'name': 'Qwen/Qwen2.5-0.5B',  # Best from benchmark
        'torch_dtype': 'float16',
    },
    
    # LoRA configuration for efficient fine-tuning
    'lora': {
        'r': 32,  # Higher rank for better quality
        'alpha': 64,
        'dropout': 0.05,
        'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        'bias': 'none',
        'task_type': 'CAUSAL_LM'
    },
    
    # Training configuration
    'training': {
        'num_epochs': 1,
        'batch_size': 4,
        'gradient_accumulation_steps': 8,  # Effective batch = 32
        'learning_rate': 2e-4,
        'weight_decay': 0.01,
        'warmup_ratio': 0.03,
        'lr_scheduler': 'cosine',
        'max_grad_norm': 1.0,
        'fp16': True,
        'optim': 'adamw_torch',
        'logging_steps': 10,
        'eval_steps': 200,
        'save_steps': 200,
        'save_total_limit': 3,
    },
    
    # Inference configuration
    'inference': {
        'max_new_tokens': 256,
        'num_beams': 5,
        'temperature': 0.8,
        'do_sample': False,
        'top_p': 0.9,
        'repetition_penalty': 1.1,
    },
    
    # Output paths
    'output': {
        'base_dir': 'experiments/vlsp_qwen3b_vi2en',
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs',
        'predictions_dir': 'predictions',
        'data_dir': 'processed_data',
    },
    
    # Weights & Biases logging
    'wandb': {
        'enabled': True,
        'project': 'vlsp2025-medical-mt',
        'name': 'qwen3b_vi2en',
        'entity': None,  # Your wandb username/team
        'tags': ['qwen2.5-3b', 'medical', 'vi2en', 'vlsp2025', 'lora'],
        'notes': 'Medical translation Vi→En using Qwen2.5-3B with LoRA fine-tuning',
    }
}


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def preprocess_line(line: str, config: dict) -> str:
    """Preprocess a single line."""
    text = line.strip()
    
    if config['remove_extra_spaces']:
        text = ' '.join(text.split())
    
    return text


def preprocess_dataset(src_file: str, tgt_file: str, config: dict) -> Tuple[List[str], List[str]]:
    """Preprocess and filter parallel da taset."""
    logger.info(f"Preprocessing dataset: {src_file}, {tgt_file}")
    
    with open(src_file, 'r', encoding='utf-8') as f:
        src_lines_raw = f.readlines()
    
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_lines_raw = f.readlines()
    
    # Handle mismatched file lengths - use minimum
    if len(src_lines_raw) != len(tgt_lines_raw):
        logger.warning(f"Line count mismatch: {len(src_lines_raw)} src vs {len(tgt_lines_raw)} tgt")
        min_len = min(len(src_lines_raw), len(tgt_lines_raw))
        logger.warning(f"Using first {min_len} parallel lines")
        src_lines_raw = src_lines_raw[:min_len]
        tgt_lines_raw = tgt_lines_raw[:min_len]
    
    src_lines = [preprocess_line(line, config) for line in src_lines_raw]
    tgt_lines = [preprocess_line(line, config) for line in tgt_lines_raw]
    
    # Filter by length
    filtered_pairs = []
    for src, tgt in zip(src_lines, tgt_lines):
        src_tokens = len(src.split())
        tgt_tokens = len(tgt.split())
        
        if (src_tokens >= config['min_length'] and tgt_tokens >= config['min_length'] and
            src_tokens <= config['max_length'] and tgt_tokens <= config['max_length'] and
            src.strip() and tgt.strip()):
            filtered_pairs.append((src, tgt))
    
    logger.info(f"Kept {len(filtered_pairs)}/{len(src_lines)} pairs after filtering")
    
    src_filtered = [pair[0] for pair in filtered_pairs]
    tgt_filtered = [pair[1] for pair in filtered_pairs]
    
    return src_filtered, tgt_filtered


def split_data(src_lines: List[str], tgt_lines: List[str], val_ratio: float, seed: int = 42) -> Dict:
    """Split data into train and validation sets."""
    logger.info(f"Splitting data: {len(src_lines)} pairs, val_ratio={val_ratio}")
    
    # Shuffle with seed for reproducibility
    indices = list(range(len(src_lines)))
    random.Random(seed).shuffle(indices)
    
    split_idx = int(len(indices) * (1 - val_ratio))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    splits = {
        'train_src': [src_lines[i] for i in train_indices],
        'train_tgt': [tgt_lines[i] for i in train_indices],
        'val_src': [src_lines[i] for i in val_indices],
        'val_tgt': [tgt_lines[i] for i in val_indices],
    }
    
    logger.info(f"Train: {len(splits['train_src'])}, Val: {len(splits['val_src'])}")
    
    return splits


def save_splits(splits: Dict, output_dir: str):
    """Save data splits to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for key, lines in splits.items():
        filepath = os.path.join(output_dir, f'{key}.txt')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        logger.info(f"Saved {key}: {filepath}")


# ============================================================================
# DATASET
# ============================================================================

class MedicalTranslationDataset(Dataset):
    """Dataset for medical translation with instruction-based prompting."""
    
    def __init__(self, src_lines: List[str], tgt_lines: List[str], 
                 tokenizer, src_lang: str, tgt_lang: str, max_length: int = 512):
        self.src_lines = src_lines
        self.tgt_lines = tgt_lines
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length
        
        # Create instruction template
        if src_lang == 'vi':
            self.instruction_template = "Translate the following Vietnamese medical text to English:\n\n{}\n\nTranslation:"
        else:
            self.instruction_template = "Translate the following English medical text to Vietnamese:\n\n{}\n\nTranslation:"
    
    def __len__(self):
        return len(self.src_lines)
    
    def __getitem__(self, idx):
        src_text = self.src_lines[idx]
        tgt_text = self.tgt_lines[idx]
        
        # Create instruction prompt
        instruction = self.instruction_template.format(src_text)
        full_text = instruction + " " + tgt_text
        
        # Tokenize
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Tokenize instruction only (to mask it in loss)
        instruction_encodings = self.tokenizer(
            instruction,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False
        )
        
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        
        # Create labels - mask instruction part
        labels = input_ids.clone()
        instruction_len = len(instruction_encodings['input_ids'])
        labels[:instruction_len] = -100  # Don't compute loss on instruction
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_and_tokenizer(config: dict):
    """Load Qwen model with LoRA for fine-tuning."""
    model_name = config['model']['name']
    logger.info(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side='right'
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if config['model']['torch_dtype'] == 'float16' else torch.float32,
        device_map='auto'
    )
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['alpha'],
        lora_dropout=config['lora']['dropout'],
        target_modules=config['lora']['target_modules'],
        bias=config['lora']['bias'],
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


# ============================================================================
# TRAINING
# ============================================================================

class BLEUEvaluationCallback(TrainerCallback):
    """Callback to compute BLEU during training."""
    
    def __init__(self, eval_dataset, tokenizer, src_lang, tgt_lang, config):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.config = config
        self.bleu = evaluate.load('sacrebleu')
    
    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        """Compute BLEU on validation set."""
        if metrics is not None:
            # Sample a few translations for BLEU
            model.eval()
            predictions = []
            references = []
            
            # Sample 100 examples for speed
            sample_size = min(100, len(self.eval_dataset.src_lines))
            indices = random.sample(range(len(self.eval_dataset.src_lines)), sample_size)
            
            for idx in indices:
                src_text = self.eval_dataset.src_lines[idx]
                ref_text = self.eval_dataset.tgt_lines[idx]
                
                pred_text = translate_single(
                    model, self.tokenizer, src_text,
                    self.src_lang, self.tgt_lang, self.config['inference']
                )
                
                predictions.append(pred_text)
                references.append([ref_text])
            
            bleu_score = self.bleu.compute(predictions=predictions, references=references)
            metrics['eval_bleu'] = bleu_score['score']
            logger.info(f"Validation BLEU: {bleu_score['score']:.2f}")


def train_model(model, tokenizer, train_dataset, val_dataset, config: dict):
    """Train the model."""
    output_dir = os.path.join(config['output']['base_dir'], config['output']['checkpoint_dir'])
    log_dir = os.path.join(config['output']['base_dir'], config['output']['log_dir'])
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    training_config = config['training']
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config['num_epochs'],
        per_device_train_batch_size=training_config['batch_size'],
        per_device_eval_batch_size=training_config['batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        warmup_ratio=training_config['warmup_ratio'],
        lr_scheduler_type=training_config['lr_scheduler'],
        max_grad_norm=training_config['max_grad_norm'],
        fp16=training_config['fp16'],
        optim=training_config['optim'],
        logging_dir=log_dir,
        logging_steps=training_config['logging_steps'],
        eval_strategy='steps',
        eval_steps=training_config['eval_steps'],
        save_strategy='steps',
        save_steps=training_config['save_steps'],
        save_total_limit=training_config['save_total_limit'],
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        report_to='wandb' if config.get('wandb', {}).get('enabled', False) else 'none',
        remove_unused_columns=False,
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[BLEUEvaluationCallback(
            val_dataset, tokenizer,
            config['data']['src_lang'],
            config['data']['tgt_lang'],
            config
        )]
    )
    
    # Train
    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)
    trainer.train()
    
    # Save final model
    final_output_dir = os.path.join(config['output']['base_dir'], 'final_model')
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    logger.info(f"✓ Final model saved to {final_output_dir}")
    
    return trainer


# ============================================================================
# INFERENCE & EVALUATION
# ============================================================================

def translate_single(model, tokenizer, text: str, src_lang: str, tgt_lang: str, inf_config: dict) -> str:
    """Translate a single sentence."""
    # Create prompt
    if src_lang == 'vi':
        prompt = f"Translate the following Vietnamese medical text to English:\n\n{text}\n\nTranslation:"
    else:
        prompt = f"Translate the following English medical text to Vietnamese:\n\n{text}\n\nTranslation:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=inf_config['max_new_tokens'],
            num_beams=inf_config['num_beams'],
            temperature=inf_config.get('temperature', 1.0),
            do_sample=inf_config['do_sample'],
            top_p=inf_config.get('top_p', 1.0),
            repetition_penalty=inf_config.get('repetition_penalty', 1.0),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    translation = full_output[len(prompt):].strip()
    
    return translation


def translate_file(model, tokenizer, src_file: str, output_file: str, 
                  src_lang: str, tgt_lang: str, inf_config: dict):
    """Translate entire file."""
    logger.info(f"Translating {src_file} → {output_file}")
    
    with open(src_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f]
    
    translations = []
    model.eval()
    
    for line in tqdm(lines, desc="Translating"):
        if line:
            translation = translate_single(model, tokenizer, line, src_lang, tgt_lang, inf_config)
            translations.append(translation)
        else:
            translations.append('')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(translations) + '\n')
    
    logger.info(f"✓ Saved translations to {output_file}")


def evaluate_bleu(predictions_file: str, references_file: str) -> float:
    """Compute BLEU score."""
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = [line.strip() for line in f]
    
    with open(references_file, 'r', encoding='utf-8') as f:
        references = [[line.strip()] for line in f]
    
    bleu = evaluate.load('sacrebleu')
    result = bleu.compute(predictions=predictions, references=references)
    
    return result


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="VLSP 2025 Medical Translation Pipeline")
    parser.add_argument('--mode', choices=['all', 'preprocess', 'train', 'evaluate'], 
                       default='all', help='Pipeline mode')
    parser.add_argument('--config_file', type=str, help='Custom config JSON file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--wandb_key', type=str, default=None, help='Weights & Biases API key')
    parser.add_argument('--wandb_project', type=str, default=None, help='W&B project name')
    parser.add_argument('--wandb_name', type=str, default=None, help='W&B run name')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load custom config if provided
    config = CONFIG.copy()
    if args.no_wandb:
        config['wandb']['enabled'] = False
    if args.wandb_project:
        config['wandb']['project'] = args.wandb_project
    if args.wandb_name:
        config['wandb']['name'] = args.wandb_name
    
    # Initialize wandb if enabled
    if config['wandb']['enabled'] and args.mode in ['all', 'train']:
        try:
            import wandb
            
            # Set API key if provided
            if args.wandb_key:
                wandb.login(key=args.wandb_key)
            elif os.getenv('WANDB_API_KEY'):
                wandb.login(key=os.getenv('WANDB_API_KEY'))
            
            wandb_config = config['wandb']
            wandb.init(
                project=wandb_config['project'],
                name=wandb_config['name'],
                entity=wandb_config.get('entity'),
                tags=wandb_config.get('tags', []),
                notes=wandb_config.get('notes', ''),
                config={
                    'model': config['model'],
                    'lora': config['lora'],
                    'training': config['training'],
                    'data': {
                        'src_lang': config['data']['src_lang'],
                        'tgt_lang': config['data']['tgt_lang'],
                        'val_split': config['data']['val_split'],
                    }
                }
            )
            logger.info(f"✓ Wandb initialized: {wandb_config['project']}/{wandb_config['name']}")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            config['wandb']['enabled'] = False
    
    #   with open(args.config_file, 'r') as f:
            custom_config = json.load(f)
            config.update(custom_config)
    
    # Create output directories
    os.makedirs(config['output']['base_dir'], exist_ok=True)
    
    # Save config
    config_path = os.path.join(config['output']['base_dir'], 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved to {config_path}")
    
    # ========================================================================
    # STEP 1: PREPROCESSING
    # ========================================================================
    if args.mode in ['all', 'preprocess']:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: DATA PREPROCESSING")
        logger.info("=" * 80)
        
        # Preprocess training data
        src_lines, tgt_lines = preprocess_dataset(
            config['data']['train_src'],
            config['data']['train_tgt'],
            config['data']['preprocessing']
        )
        
        # Split into train/val
        splits = split_data(src_lines, tgt_lines, config['data']['val_split'], args.seed)
        
        # Save splits
        data_output_dir = os.path.join(config['output']['base_dir'], config['output']['data_dir'])
        save_splits(splits, data_output_dir)
        
        # Update config with processed data paths
        config['data']['processed'] = {
            'train_src': os.path.join(data_output_dir, 'train_src.txt'),
            'train_tgt': os.path.join(data_output_dir, 'train_tgt.txt'),
            'val_src': os.path.join(data_output_dir, 'val_src.txt'),
            'val_tgt': os.path.join(data_output_dir, 'val_tgt.txt'),
        }
    
    # ========================================================================
    # STEP 2: TRAINING
    # ========================================================================
    if args.mode in ['all', 'train']:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: MODEL FINE-TUNING")
        logger.info("=" * 80)
        
        # Load model
        model, tokenizer = load_model_and_tokenizer(config)
        
        # Load processed data
        data_paths = config['data']['processed']
        with open(data_paths['train_src'], 'r') as f:
            train_src = [line.strip() for line in f]
        with open(data_paths['train_tgt'], 'r') as f:
            train_tgt = [line.strip() for line in f]
        with open(data_paths['val_src'], 'r') as f:
            val_src = [line.strip() for line in f]
        with open(data_paths['val_tgt'], 'r') as f:
            val_tgt = [line.strip() for line in f]
        
        # Create datasets
        train_dataset = MedicalTranslationDataset(
            train_src, train_tgt, tokenizer,
            config['data']['src_lang'], config['data']['tgt_lang'],
            config['data']['max_length']
        )
        val_dataset = MedicalTranslationDataset(
            val_src, val_tgt, tokenizer,
            config['data']['src_lang'], config['data']['tgt_lang'],
            config['data']['max_length']
        )
        
        # Train
        trainer = train_model(model, tokenizer, train_dataset, val_dataset, config)
    
    # ========================================================================
    # STEP 3: EVALUATION
    # ========================================================================
    if args.mode in ['all', 'evaluate']:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: EVALUATION")
        logger.info("=" * 80)
        
        # Load fine-tuned model
        model_path = os.path.join(config['output']['base_dir'], 'final_model')
        logger.info(f"Loading model from {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            config['model']['name'],
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map='auto'
        )
    
    # Finish wandb run
    if config['wandb']['enabled']:
        try:
            import wandb
            wandb.finish()
        except:
            pass
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()
        
        # Translate test set
        predictions_dir = os.path.join(config['output']['base_dir'], config['output']['predictions_dir'])
        os.makedirs(predictions_dir, exist_ok=True)
        
        predictions_file = os.path.join(predictions_dir, 'public_test_predictions.txt')
        translate_file(
            model, tokenizer,
            config['data']['test_src'],
            predictions_file,
            config['data']['src_lang'],
            config['data']['tgt_lang'],
            config['inference']
        )
        
        # Compute BLEU
        bleu_result = evaluate_bleu(predictions_file, config['data']['test_tgt'])
        
        logger.info("\n" + "=" * 80)
        logger.info("FINAL RESULTS")
        logger.info("=" * 80)
        logger.info(f"BLEU Score: {bleu_result['score']:.2f}")
        logger.info(f"Predictions: {predictions_file}")
        
        # Save results
        results = {
            'bleu_score': bleu_result['score'],
            'bleu_details': bleu_result,
            'predictions_file': predictions_file,
            'timestamp': datetime.now().isoformat()
        }
        
        results_file = os.path.join(config['output']['base_dir'], 'results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_file}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ PIPELINE COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

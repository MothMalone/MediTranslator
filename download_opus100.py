"""
Download OPUS-100 en-vi dataset from HuggingFace
"""
from datasets import load_dataset
import os
import argparse


def download_opus100(output_dir="MainProblem/data/raw", language_pair=("en", "vi")):
    """
    Download OPUS-100 dataset and save to files
    
    Args:
        output_dir: Directory to save the data
        language_pair: Tuple of (source_lang, target_lang)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Downloading OPUS-100 En-Vi Dataset")
    print("=" * 60)
    
    src_lang, tgt_lang = language_pair
    
    # Load dataset from HuggingFace
    print(f"\nLoading OPUS-100 {src_lang}-{tgt_lang} dataset from HuggingFace...")
    
    try:
        dataset = load_dataset("Helsinki-NLP/opus-100", f"{src_lang}-{tgt_lang}")
        print(f"✓ Dataset loaded")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        print("\nTrying alternative format...")
        try:
            dataset = load_dataset("opus100", f"{src_lang}-{tgt_lang}")
            print(f"✓ Dataset loaded")
        except Exception as e2:
            print(f"✗ Error: {e2}")
            return
    
    print(f"  Train: {len(dataset['train'])} pairs")
    print(f"  Validation: {len(dataset['validation'])} pairs")
    print(f"  Test: {len(dataset['test'])} pairs")
    
    # Save to files
    splits = {
        'train': dataset['train'],
        'val': dataset['validation'],
        'public_test': dataset['test']
    }
    
    for split_name, split_data in splits.items():
        src_file = os.path.join(output_dir, f"{split_name}.{src_lang}.txt")
        tgt_file = os.path.join(output_dir, f"{split_name}.{tgt_lang}.txt")
        
        print(f"\nSaving {split_name}...")
        with open(src_file, 'w', encoding='utf-8') as f_src, \
             open(tgt_file, 'w', encoding='utf-8') as f_tgt:
            
            for example in split_data:
                translation = example['translation']
                src_text = translation[src_lang].strip()
                tgt_text = translation[tgt_lang].strip()
                
                # Only save non-empty pairs
                if src_text and tgt_text:
                    f_src.write(src_text + '\n')
                    f_tgt.write(tgt_text + '\n')
        
        # Count lines
        with open(src_file, 'r', encoding='utf-8') as f:
            src_lines = len(f.readlines())
        with open(tgt_file, 'r', encoding='utf-8') as f:
            tgt_lines = len(f.readlines())
        
        print(f"  ✓ {src_file} ({src_lines:,} lines)")
        print(f"  ✓ {tgt_file} ({tgt_lines:,} lines)")
    
    print("\n" + "=" * 60)
    print("Dataset downloaded and saved successfully!")
    print("=" * 60)
    print(f"\nFiles saved in: {output_dir}/")
    print(f"  train.{src_lang}.txt / train.{tgt_lang}.txt")
    print(f"  val.{src_lang}.txt / val.{tgt_lang}.txt")
    print(f"  public_test.{src_lang}.txt / public_test.{tgt_lang}.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download OPUS-100 dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="MainProblem/data/raw_opus100",
        help="Output directory for data files"
    )
    parser.add_argument(
        "--src-lang",
        type=str,
        default="en",
        help="Source language (en or vi)"
    )
    parser.add_argument(
        "--tgt-lang",
        type=str,
        default="vi",
        help="Target language (vi or en)"
    )
    
    args = parser.parse_args()
    
    download_opus100(args.output_dir, (args.src_lang, args.tgt_lang))

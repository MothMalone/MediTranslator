"""
Download IWSLT mt_eng_vietnamese dataset from HuggingFace
"""
from datasets import load_dataset
import os
import argparse


def download_iwslt(output_dir="data/raw_iwslt", language_pair=("vi", "en")):
    """
    Download IWSLT mt_eng_vietnamese dataset and save to files
    
    Args:
        output_dir: Directory to save the data
        language_pair: Tuple of (source_lang, target_lang)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Downloading IWSLT mt_eng_vietnamese Dataset")
    print("=" * 60)
    
    src_lang, tgt_lang = language_pair
    
    # Determine config name based on language pair
    config_name = f"iwslt2015-{src_lang}-{tgt_lang}"
    
    # Load dataset from HuggingFace
    print(f"\nLoading IWSLT mt_eng_vietnamese dataset from HuggingFace...")
    print(f"Config: {config_name}")
    print("Note: This dataset requires trust_remote_code=True")
    
    try:
        # IWSLT mt_eng_vietnamese dataset from HuggingFace
        dataset = load_dataset("IWSLT/mt_eng_vietnamese", config_name, trust_remote_code=True)
        print(f"✓ Dataset loaded")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        print("\nThe external download URL may be unavailable.")
        print("\nTrying alternative: iwslt2017 dataset...")
        try:
            # Try iwslt2017 as alternative
            dataset = load_dataset("iwslt2017", "iwslt2017-en-vi")
            print(f"✓ Alternative dataset loaded (iwslt2017)")
            # Adjust language codes for iwslt2017
            src_lang = "en"
            tgt_lang = "vi"
        except Exception as e2:
            print(f"✗ Alternative also failed: {e2}")
            print("\nPlease try one of these alternatives:")
            print("  1. Use OPUS-100: python download_opus100.py")
            print("  2. Download manually from: https://nlp.stanford.edu/projects/nmt/")
            print("  3. Try again later if Stanford's server is temporarily down")
            return
    
    # Check available splits
    print(f"\nAvailable splits: {list(dataset.keys())}")
    for split in dataset.keys():
        print(f"  {split}: {len(dataset[split])} pairs")
    
    # Save to files
    splits_mapping = {
        'train': 'train',
        'validation': 'val',
        'test': 'public_test'
    }
    
    for dataset_split, file_split in splits_mapping.items():
        if dataset_split not in dataset:
            print(f"\nWarning: {dataset_split} split not found, skipping...")
            continue
            
        split_data = dataset[dataset_split]
        src_file = os.path.join(output_dir, f"{file_split}.{src_lang}.txt")
        tgt_file = os.path.join(output_dir, f"{file_split}.{tgt_lang}.txt")
        
        print(f"\nSaving {file_split} (from {dataset_split})...")
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
    
    # Show what files were created
    if os.path.exists(os.path.join(output_dir, f"train.{src_lang}.txt")):
        print(f"  train.{src_lang}.txt / train.{tgt_lang}.txt")
    if os.path.exists(os.path.join(output_dir, f"val.{src_lang}.txt")):
        print(f"  val.{src_lang}.txt / val.{tgt_lang}.txt")
    if os.path.exists(os.path.join(output_dir, f"public_test.{src_lang}.txt")):
        print(f"  public_test.{src_lang}.txt / public_test.{tgt_lang}.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download IWSLT mt_eng_vietnamese dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw_iwslt",
        help="Output directory for data files"
    )
    parser.add_argument(
        "--src-lang",
        type=str,
        default="vi",
        help="Source language (vi or en)"
    )
    parser.add_argument(
        "--tgt-lang",
        type=str,
        default="en",
        help="Target language (en or vi)"
    )
    
    args = parser.parse_args()
    
    download_iwslt(args.output_dir, (args.src_lang, args.tgt_lang))

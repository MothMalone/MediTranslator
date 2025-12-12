"""
Data Check Script
Verify data files and show statistics.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_file(filepath):
    """Check if file exists and show stats."""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return True, len(lines)
    return False, 0


def main():
    print("=" * 60)
    print("DATA FILE CHECK")
    print("=" * 60)
    
    data_dir = "data/raw"
    
    # Expected files based on what you have
    files = [
        "train.vi.txt",
        "train.en.txt",
        "public_test.vi.txt",
        "public_test.en.txt"
    ]
    
    print(f"\nChecking data directory: {data_dir}\n")
    
    all_exist = True
    stats = {}
    
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        exists, count = check_file(filepath)
        stats[filename] = (exists, count)
        
        status = "✅" if exists else "❌"
        print(f"{status} {filename:25s} ", end="")
        
        if exists:
            print(f"({count:,} lines)")
        else:
            print("NOT FOUND")
            all_exist = False
    
    print("\n" + "=" * 60)
    
    if all_exist:
        print("✅ All required data files found!")
        print("\nData Summary:")
        print(f"  Training set:   {stats['train.vi.txt'][1]:,} sentence pairs")
        print(f"  Test set:       {stats['public_test.vi.txt'][1]:,} sentence pairs")
        
        # Check if files are aligned
        if stats['train.vi.txt'][1] != stats['train.en.txt'][1]:
            print(f"\n⚠️  WARNING: Training files not aligned!")
            print(f"     train.vi.txt: {stats['train.vi.txt'][1]} lines")
            print(f"     train.en.txt: {stats['train.en.txt'][1]} lines")
        
        if stats['public_test.vi.txt'][1] != stats['public_test.en.txt'][1]:
            print(f"\n⚠️  WARNING: Test files not aligned!")
            print(f"     public_test.vi.txt: {stats['public_test.vi.txt'][1]} lines")
            print(f"     public_test.en.txt: {stats['public_test.en.txt'][1]} lines")
        
        print("\nNote: Since validation files don't exist, training will")
        print("      automatically split 10% of training data for validation.")
        
    else:
        print("❌ Some data files are missing!")
        print("\nExpected file structure:")
        print("  data/raw/")
        print("    ├── train.vi.txt")
        print("    ├── train.en.txt")
        print("    ├── public_test.vi.txt")
        print("    └── public_test.en.txt")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Calculate BLEU scores for prediction files and compare results
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.bleu import BLEUCalculator
import pandas as pd


def calculate_bleu(predictions_file: str, reference_file: str):
    """Calculate BLEU score for a predictions file."""
    # Load predictions
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = [line.strip() for line in f]
    
    # Load references
    with open(reference_file, 'r', encoding='utf-8') as f:
        references = [line.strip() for line in f]
    
    # Calculate BLEU
    bleu_calc = BLEUCalculator()
    result = bleu_calc.calculate(predictions, references)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Calculate BLEU and compare predictions")
    parser.add_argument(
        "--predictions",
        nargs='+',
        required=True,
        help="Prediction files to evaluate"
    )
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Reference file"
    )
    parser.add_argument(
        "--names",
        nargs='+',
        help="Names for each prediction file (optional)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="bleu_comparison.csv",
        help="Output file for comparison"
    )
    
    args = parser.parse_args()
    
    # Use filenames as names if not provided
    if args.names is None:
        args.names = [os.path.basename(f).replace('.txt', '') for f in args.predictions]
    
    if len(args.names) != len(args.predictions):
        print(f"Error: Number of names ({len(args.names)}) must match predictions ({len(args.predictions)})")
        return
    
    print("\n" + "=" * 80)
    print("BLEU SCORE COMPARISON")
    print("=" * 80)
    print(f"Reference: {args.reference}")
    print("=" * 80)
    
    results = []
    
    for pred_file, name in zip(args.predictions, args.names):
        if not os.path.exists(pred_file):
            print(f"⚠️  File not found: {pred_file}")
            continue
        
        print(f"\nEvaluating: {name}")
        print(f"  File: {pred_file}")
        
        result = calculate_bleu(pred_file, args.reference)
        
        row = {
            'Model': name,
            'File': os.path.basename(pred_file),
            'BLEU': f"{result['bleu']:.2f}",
            '1-gram': f"{result['precisions'][0]:.1f}" if 'precisions' in result else '-',
            '2-gram': f"{result['precisions'][1]:.1f}" if 'precisions' in result else '-',
            '3-gram': f"{result['precisions'][2]:.1f}" if 'precisions' in result else '-',
            '4-gram': f"{result['precisions'][3]:.1f}" if 'precisions' in result else '-',
            'BP': f"{result['bp']:.3f}" if 'bp' in result else '-',
            'Length Ratio': f"{result['ratio']:.3f}" if 'ratio' in result else '-',
        }
        
        print(f"  BLEU: {result['bleu']:.2f}")
        if 'precisions' in result:
            print(f"    1-gram: {result['precisions'][0]:.1f}")
            print(f"    2-gram: {result['precisions'][1]:.1f}")
            print(f"    3-gram: {result['precisions'][2]:.1f}")
            print(f"    4-gram: {result['precisions'][3]:.1f}")
        if 'bp' in result:
            print(f"  Brevity Penalty: {result['bp']:.3f}")
        if 'ratio' in result:
            print(f"  Length Ratio: {result['ratio']:.3f}")
        
        results.append(row)
    
    # Create DataFrame and sort by BLEU
    df = pd.DataFrame(results)
    df['BLEU_numeric'] = df['BLEU'].astype(float)
    df = df.sort_values('BLEU_numeric', ascending=False)
    df = df.drop('BLEU_numeric', axis=1)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    
    # Save results
    df.to_csv(args.output, index=False)
    print(f"\n✅ Saved comparison to {args.output}")
    
    # Also save as markdown
    md_file = args.output.replace('.csv', '.md')
    with open(md_file, 'w') as f:
        f.write("# BLEU Score Comparison\n\n")
        f.write(f"**Reference**: `{args.reference}`\n\n")
        f.write(df.to_markdown(index=False))
    print(f"✅ Saved markdown to {md_file}")


if __name__ == "__main__":
    main()

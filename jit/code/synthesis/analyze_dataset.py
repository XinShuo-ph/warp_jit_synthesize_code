#!/usr/bin/env python3
"""
Analyze M5 dataset and generate statistics
"""
import json
import os


def analyze_dataset(filepath):
    """Analyze dataset and compute statistics"""
    with open(filepath, 'r') as f:
        pairs = json.load(f)
    
    print("=" * 70)
    print("Dataset Statistics")
    print("=" * 70)
    
    # Basic counts
    total = len(pairs)
    print(f"\nTotal training pairs: {total}")
    
    # Category distribution
    categories = {}
    for pair in pairs:
        cat = pair['metadata']['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nCategory distribution:")
    for cat, count in sorted(categories.items()):
        pct = 100.0 * count / total
        print(f"  {cat:15s}: {count:5d} ({pct:5.1f}%)")
    
    # IR size statistics
    jaxpr_lengths = []
    stablehlo_lengths = []
    
    for pair in pairs:
        if 'jaxpr' in pair:
            jaxpr_lengths.append(len(pair['jaxpr']))
        if 'stablehlo' in pair:
            stablehlo_lengths.append(len(pair['stablehlo']))
    
    if jaxpr_lengths:
        print(f"\nJaxpr statistics:")
        print(f"  Mean length: {sum(jaxpr_lengths)/len(jaxpr_lengths):.1f} chars")
        print(f"  Min length: {min(jaxpr_lengths)} chars")
        print(f"  Max length: {max(jaxpr_lengths)} chars")
    
    if stablehlo_lengths:
        print(f"\nStableHLO statistics:")
        print(f"  Mean length: {sum(stablehlo_lengths)/len(stablehlo_lengths):.1f} chars")
        print(f"  Min length: {min(stablehlo_lengths)} chars")
        print(f"  Max length: {max(stablehlo_lengths)} chars")
    
    # Operations diversity
    operations = {}
    for pair in pairs:
        op = pair['metadata']['operation']
        operations[op] = operations.get(op, 0) + 1
    
    print(f"\nOperation diversity: {len(operations)} unique operations")
    
    # Top 10 operations
    print("\nTop 10 most common operations:")
    top_ops = sorted(operations.items(), key=lambda x: x[1], reverse=True)[:10]
    for op, count in top_ops:
        print(f"  {op:20s}: {count:4d}")
    
    return {
        'total': total,
        'categories': categories,
        'operations': len(operations),
        'jaxpr_mean': sum(jaxpr_lengths)/len(jaxpr_lengths) if jaxpr_lengths else 0,
        'stablehlo_mean': sum(stablehlo_lengths)/len(stablehlo_lengths) if stablehlo_lengths else 0,
    }


if __name__ == "__main__":
    dataset_path = "../../data/m5_dataset_final.json"
    stats = analyze_dataset(dataset_path)
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)

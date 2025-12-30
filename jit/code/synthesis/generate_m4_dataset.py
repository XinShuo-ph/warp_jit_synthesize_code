#!/usr/bin/env python3
"""
Generate M4 sample dataset: 100+ training pairs
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from pipeline import SynthesisPipeline


def main():
    print("=" * 70)
    print("Generating M4 Sample Dataset")
    print("=" * 70)
    
    pipeline = SynthesisPipeline(ir_type="both", seed=42)
    
    # Generate diverse batch: 20 from each category
    print("\nGenerating diverse samples (20 per category)...")
    pairs = pipeline.generate_diverse_batch(n_per_category=20, verbose=True)
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"Total pairs generated: {len(pairs)}")
    
    # Category breakdown
    category_counts = {}
    for pair in pairs:
        cat = pair['metadata']['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("\nCategory distribution:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat:15s}: {count:3d}")
    
    # Save
    output_path = "../../data/samples/m4_dataset.json"
    pipeline.save_pairs(pairs, output_path)
    print(f"\nSaved to: {output_path}")
    
    # Validate
    is_valid = pipeline.validate_pairs(pairs)
    print(f"All pairs valid: {is_valid}")
    
    print("\n" + "=" * 70)
    print("SUCCESS: M4 dataset generated!")
    print("=" * 70)


if __name__ == "__main__":
    main()

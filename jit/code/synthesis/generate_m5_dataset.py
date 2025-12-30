#!/usr/bin/env python3
"""
Generate M5 large-scale dataset: 10k+ training pairs
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from batch_generator import BatchGenerator


def main():
    print("=" * 70)
    print("M5: Large-Scale Dataset Generation")
    print("=" * 70)
    
    generator = BatchGenerator(output_dir="../../data", ir_type="both")
    
    # Generate 10k+ pairs with diverse distribution
    # 2000 per category × 6 categories = 12000 pairs
    pairs = generator.generate_diverse_large_dataset(
        n_per_category=2000,
        prefix="m5_dataset"
    )
    
    print("\n" + "=" * 70)
    print("M5 DATASET GENERATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
Generate CUDA samples for all kernel types.
This script creates a comprehensive CUDA dataset for validation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "synthesis"))

import warp as wp
from pipeline import synthesize_batch
import json


def generate_cuda_dataset(samples_per_category=5):
    """Generate CUDA samples for each kernel category."""
    
    wp.init()
    
    categories = [
        "arithmetic",
        "vector",
        "matrix", 
        "control_flow",
        "math",
        "atomic",
        "nested_loop",
        "multi_conditional",
        "combined",
        "scalar_param"
    ]
    
    output_dir = Path(__file__).parent.parent.parent / "data" / "cuda_samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_pairs = []
    stats = {}
    
    print("=" * 70)
    print("CUDA Dataset Generation")
    print("=" * 70)
    print(f"Generating {samples_per_category} samples per category")
    print(f"Total categories: {len(categories)}")
    print(f"Expected samples: {samples_per_category * len(categories)}")
    print()
    
    for cat in categories:
        print(f"Generating {cat}...")
        pairs = synthesize_batch(
            n=samples_per_category,
            categories=[cat],
            seed=hash(cat) % 10000,
            device="cuda"
        )
        
        stats[cat] = len(pairs)
        all_pairs.extend(pairs)
        print(f"  ✓ Generated {len(pairs)} samples\n")
    
    # Save all samples
    print("Saving samples...")
    for i, pair in enumerate(all_pairs):
        cat = pair["metadata"]["category"]
        filename = f"cuda_{cat}_{i:04d}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(pair, f, indent=2)
    
    print(f"✓ Saved {len(all_pairs)} samples to {output_dir}")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("Generation Statistics")
    print("=" * 70)
    for cat, count in sorted(stats.items()):
        print(f"  {cat:20s}: {count:3d} samples")
    
    print(f"\n  {'TOTAL':20s}: {len(all_pairs):3d} samples")
    
    # Create summary file
    summary = {
        "total_samples": len(all_pairs),
        "device": "cuda",
        "categories": stats,
        "output_directory": str(output_dir)
    }
    
    summary_file = output_dir / "dataset_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved to {summary_file}")
    
    return all_pairs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate CUDA training dataset")
    parser.add_argument("-n", "--samples-per-category", type=int, default=5,
                        help="Number of samples per kernel category (default: 5)")
    
    args = parser.parse_args()
    
    generate_cuda_dataset(args.samples_per_category)

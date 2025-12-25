"""
Generate full dataset for M4.

Generates 120 Python→IR pairs for training data.
"""

import sys
sys.path.insert(0, '/workspace/code')

import warp as wp
from synthesis.pipeline import SynthesisPipeline


def main():
    wp.init()
    
    print("="*60)
    print("GENERATING FULL DATASET (120 SAMPLES)")
    print("="*60)
    print()
    
    output_dir = "/workspace/data/samples"
    pipeline = SynthesisPipeline(output_dir)
    
    try:
        # Generate 120 samples
        stats = pipeline.generate_dataset(count=120, quiet=False)
        
        # Save statistics
        import json
        stats_file = f"{output_dir}/dataset_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n✓ Statistics saved to {stats_file}")
        
    finally:
        pipeline.cleanup()
    
    print("\n✓ Dataset generation complete!")


if __name__ == "__main__":
    main()

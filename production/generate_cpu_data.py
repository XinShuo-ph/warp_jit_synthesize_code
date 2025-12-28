#!/usr/bin/env python3
"""
Continuous dataset generator with progress tracking and resumability.
Generates CPU training data efficiently.
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent / "code"))

def count_existing_samples(data_dir):
    """Count existing JSON samples in data directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return 0, 0
    
    json_files = list(data_path.rglob("*.json"))
    total_size = sum(f.stat().st_size for f in json_files)
    return len(json_files), total_size

def generate_batch(output_dir, count, seed):
    """Generate a batch of samples."""
    from pipeline import run_pipeline
    
    print(f"Generating {count} samples with seed {seed}...")
    start = time.time()
    run_pipeline(output_dir, count, seed)
    elapsed = time.time() - start
    print(f"Batch completed in {elapsed:.1f}s ({elapsed/count:.2f}s per sample)")

def main():
    parser = argparse.ArgumentParser(description="Continuous dataset generation")
    parser.add_argument("--target-mb", type=int, default=200, help="Target size in MB")
    parser.add_argument("--batch-size", type=int, default=500, help="Samples per batch")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--start-seed", type=int, default=2000, help="Starting seed")
    
    args = parser.parse_args()
    
    os.chdir(Path(__file__).parent / "cpu" / "code")
    
    target_bytes = args.target_mb * 1024 * 1024
    output_path = Path(__file__).parent / "cpu" / "data" / args.output
    output_path.mkdir(parents=True, exist_ok=True)
    
    batch_num = 0
    seed = args.start_seed
    
    while True:
        # Check current progress
        sample_count, current_size = count_existing_samples(output_path.parent)
        size_mb = current_size / (1024 * 1024)
        progress = (current_size / target_bytes) * 100
        
        print(f"\n=== Progress: {size_mb:.1f} MB / {args.target_mb} MB ({progress:.1f}%) ===")
        print(f"Total samples: {sample_count}")
        
        if current_size >= target_bytes:
            print(f"\n✅ Target reached! Generated {size_mb:.1f} MB in {sample_count} samples")
            break
        
        # Generate next batch
        batch_num += 1
        batch_dir = output_path / f"batch_{batch_num:03d}"
        batch_dir.mkdir(exist_ok=True)
        
        try:
            generate_batch(str(batch_dir), args.batch_size, seed)
            seed += 1
        except KeyboardInterrupt:
            print(f"\n⚠️ Interrupted. Progress saved. Resume with --start-seed {seed}")
            break
        except Exception as e:
            print(f"❌ Error in batch {batch_num}: {e}")
            seed += 1
            continue

if __name__ == "__main__":
    main()

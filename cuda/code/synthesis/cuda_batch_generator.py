"""
CUDA Batch Generator: Generate large datasets of Python→CUDA IR pairs.
"""
import argparse
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from cuda_pipeline import run_cuda_pipeline


def generate_cuda_batch(
    total: int,
    batch_size: int = 100,
    output_dir: str = "/workspace/cuda/data/cuda_dataset",
    seed: int = 42
):
    """
    Generate a large CUDA dataset in batches.
    
    Args:
        total: Total number of pairs to generate
        batch_size: Number of pairs per batch
        output_dir: Output directory
        seed: Base random seed
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("CUDA Batch Generator")
    print("=" * 70)
    print(f"Total pairs: {total}")
    print(f"Batch size: {batch_size}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    all_pairs = []
    num_batches = (total + batch_size - 1) // batch_size
    
    start_time = time.time()
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_count = min(batch_size, total - batch_start)
        
        print(f"\nBatch {batch_idx + 1}/{num_batches} ({batch_start + 1}-{batch_start + batch_count})...")
        
        # Generate batch with different seed
        pairs = run_cuda_pipeline(
            n=batch_count,
            output_dir=f"{output_dir}/batch_{batch_idx:03d}",
            seed=seed + batch_idx * 1000,
            device="cuda"
        )
        
        all_pairs.extend(pairs)
        
        # Save progress
        progress = {
            "total_generated": len(all_pairs),
            "target": total,
            "batches_completed": batch_idx + 1,
            "timestamp": time.time()
        }
        
        with open(output_path / "progress.json", 'w') as f:
            json.dump(progress, f, indent=2)
    
    elapsed = time.time() - start_time
    
    # Generate statistics
    stats = {
        "total_pairs": len(all_pairs),
        "generation_time_seconds": elapsed,
        "pairs_per_second": len(all_pairs) / elapsed if elapsed > 0 else 0,
        "categories": {},
        "device": "cuda"
    }
    
    for pair in all_pairs:
        cat = pair["metadata"]["category"]
        stats["categories"][cat] = stats["categories"].get(cat, 0) + 1
    
    stats_file = output_path / "generation_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Generation Complete")
    print("=" * 70)
    print(f"Total pairs: {len(all_pairs)}")
    print(f"Time: {elapsed:.1f}s ({len(all_pairs)/elapsed:.1f} pairs/sec)")
    print(f"\nCategory distribution:")
    for cat, count in sorted(stats["categories"].items()):
        pct = 100 * count / len(all_pairs)
        print(f"  {cat:15s}: {count:4d} ({pct:5.1f}%)")
    print(f"\nStatistics saved to: {stats_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CUDA Batch Generator")
    parser.add_argument("-n", "--total", type=int, default=1000,
                        help="Total number of pairs to generate")
    parser.add_argument("-b", "--batch-size", type=int, default=100,
                        help="Number of pairs per batch")
    parser.add_argument("-o", "--output", default="/workspace/cuda/data/cuda_dataset",
                        help="Output directory")
    parser.add_argument("-s", "--seed", type=int, default=42,
                        help="Base random seed")
    
    args = parser.parse_args()
    
    generate_cuda_batch(args.total, args.batch_size, args.output, args.seed)

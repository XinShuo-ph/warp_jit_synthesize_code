"""Batch generator for large-scale Python→IR pair generation."""
import os
import sys
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'extraction'))

import jax.numpy as jnp
from generator import (
    generate_unary_function,
    generate_binary_function,
    generate_reduction_function,
    generate_mixed_function,
    generate_matmul_function,
    generate_conditional_function,
    generate_normalize_function,
    GENERATORS,
)
from ir_extractor import extract_jaxpr, extract_hlo


def generate_single_pair(idx: int, seed: int = None) -> Dict:
    """Generate a single pair with a specific index."""
    if seed is not None:
        random.seed(seed + idx)
    
    # Select generator based on index to ensure variety
    gen_idx = idx % len(GENERATORS)
    generator = GENERATORS[gen_idx]
    
    try:
        name, code, fn, args = generator()
        jaxpr = extract_jaxpr(fn, *args)
        hlo = extract_hlo(fn, *args)
        
        input_shapes = []
        for arg in args:
            if hasattr(arg, 'shape'):
                input_shapes.append(list(arg.shape))
            else:
                input_shapes.append(None)
        
        return {
            "name": name,
            "source": code,
            "jaxpr": jaxpr,
            "hlo": hlo,
            "input_shapes": input_shapes,
        }
    except Exception as e:
        return None


def batch_generate(
    n: int,
    output_dir: str,
    seed: int = 42,
    batch_size: int = 1000,
    start_idx: int = 0,
) -> Dict:
    """Generate pairs in batches.
    
    Args:
        n: Total number of pairs to generate
        output_dir: Output directory
        seed: Base random seed
        batch_size: Pairs per batch
        start_idx: Starting index (for resumable generation)
    
    Returns:
        Statistics dict
    """
    os.makedirs(output_dir, exist_ok=True)
    
    total_generated = 0
    total_failed = 0
    start_time = time.time()
    
    num_batches = (n + batch_size - 1) // batch_size
    
    for batch_num in range(num_batches):
        batch_start = start_idx + batch_num * batch_size
        batch_end = min(batch_start + batch_size, start_idx + n)
        batch_count = batch_end - batch_start
        
        print(f"Batch {batch_num + 1}/{num_batches}: generating {batch_count} pairs...")
        
        for idx in range(batch_start, batch_end):
            pair = generate_single_pair(idx, seed)
            
            if pair is not None:
                filepath = os.path.join(output_dir, f"pair_{idx:06d}.json")
                with open(filepath, 'w') as f:
                    json.dump(pair, f, indent=2)
                total_generated += 1
            else:
                total_failed += 1
        
        elapsed = time.time() - start_time
        rate = total_generated / elapsed if elapsed > 0 else 0
        print(f"  Progress: {total_generated}/{n} ({rate:.1f} pairs/sec)")
    
    elapsed = time.time() - start_time
    
    stats = {
        "requested": n,
        "generated": total_generated,
        "failed": total_failed,
        "elapsed_seconds": elapsed,
        "pairs_per_second": total_generated / elapsed if elapsed > 0 else 0,
        "output_dir": output_dir,
    }
    
    return stats


def count_existing(output_dir: str) -> int:
    """Count existing pair files in output directory."""
    if not os.path.exists(output_dir):
        return 0
    return len([f for f in os.listdir(output_dir) if f.endswith('.json')])


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch generate Python→IR pairs")
    parser.add_argument("-n", type=int, default=10000, help="Number of pairs to generate")
    parser.add_argument("-o", "--output", default="../../data", help="Output directory")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-b", "--batch-size", type=int, default=1000, help="Batch size")
    parser.add_argument("--resume", action="store_true", help="Resume from existing files")
    args = parser.parse_args()
    
    existing = count_existing(args.output)
    if args.resume and existing > 0:
        print(f"Found {existing} existing pairs, resuming...")
        start_idx = existing
        remaining = max(0, args.n - existing)
    else:
        start_idx = 0
        remaining = args.n
    
    if remaining > 0:
        stats = batch_generate(
            n=remaining,
            output_dir=args.output,
            seed=args.seed,
            batch_size=args.batch_size,
            start_idx=start_idx,
        )
        print(f"\nFinal stats: {stats}")
    else:
        print(f"Already have {existing} pairs, nothing to do.")
    
    # Final count
    final_count = count_existing(args.output)
    print(f"\nTotal pairs in {args.output}: {final_count}")

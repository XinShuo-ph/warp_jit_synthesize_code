"""Batch generation for large-scale dataset creation."""

import os
import json
import random
import hashlib
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import jax
import jax.numpy as jnp

from generator import GENERATORS


def generate_single(args):
    """Generate a single pair (for parallel execution)."""
    seed, output_dir = args
    random.seed(seed)
    
    try:
        # Pick generator and generate
        generator = random.choice(GENERATORS)
        fn, source, fn_args = generator()
        
        # Extract IR
        jaxpr = jax.make_jaxpr(fn)(*fn_args)
        lowered = jax.jit(fn).lower(*fn_args)
        hlo_text = lowered.as_text()
        
        # Get shapes
        input_shapes = []
        for arg in fn_args:
            if hasattr(arg, 'shape'):
                input_shapes.append(f"{arg.dtype}{list(arg.shape)}")
            else:
                input_shapes.append(str(type(arg).__name__))
        
        pair = {
            "python_source": source,
            "jaxpr": str(jaxpr),
            "hlo_text": hlo_text,
            "input_shapes": input_shapes,
            "seed": seed,
        }
        
        # Save to file
        content_hash = hashlib.md5(f"{source}{seed}".encode()).hexdigest()[:8]
        filename = f"pair_{seed:06d}_{content_hash}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(pair, f, indent=2)
        
        return True, seed
        
    except Exception as e:
        return False, seed


def batch_generate(n_samples: int, output_dir: str, n_workers: int = None) -> dict:
    """Generate n_samples pairs in batches using multiprocessing."""
    os.makedirs(output_dir, exist_ok=True)
    
    if n_workers is None:
        n_workers = min(4, multiprocessing.cpu_count())
    
    print(f"Generating {n_samples} pairs with {n_workers} workers...")
    
    success = 0
    failed = 0
    
    # Generate in batches (multiprocessing has overhead)
    batch_size = 100
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_samples)
        batch_args = [(seed, output_dir) for seed in range(start, end)]
        
        # Use ThreadPool for I/O bound, ProcessPool for CPU bound
        # JAX tracing is CPU bound but has GIL issues, so sequential is often faster
        for args in batch_args:
            ok, seed = generate_single(args)
            if ok:
                success += 1
            else:
                failed += 1
        
        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            print(f"  Batch {batch_idx + 1}/{n_batches}: {success} success, {failed} failed")
    
    stats = {
        "total_attempted": n_samples,
        "success": success,
        "failed": failed,
        "output_dir": output_dir,
    }
    
    # Save stats
    stats_path = os.path.join(output_dir, "_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


def count_pairs(output_dir: str) -> int:
    """Count number of pair files in directory."""
    return len([f for f in os.listdir(output_dir) if f.startswith("pair_") and f.endswith(".json")])


def validate_dataset(output_dir: str, sample_size: int = 100) -> dict:
    """Validate a sample of the dataset."""
    files = [f for f in os.listdir(output_dir) if f.startswith("pair_") and f.endswith(".json")]
    
    if len(files) > sample_size:
        files = random.sample(files, sample_size)
    
    stats = {
        "total_files": len(files),
        "valid": 0,
        "invalid": 0,
        "total_jaxpr_chars": 0,
        "total_hlo_chars": 0,
    }
    
    for filename in files:
        filepath = os.path.join(output_dir, filename)
        try:
            with open(filepath, 'r') as f:
                pair = json.load(f)
            
            if pair.get("jaxpr") and pair.get("hlo_text") and pair.get("python_source"):
                stats["valid"] += 1
                stats["total_jaxpr_chars"] += len(pair["jaxpr"])
                stats["total_hlo_chars"] += len(pair["hlo_text"])
            else:
                stats["invalid"] += 1
        except:
            stats["invalid"] += 1
    
    if stats["valid"] > 0:
        stats["avg_jaxpr_chars"] = stats["total_jaxpr_chars"] // stats["valid"]
        stats["avg_hlo_chars"] = stats["total_hlo_chars"] // stats["valid"]
    
    return stats


if __name__ == "__main__":
    import sys
    
    output_dir = "../../data/full"
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    
    print(f"Starting batch generation of {n_samples} pairs...")
    print(f"Output directory: {output_dir}")
    print()
    
    stats = batch_generate(n_samples, output_dir)
    
    print(f"\nGeneration complete!")
    print(f"  Success: {stats['success']}")
    print(f"  Failed: {stats['failed']}")
    
    print(f"\nValidating dataset...")
    val_stats = validate_dataset(output_dir)
    print(f"  Valid pairs: {val_stats['valid']}")
    print(f"  Avg jaxpr length: {val_stats.get('avg_jaxpr_chars', 0)} chars")
    print(f"  Avg HLO length: {val_stats.get('avg_hlo_chars', 0)} chars")

"""Batch generator for large-scale Python→IR data generation."""
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'extraction'))

import jax
import jax.numpy as jnp
from generator import FunctionGenerator
from ir_extractor import extract_ir


def generate_single_pair(gen: FunctionGenerator, ir_format: str = "stablehlo") -> Optional[dict]:
    """Generate a single pair (thread-safe)."""
    try:
        fn, source, meta = gen.generate_random()
        ir = extract_ir(fn, *meta['args'], format=ir_format)
        
        return {
            'python': source,
            'ir': ir,
            'ir_format': ir_format,
            'function_type': meta['type'],
            'arg_shapes': [
                {'shape': list(arg.shape), 'dtype': str(arg.dtype)}
                for arg in meta['args']
            ],
        }
    except Exception as e:
        return None


def save_batch(pairs: list, output_dir: Path, batch_id: int, start_idx: int) -> int:
    """Save a batch of pairs to disk."""
    saved = 0
    for i, pair in enumerate(pairs):
        if pair is not None:
            filepath = output_dir / f"pair_{start_idx + i:06d}.json"
            with open(filepath, 'w') as f:
                json.dump(pair, f)
            saved += 1
    return saved


def run_batch_generation(
    n_pairs: int = 10000,
    output_dir: str = "data/training",
    batch_size: int = 100,
    ir_format: str = "stablehlo",
    seed: int = 42,
    checkpoint_every: int = 1000,
) -> dict:
    """Run large-scale batch generation.
    
    Args:
        n_pairs: Total number of pairs to generate
        output_dir: Output directory
        batch_size: Pairs per batch
        ir_format: IR format to extract
        seed: Random seed
        checkpoint_every: Save checkpoint every N pairs
        
    Returns:
        Generation statistics
    """
    output_path = Path(__file__).parent.parent.parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check for existing checkpoint
    checkpoint_file = output_path / "checkpoint.json"
    start_from = 0
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
            start_from = checkpoint.get('next_index', 0)
            print(f"Resuming from checkpoint: {start_from}")
    
    generator = FunctionGenerator(seed=seed)
    # Fast-forward generator state
    generator.counter = start_from
    
    stats = {
        'total_attempted': 0,
        'total_generated': 0,
        'by_type': {},
        'failures': 0,
        'output_dir': str(output_path),
        'start_time': datetime.now().isoformat(),
    }
    
    print(f"Generating {n_pairs} Python→IR pairs (starting from {start_from})...")
    print(f"Output directory: {output_path}")
    print(f"Batch size: {batch_size}")
    print("-" * 60)
    
    start_time = time.time()
    
    current_idx = start_from
    while current_idx < n_pairs:
        # Generate a batch
        batch_pairs = []
        batch_end = min(current_idx + batch_size, n_pairs)
        
        for i in range(current_idx, batch_end):
            stats['total_attempted'] += 1
            pair = generate_single_pair(generator, ir_format)
            
            if pair is not None:
                batch_pairs.append(pair)
                stats['total_generated'] += 1
                fn_type = pair['function_type']
                stats['by_type'][fn_type] = stats['by_type'].get(fn_type, 0) + 1
            else:
                batch_pairs.append(None)
                stats['failures'] += 1
        
        # Save batch
        save_batch(batch_pairs, output_path, 0, current_idx)
        
        current_idx = batch_end
        
        # Progress report
        elapsed = time.time() - start_time
        rate = stats['total_generated'] / elapsed if elapsed > 0 else 0
        eta = (n_pairs - current_idx) / rate if rate > 0 else 0
        
        print(f"Progress: {current_idx}/{n_pairs} ({100*current_idx/n_pairs:.1f}%) "
              f"| Rate: {rate:.1f}/s | ETA: {eta:.0f}s")
        
        # Checkpoint
        if current_idx % checkpoint_every == 0:
            checkpoint = {
                'next_index': current_idx,
                'stats': stats,
                'timestamp': datetime.now().isoformat(),
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
    
    # Final statistics
    total_time = time.time() - start_time
    stats['total_time_seconds'] = total_time
    stats['pairs_per_second'] = stats['total_generated'] / total_time if total_time > 0 else 0
    stats['end_time'] = datetime.now().isoformat()
    
    # Save final summary
    summary_path = output_path / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Remove checkpoint on successful completion
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    
    print("-" * 60)
    print(f"Complete! Generated {stats['total_generated']} pairs in {total_time:.1f}s")
    print(f"Rate: {stats['pairs_per_second']:.1f} pairs/second")
    print(f"By type: {stats['by_type']}")
    
    return stats


def compute_statistics(output_dir: str = "data/training") -> dict:
    """Compute statistics for generated dataset."""
    output_path = Path(__file__).parent.parent.parent / output_dir
    
    pair_files = list(output_path.glob("pair_*.json"))
    
    if not pair_files:
        print("No pairs found!")
        return {}
    
    stats = {
        'total_pairs': len(pair_files),
        'by_type': {},
        'python_lengths': [],
        'ir_lengths': [],
    }
    
    for filepath in pair_files:
        with open(filepath) as f:
            pair = json.load(f)
            
        fn_type = pair.get('function_type', 'unknown')
        stats['by_type'][fn_type] = stats['by_type'].get(fn_type, 0) + 1
        stats['python_lengths'].append(len(pair.get('python', '')))
        stats['ir_lengths'].append(len(pair.get('ir', '')))
    
    # Compute averages
    stats['avg_python_length'] = sum(stats['python_lengths']) / len(stats['python_lengths'])
    stats['avg_ir_length'] = sum(stats['ir_lengths']) / len(stats['ir_lengths'])
    stats['min_ir_length'] = min(stats['ir_lengths'])
    stats['max_ir_length'] = max(stats['ir_lengths'])
    
    # Remove raw lists for cleaner output
    del stats['python_lengths']
    del stats['ir_lengths']
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch generate Python→IR pairs")
    parser.add_argument("-n", "--n-pairs", type=int, default=10000,
                       help="Number of pairs to generate")
    parser.add_argument("-o", "--output-dir", default="data/training",
                       help="Output directory")
    parser.add_argument("-b", "--batch-size", type=int, default=100,
                       help="Batch size")
    parser.add_argument("-s", "--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    stats = run_batch_generation(
        n_pairs=args.n_pairs,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    
    print("\nDataset Statistics:")
    final_stats = compute_statistics(args.output_dir)
    for k, v in final_stats.items():
        print(f"  {k}: {v}")

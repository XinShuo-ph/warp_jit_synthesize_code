"""Batch generator for large-scale Python→IR dataset generation.

Supports:
- Multiprocessing for parallel generation
- Progress tracking and resume
- Deduplication
- Configurable output formats
"""
import sys
import os
import json
import hashlib
import time
import multiprocessing as mp
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from generator import GENERATORS, generate_random_kernel


def _worker_init():
    """Initialize warp in worker process."""
    import warp as wp
    wp.init()


def _generate_single_pair(args: tuple) -> Optional[dict]:
    """Worker function to generate a single pair.
    
    Args:
        args: Tuple of (index, kernel_type, output_dir)
        
    Returns:
        Dictionary with pair data or None if failed
    """
    idx, kernel_type, output_dir = args
    
    try:
        # Import here to ensure fresh imports in each worker
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from pipeline import generate_pair
        
        pair = generate_pair(kernel_type)
        
        if pair and output_dir:
            # Save to file
            filename = f"{pair['kernel_type']}_{pair['hash']}.json"
            filepath = os.path.join(output_dir, filename)
            
            # Skip if already exists (deduplication)
            if not os.path.exists(filepath):
                with open(filepath, 'w') as f:
                    json.dump(pair, f, indent=2)
        
        return pair
    
    except Exception as e:
        return None


def generate_batch_parallel(
    count: int,
    output_dir: str,
    num_workers: int = 1,
    quiet: bool = False
) -> dict:
    """Generate a batch of pairs using parallel workers.
    
    Note: Due to warp's global state, parallel generation with multiprocessing
    can be tricky. This implementation uses sequential generation with batching
    for reliability, with optional multiprocessing for I/O operations.
    
    Args:
        count: Number of pairs to generate
        output_dir: Directory to save JSON files
        num_workers: Number of parallel workers (1 = sequential)
        quiet: Suppress progress output
        
    Returns:
        Dictionary with generation statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of kernel types for balanced generation
    kernel_types = list(GENERATORS.keys())
    
    # Generate work items
    work_items = [
        (i, kernel_types[i % len(kernel_types)], output_dir)
        for i in range(count)
    ]
    
    stats = {
        'total': count,
        'success': 0,
        'failed': 0,
        'start_time': time.time(),
        'type_counts': {k: 0 for k in kernel_types}
    }
    
    # Sequential generation (most reliable for warp)
    # Note: Warp has global state that makes multiprocessing tricky
    import warp as wp
    wp.init()
    
    from pipeline import generate_pair
    
    for i, (idx, kernel_type, out_dir) in enumerate(work_items):
        pair = generate_pair(kernel_type)
        
        if pair:
            stats['success'] += 1
            stats['type_counts'][kernel_type] = stats['type_counts'].get(kernel_type, 0) + 1
            
            # Save to file
            filename = f"{pair['kernel_type']}_{pair['hash']}.json"
            filepath = os.path.join(out_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(pair, f, indent=2)
        else:
            stats['failed'] += 1
        
        if not quiet and (i + 1) % 10 == 0:
            elapsed = time.time() - stats['start_time']
            rate = (i + 1) / elapsed
            eta = (count - i - 1) / rate if rate > 0 else 0
            print(f"[{i+1:5d}/{count}] Success: {stats['success']}, "
                  f"Failed: {stats['failed']}, "
                  f"Rate: {rate:.1f}/s, ETA: {eta:.0f}s")
    
    stats['end_time'] = time.time()
    stats['duration'] = stats['end_time'] - stats['start_time']
    stats['rate'] = stats['success'] / stats['duration'] if stats['duration'] > 0 else 0
    
    return stats


def get_existing_count(output_dir: str) -> int:
    """Count existing pairs in output directory."""
    if not os.path.exists(output_dir):
        return 0
    return len([f for f in os.listdir(output_dir) if f.endswith('.json')])


def resume_generation(target_count: int, output_dir: str, quiet: bool = False) -> dict:
    """Resume generation from existing state.
    
    Args:
        target_count: Target number of pairs
        output_dir: Directory with existing pairs
        quiet: Suppress output
        
    Returns:
        Generation statistics
    """
    existing = get_existing_count(output_dir)
    remaining = max(0, target_count - existing)
    
    if remaining == 0:
        print(f"Target already reached: {existing} pairs exist")
        return {'total': existing, 'success': existing, 'failed': 0, 'new': 0}
    
    if not quiet:
        print(f"Resuming: {existing} existing, {remaining} to generate")
    
    stats = generate_batch_parallel(remaining, output_dir, quiet=quiet)
    stats['existing'] = existing
    stats['new'] = stats['success']
    
    return stats


def compute_dataset_stats(data_dir: str) -> dict:
    """Compute statistics for the generated dataset.
    
    Args:
        data_dir: Directory containing JSON pairs
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'total_pairs': 0,
        'type_counts': {},
        'source_lengths': [],
        'ir_lengths': [],
        'unique_hashes': set()
    }
    
    for filename in os.listdir(data_dir):
        if not filename.endswith('.json'):
            continue
        
        filepath = os.path.join(data_dir, filename)
        with open(filepath) as f:
            pair = json.load(f)
        
        stats['total_pairs'] += 1
        ktype = pair.get('kernel_type', 'unknown')
        stats['type_counts'][ktype] = stats['type_counts'].get(ktype, 0) + 1
        stats['source_lengths'].append(pair.get('source_length', 0))
        stats['ir_lengths'].append(pair.get('ir_length', 0))
        stats['unique_hashes'].add(pair.get('hash', ''))
    
    # Compute summaries
    if stats['source_lengths']:
        stats['avg_source_length'] = sum(stats['source_lengths']) / len(stats['source_lengths'])
        stats['avg_ir_length'] = sum(stats['ir_lengths']) / len(stats['ir_lengths'])
        stats['min_ir_length'] = min(stats['ir_lengths'])
        stats['max_ir_length'] = max(stats['ir_lengths'])
    
    stats['unique_count'] = len(stats['unique_hashes'])
    del stats['source_lengths']
    del stats['ir_lengths']
    del stats['unique_hashes']
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100, help="Number of pairs to generate")
    parser.add_argument("--output", type=str, default="../../data/generated", help="Output directory")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers (1=sequential)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing state")
    parser.add_argument("--stats", action="store_true", help="Just compute stats for existing data")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()
    
    output_dir = os.path.join(os.path.dirname(__file__), args.output)
    
    print("="*60)
    print("Batch Generator for Python→IR Training Data")
    print("="*60)
    
    if args.stats:
        stats = compute_dataset_stats(output_dir)
        print(f"\nDataset Statistics:")
        print(f"  Total pairs: {stats['total_pairs']}")
        print(f"  Unique pairs: {stats['unique_count']}")
        print(f"  Type distribution:")
        for ktype, count in sorted(stats['type_counts'].items()):
            print(f"    {ktype}: {count}")
        if 'avg_source_length' in stats:
            print(f"  Avg source length: {stats['avg_source_length']:.0f} chars")
            print(f"  Avg IR length: {stats['avg_ir_length']:.0f} chars")
            print(f"  IR length range: [{stats['min_ir_length']}, {stats['max_ir_length']}]")
    elif args.resume:
        stats = resume_generation(args.count, output_dir, args.quiet)
        print(f"\nGeneration complete:")
        print(f"  Existing: {stats.get('existing', 0)}")
        print(f"  New: {stats.get('new', 0)}")
        print(f"  Total: {stats['success'] + stats.get('existing', 0)}")
    else:
        stats = generate_batch_parallel(args.count, output_dir, args.workers, args.quiet)
        print(f"\nGeneration complete:")
        print(f"  Success: {stats['success']}/{stats['total']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Duration: {stats['duration']:.1f}s")
        print(f"  Rate: {stats['rate']:.2f} pairs/s")
        print(f"  Type distribution:")
        for ktype, count in sorted(stats['type_counts'].items()):
            print(f"    {ktype}: {count}")

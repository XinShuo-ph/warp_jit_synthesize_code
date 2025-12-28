#!/usr/bin/env python3
"""
CUDA Production Pipeline

Production-ready script for generating large-scale CUDA training datasets.
Generates Python→CUDA IR pairs without requiring a GPU.

Key insight: CUDA code generation via warp's builder.codegen("cuda") is pure Python
and works WITHOUT a GPU. The GPU is only needed for execution, not code generation.

Usage:
    python cuda_producer.py --count 1000 --output ../data/production
    python cuda_producer.py --count 500 --categories arithmetic vector math
"""
import os
import sys
import json
import time
import hashlib
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent))

import warp as wp

from generator import KernelGenerator, KernelSpec
from pipeline import compile_kernel_from_source, extract_ir_from_kernel, KERNEL_CATEGORIES


@dataclass
class GeneratedPair:
    """A validated CUDA Python→IR pair."""
    id: str
    kernel_name: str
    category: str
    python_source: str
    cuda_forward: str
    cuda_backward: Optional[str]
    metadata: dict


@dataclass
class GenerationStats:
    """Statistics for a generation run."""
    total_attempted: int
    total_generated: int
    total_failed: int
    by_category: dict
    avg_forward_lines: float
    avg_backward_lines: float
    generation_time_seconds: float
    pairs_per_second: float


def count_lines(code: str) -> int:
    """Count non-empty lines in code."""
    if not code:
        return 0
    return len([l for l in code.split('\n') if l.strip()])


def generate_single_pair(
    category: str,
    seed: int,
    pair_id: str
) -> Optional[GeneratedPair]:
    """
    Generate a single CUDA pair.
    
    This function is designed to be called in a subprocess.
    """
    try:
        start_time = time.time()
        
        gen = KernelGenerator(seed=seed)
        spec = gen.generate(category)
        source = gen.to_python_source(spec)
        
        # Compile the kernel
        kernel = compile_kernel_from_source(source, spec.name)
        
        # Extract CUDA IR (forward and backward)
        ir = extract_ir_from_kernel(kernel, device="cuda", include_backward=True)
        
        if ir["forward_code"] is None:
            return None
        
        generation_time_ms = int((time.time() - start_time) * 1000)
        
        return GeneratedPair(
            id=pair_id,
            kernel_name=spec.name,
            category=category,
            python_source=source,
            cuda_forward=ir["forward_code"],
            cuda_backward=ir["backward_code"],
            metadata={
                "device": "cuda",
                "has_backward": ir["backward_code"] is not None,
                "forward_lines": count_lines(ir["forward_code"]),
                "backward_lines": count_lines(ir["backward_code"]) if ir["backward_code"] else 0,
                "generation_time_ms": generation_time_ms,
                "seed": seed,
            }
        )
    
    except Exception as e:
        return None


def validate_cuda_pair(pair: GeneratedPair) -> bool:
    """Validate that a CUDA pair has required patterns."""
    if not pair.cuda_forward:
        return False
    
    # Check for CUDA patterns
    required_patterns = ["blockDim", "threadIdx", "cuda_kernel_forward"]
    for pattern in required_patterns:
        if pattern not in pair.cuda_forward:
            return False
    
    return True


def generate_dataset(
    count: int,
    output_dir: Path,
    categories: Optional[list[str]] = None,
    base_seed: int = 42,
    workers: int = 1,
    resume: bool = True
) -> GenerationStats:
    """
    Generate a large-scale CUDA dataset.
    
    Args:
        count: Target number of pairs to generate
        output_dir: Directory to save JSON files
        categories: List of kernel categories to use (None = all)
        base_seed: Base random seed for reproducibility
        workers: Number of parallel workers (1 = sequential)
        resume: If True, skip existing files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if categories is None:
        categories = KERNEL_CATEGORIES
    
    # Check for existing files if resuming
    existing_ids = set()
    if resume:
        for f in output_dir.glob("cuda_*.json"):
            existing_ids.add(f.stem)
    
    print("=" * 60)
    print("CUDA Production Pipeline")
    print("=" * 60)
    print(f"Target count: {count}")
    print(f"Output directory: {output_dir}")
    print(f"Categories: {categories}")
    print(f"Workers: {workers}")
    if existing_ids:
        print(f"Resuming: {len(existing_ids)} existing files found")
    print()
    
    # Initialize warp
    wp.init()
    
    start_time = time.time()
    
    generated = 0
    failed = 0
    category_counts = {cat: 0 for cat in categories}
    total_forward_lines = 0
    total_backward_lines = 0
    
    # Generate pairs sequentially (warp doesn't parallelize well across processes)
    import random
    random.seed(base_seed)
    
    attempt = 0
    while generated < count:
        # Pick category (round-robin for balance)
        cat_idx = attempt % len(categories)
        category = categories[cat_idx]
        
        # Generate unique seed
        seed = base_seed + attempt * 7919  # Use prime for better distribution
        
        # Generate ID
        pair_id = f"cuda_{generated:06d}"
        
        # Skip if already exists
        if pair_id in existing_ids:
            attempt += 1
            continue
        
        # Generate pair
        pair = generate_single_pair(category, seed, pair_id)
        
        if pair and validate_cuda_pair(pair):
            # Save to file
            filepath = output_dir / f"{pair_id}.json"
            pair_dict = {
                "id": pair.id,
                "kernel_name": pair.kernel_name,
                "category": pair.category,
                "python_source": pair.python_source,
                "cuda_forward": pair.cuda_forward,
                "cuda_backward": pair.cuda_backward,
                "metadata": pair.metadata,
            }
            
            with open(filepath, 'w') as f:
                json.dump(pair_dict, f, indent=2)
            
            generated += 1
            category_counts[category] += 1
            total_forward_lines += pair.metadata["forward_lines"]
            total_backward_lines += pair.metadata["backward_lines"]
            
            if generated % 100 == 0:
                elapsed = time.time() - start_time
                rate = generated / elapsed
                print(f"  Generated {generated}/{count} pairs ({rate:.1f} pairs/sec)")
        else:
            failed += 1
        
        attempt += 1
        
        # Safety limit
        if attempt > count * 3:
            print(f"Warning: Too many failures, stopping at {generated} pairs")
            break
    
    elapsed = time.time() - start_time
    
    # Calculate statistics
    stats = GenerationStats(
        total_attempted=attempt,
        total_generated=generated,
        total_failed=failed,
        by_category=category_counts,
        avg_forward_lines=total_forward_lines / generated if generated else 0,
        avg_backward_lines=total_backward_lines / generated if generated else 0,
        generation_time_seconds=elapsed,
        pairs_per_second=generated / elapsed if elapsed > 0 else 0,
    )
    
    # Save statistics
    stats_file = output_dir / "stats.json"
    with open(stats_file, 'w') as f:
        json.dump(asdict(stats), f, indent=2)
    
    return stats


def print_stats(stats: GenerationStats):
    """Print generation statistics."""
    print()
    print("=" * 60)
    print("Generation Complete")
    print("=" * 60)
    print(f"Total attempted: {stats.total_attempted}")
    print(f"Total generated: {stats.total_generated}")
    print(f"Total failed: {stats.total_failed}")
    print(f"Success rate: {stats.total_generated / stats.total_attempted * 100:.1f}%")
    print()
    print(f"Generation time: {stats.generation_time_seconds:.1f} seconds")
    print(f"Rate: {stats.pairs_per_second:.2f} pairs/second")
    print()
    print(f"Average forward lines: {stats.avg_forward_lines:.1f}")
    print(f"Average backward lines: {stats.avg_backward_lines:.1f}")
    print()
    print("Category distribution:")
    for cat, count in sorted(stats.by_category.items()):
        print(f"  {cat}: {count}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate large-scale CUDA training dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 1000 CUDA pairs
  python cuda_producer.py --count 1000 --output ../data/production
  
  # Generate with specific categories
  python cuda_producer.py --count 500 --categories arithmetic vector math
  
  # Resume interrupted generation
  python cuda_producer.py --count 1000 --output ../data/production --resume
"""
    )
    
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=1000,
        help="Number of pairs to generate (default: 1000)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="../data/production",
        help="Output directory (default: ../data/production)"
    )
    parser.add_argument(
        "--categories", "-c",
        nargs="+",
        choices=KERNEL_CATEGORIES,
        help="Categories to generate (default: all)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Base random seed (default: 42)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing files"
    )
    
    args = parser.parse_args()
    
    # Resolve output path
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent / output_dir
    
    stats = generate_dataset(
        count=args.count,
        output_dir=output_dir,
        categories=args.categories,
        base_seed=args.seed,
        workers=args.workers,
        resume=args.resume
    )
    
    print_stats(stats)


if __name__ == "__main__":
    main()

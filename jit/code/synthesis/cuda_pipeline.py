#!/usr/bin/env python3
"""
CUDA IR Production Pipeline

Generates large-scale Python→CUDA IR paired data for LLM training.
No GPU required - CUDA code generation works on any machine.

Usage:
    python3 cuda_pipeline.py -n 10000 -o data/cuda_training
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Any

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from generator import generate_kernel, GENERATORS, KernelSpec
from pipeline import compile_kernel_from_source, extract_ir_from_kernel

import warp as wp


@dataclass
class ProductionStats:
    """Statistics for a production run."""
    total_attempted: int = 0
    total_generated: int = 0
    failed: int = 0
    category_counts: dict = None
    start_time: float = 0
    end_time: float = 0
    
    def __post_init__(self):
        if self.category_counts is None:
            self.category_counts = {}
    
    @property
    def success_rate(self) -> float:
        if self.total_attempted == 0:
            return 0.0
        return self.total_generated / self.total_attempted
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def rate(self) -> float:
        if self.duration == 0:
            return 0.0
        return self.total_generated / self.duration
    
    def to_dict(self) -> dict:
        return {
            "total_attempted": self.total_attempted,
            "total_generated": self.total_generated,
            "failed": self.failed,
            "success_rate": self.success_rate,
            "category_distribution": self.category_counts,
            "duration_seconds": self.duration,
            "pairs_per_second": self.rate,
        }


def synthesize_cuda_pair(spec: KernelSpec) -> dict[str, Any] | None:
    """
    Synthesize a Python→CUDA IR pair from a kernel specification.
    
    Returns None if compilation fails.
    """
    try:
        # Compile kernel
        kernel = compile_kernel_from_source(spec.source, spec.name)
        
        # Force module to be loaded
        _ = kernel.module
        
        # Extract CUDA IR
        ir = extract_ir_from_kernel(kernel, device="cuda")
        
        if ir["forward_code"] is None:
            return None
        
        # Validate it's actually CUDA code
        if "_cuda_kernel_forward" not in ir["forward_code"]:
            return None
        
        return {
            "python_source": spec.source,
            "cuda_forward": ir["forward_code"],
            "metadata": {
                "kernel_name": spec.name,
                "category": spec.category,
                "description": spec.description,
                "device": "cuda",
                **spec.metadata
            }
        }
    
    except Exception as e:
        return None


def validate_cuda_pair(pair: dict) -> bool:
    """Validate that a generated pair contains proper CUDA IR."""
    if not pair:
        return False
    
    cuda_code = pair.get("cuda_forward", "")
    if not cuda_code:
        return False
    
    # Check for CUDA-specific patterns
    required_patterns = [
        "_cuda_kernel_forward",
    ]
    
    cuda_patterns = [
        "blockDim",
        "blockIdx", 
        "threadIdx",
    ]
    
    # Must have the function name pattern
    if not all(p in cuda_code for p in required_patterns):
        return False
    
    # Should have at least one CUDA-specific pattern
    if not any(p in cuda_code for p in cuda_patterns):
        return False
    
    return True


def run_cuda_production(
    n: int,
    output_dir: Path,
    seed: int = 42,
    validate: bool = True,
    progress_interval: int = 100
) -> ProductionStats:
    """
    Run CUDA IR production pipeline.
    
    Args:
        n: Number of pairs to generate
        output_dir: Directory to save pairs
        seed: Random seed for reproducibility
        validate: Whether to validate each pair
        progress_interval: How often to print progress
    
    Returns:
        ProductionStats with generation statistics
    """
    import random
    random.seed(seed)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = ProductionStats()
    stats.start_time = time.time()
    stats.category_counts = {cat: 0 for cat in GENERATORS.keys()}
    
    print(f"Generating {n} CUDA Python→IR pairs...")
    print(f"Output: {output_dir}")
    print(f"Seed: {seed}")
    print()
    
    file_index = 0
    categories = list(GENERATORS.keys())
    
    while stats.total_generated < n:
        # Generate kernel spec
        cat = random.choice(categories)
        spec = generate_kernel(cat, seed=seed + stats.total_attempted)
        stats.total_attempted += 1
        
        # Synthesize CUDA pair
        pair = synthesize_cuda_pair(spec)
        
        if pair is None:
            stats.failed += 1
            continue
        
        # Validate if requested
        if validate and not validate_cuda_pair(pair):
            stats.failed += 1
            continue
        
        # Save pair
        filename = f"cuda_{file_index:06d}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(pair, f, indent=2)
        
        stats.total_generated += 1
        stats.category_counts[cat] += 1
        file_index += 1
        
        # Progress
        if stats.total_generated % progress_interval == 0:
            elapsed = time.time() - stats.start_time
            rate = stats.total_generated / elapsed if elapsed > 0 else 0
            print(f"  Progress: {stats.total_generated}/{n} ({rate:.1f} pairs/sec)")
    
    stats.end_time = time.time()
    
    # Save statistics
    stats_file = output_dir / "cuda_production_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats.to_dict(), f, indent=2)
    
    return stats


def print_summary(stats: ProductionStats):
    """Print production summary."""
    print()
    print("=" * 60)
    print("CUDA IR Production Complete")
    print("=" * 60)
    print(f"Total generated: {stats.total_generated}")
    print(f"Total attempted: {stats.total_attempted}")
    print(f"Failed: {stats.failed}")
    print(f"Success rate: {stats.success_rate:.1%}")
    print(f"Duration: {stats.duration:.1f}s")
    print(f"Rate: {stats.rate:.1f} pairs/sec")
    print()
    print("Category distribution:")
    for cat, count in sorted(stats.category_counts.items()):
        pct = count / stats.total_generated * 100 if stats.total_generated > 0 else 0
        print(f"  {cat}: {count} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate CUDA Python→IR training data (no GPU required)"
    )
    parser.add_argument("-n", type=int, default=10000,
                        help="Number of pairs to generate (default: 10000)")
    parser.add_argument("-o", "--output", default="/workspace/jit/data/cuda_training",
                        help="Output directory")
    parser.add_argument("-s", "--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip validation of generated pairs")
    parser.add_argument("--progress", type=int, default=100,
                        help="Progress interval (default: 100)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CUDA IR Production Pipeline")
    print("=" * 60)
    print("Note: CUDA IR generation works WITHOUT a GPU!")
    print("      Only kernel execution requires a GPU.")
    print()
    
    # Initialize warp
    wp.init()
    
    # Run production
    stats = run_cuda_production(
        n=args.n,
        output_dir=Path(args.output),
        seed=args.seed,
        validate=not args.no_validate,
        progress_interval=args.progress
    )
    
    print_summary(stats)
    
    return 0 if stats.total_generated == args.n else 1


if __name__ == "__main__":
    sys.exit(main())

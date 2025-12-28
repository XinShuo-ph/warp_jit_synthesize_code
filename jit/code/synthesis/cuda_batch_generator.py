#!/usr/bin/env python3
"""
CUDA IR Batch Generator

Produces CUDA IR (Python → CUDA code) pairs at scale WITHOUT requiring a GPU.
Warp's code generation system can produce CUDA IR on any machine.

Features:
- All 10 kernel types supported
- Forward and backward pass extraction
- Checkpointing for resumption after interruption
- Progress tracking
- Balanced category distribution

Usage:
    python3 cuda_batch_generator.py --count 1000 --output data/cuda_production --backward
"""
import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from typing import Any
from datetime import datetime

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))

import warp as wp
from generator import generate_kernel, GENERATORS, KernelSpec
from pipeline import compile_kernel_from_source, extract_ir_from_kernel


def load_checkpoint(checkpoint_file: Path) -> dict:
    """Load checkpoint state if exists."""
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {"generated": 0, "failed": 0, "last_seed": 0}


def save_checkpoint(checkpoint_file: Path, state: dict):
    """Save checkpoint state."""
    with open(checkpoint_file, 'w') as f:
        json.dump(state, f, indent=2)


def generate_cuda_pair(
    spec: KernelSpec,
    include_backward: bool = True
) -> dict[str, Any] | None:
    """
    Generate a single CUDA IR pair from kernel spec.
    
    Returns None if generation fails.
    """
    try:
        # Compile kernel from source
        kernel = compile_kernel_from_source(spec.source, spec.name)
        
        # Extract CUDA IR
        ir = extract_ir_from_kernel(kernel, device="cuda", include_backward=include_backward)
        
        if ir["forward_code"] is None:
            return None
        
        result = {
            "python_source": spec.source,
            "ir_forward": ir["forward_code"],
            "metadata": {
                "kernel_name": spec.name,
                "category": spec.category,
                "description": spec.description,
                "device": "cuda",
                "ir_type": "cuda",
                "generated_at": datetime.now().isoformat(),
                **spec.metadata
            }
        }
        
        if include_backward and ir["backward_code"]:
            result["ir_backward"] = ir["backward_code"]
            result["metadata"]["has_backward"] = True
        else:
            result["metadata"]["has_backward"] = False
        
        return result
    
    except Exception as e:
        return None


def generate_cuda_batch(
    count: int,
    output_dir: Path,
    include_backward: bool = True,
    use_checkpoint: bool = True,
    seed: int = 42,
    balanced: bool = True
) -> dict[str, int]:
    """
    Generate CUDA IR pairs in batches.
    
    Args:
        count: Number of pairs to generate
        output_dir: Output directory for JSON files
        include_backward: Include backward/adjoint kernels
        use_checkpoint: Enable checkpointing for resumption
        seed: Random seed for reproducibility
        balanced: Balance kernel types evenly
    
    Returns:
        Statistics dict with counts
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_file = output_dir / ".checkpoint.json"
    
    # Load checkpoint if resuming
    if use_checkpoint:
        state = load_checkpoint(checkpoint_file)
        start_idx = state["generated"]
        seed = state.get("last_seed", seed)
        if start_idx > 0:
            print(f"Resuming from checkpoint: {start_idx} pairs already generated")
    else:
        state = {"generated": 0, "failed": 0, "last_seed": seed}
        start_idx = 0
    
    random.seed(seed)
    
    # Get all kernel types
    kernel_types = list(GENERATORS.keys())
    
    # Statistics
    category_counts = {k: 0 for k in kernel_types}
    stats = {
        "generated": state["generated"],
        "failed": state["failed"],
        "start_time": time.time()
    }
    
    print(f"Generating {count} CUDA IR pairs...")
    print(f"Output: {output_dir}")
    print(f"Backward pass: {include_backward}")
    print(f"Balanced categories: {balanced}")
    print()
    
    # Initialize warp
    wp.init()
    
    for i in range(start_idx, count):
        # Select kernel type (balanced or random)
        if balanced:
            # Rotate through kernel types for even distribution
            kernel_type = kernel_types[i % len(kernel_types)]
        else:
            kernel_type = random.choice(kernel_types)
        
        # Generate kernel spec with unique seed
        pair_seed = seed + i
        spec = generate_kernel(kernel_type, seed=pair_seed)
        
        # Generate CUDA pair
        pair = generate_cuda_pair(spec, include_backward)
        
        if pair is not None:
            # Save to file
            filename = f"cuda_{i:06d}.json"
            filepath = output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(pair, f, indent=2)
            
            stats["generated"] += 1
            category_counts[kernel_type] += 1
        else:
            stats["failed"] += 1
        
        # Progress update
        if (i + 1) % 100 == 0 or i == count - 1:
            elapsed = time.time() - stats["start_time"]
            rate = (i + 1 - start_idx) / elapsed if elapsed > 0 else 0
            print(f"  Progress: {i + 1}/{count} ({rate:.1f} pairs/sec)")
            
            # Save checkpoint
            if use_checkpoint:
                state["generated"] = stats["generated"]
                state["failed"] = stats["failed"]
                state["last_seed"] = seed
                save_checkpoint(checkpoint_file, state)
    
    # Final checkpoint
    if use_checkpoint:
        state["generated"] = stats["generated"]
        state["failed"] = stats["failed"]
        state["completed"] = True
        save_checkpoint(checkpoint_file, state)
    
    # Print summary
    elapsed = time.time() - stats["start_time"]
    print()
    print("=" * 60)
    print("CUDA Batch Generation Complete")
    print("=" * 60)
    print(f"Generated: {stats['generated']} pairs")
    print(f"Failed: {stats['failed']} pairs")
    print(f"Time: {elapsed:.1f} seconds")
    print(f"Rate: {stats['generated'] / elapsed:.1f} pairs/sec")
    print()
    print("Category distribution:")
    for cat, cnt in sorted(category_counts.items()):
        if cnt > 0:
            print(f"  {cat}: {cnt}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="CUDA IR Batch Generator - Produces CUDA IR pairs without GPU"
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=1000,
        help="Number of CUDA IR pairs to generate"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for generated pairs"
    )
    parser.add_argument(
        "--backward", "-b",
        action="store_true",
        help="Include backward/adjoint kernels"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        action="store_true",
        help="Enable checkpointing for resumption"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no-balanced",
        action="store_true",
        help="Disable balanced category distribution"
    )
    
    args = parser.parse_args()
    
    generate_cuda_batch(
        count=args.count,
        output_dir=Path(args.output),
        include_backward=args.backward,
        use_checkpoint=args.checkpoint,
        seed=args.seed,
        balanced=not args.no_balanced
    )


if __name__ == "__main__":
    main()

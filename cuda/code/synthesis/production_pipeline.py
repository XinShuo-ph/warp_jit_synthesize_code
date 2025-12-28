"""
Production CUDA Pipeline

Comprehensive production system for generating large-scale CUDA kernel datasets.
Features:
- Category balancing
- Quality metrics
- Duplicate detection
- Multiple output formats
"""
import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Any
from collections import defaultdict

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))

import warp as wp
from generator import GENERATORS, generate_kernel
from advanced_generator import ADVANCED_GENERATORS, generate_advanced_kernel
from pipeline import compile_kernel_from_source, extract_ir_from_kernel


# Combine all generators
ALL_GENERATORS = {**GENERATORS, **ADVANCED_GENERATORS}


class ProductionPipeline:
    """Production pipeline for CUDA kernel generation."""
    
    def __init__(self, output_dir: str, device: str = "cuda", include_backward: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.include_backward = include_backward
        
        # Statistics
        self.stats = {
            "total_generated": 0,
            "successful": 0,
            "failed": 0,
            "category_counts": defaultdict(int),
            "duplicates_skipped": 0,
        }
        
        # Duplicate detection
        self.seen_hashes = set()
        self.load_existing_hashes()
        
        wp.init()
    
    def load_existing_hashes(self):
        """Load hashes of existing kernels to avoid duplicates."""
        for json_file in self.output_dir.glob("*.json"):
            if json_file.name == "stats.json":
                continue
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    source_hash = self.hash_source(data.get("python_source", ""))
                    self.seen_hashes.add(source_hash)
            except:
                pass
    
    def hash_source(self, source: str) -> str:
        """Generate hash of Python source code."""
        return hashlib.md5(source.encode()).hexdigest()
    
    def is_duplicate(self, source: str) -> bool:
        """Check if kernel source is a duplicate."""
        source_hash = self.hash_source(source)
        if source_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(source_hash)
        return False
    
    def generate_single(self, category: str | None = None, seed: int | None = None) -> dict[str, Any] | None:
        """Generate a single kernel pair."""
        self.stats["total_generated"] += 1
        
        # Choose category if not specified
        if category is None:
            category = self.select_balanced_category()
        
        # Generate kernel spec
        if category in ADVANCED_GENERATORS:
            spec = generate_advanced_kernel(category, seed)
        else:
            spec = generate_kernel(category, seed)
        
        # Check for duplicates
        if self.is_duplicate(spec.source):
            self.stats["duplicates_skipped"] += 1
            return None
        
        try:
            # Compile and extract IR
            kernel = compile_kernel_from_source(spec.source, spec.name)
            _ = kernel.module  # Force compilation
            
            ir = extract_ir_from_kernel(kernel, self.device, self.include_backward)
            
            if ir["forward_code"] is None:
                self.stats["failed"] += 1
                return None
            
            # Build result
            result = {
                "python_source": spec.source,
                "ir_forward": ir["forward_code"],
                "metadata": {
                    "kernel_name": spec.name,
                    "category": spec.category,
                    "description": spec.description,
                    "device": self.device,
                    "has_backward": ir["backward_code"] is not None,
                    **spec.metadata
                }
            }
            
            if ir["backward_code"] is not None:
                result["ir_backward"] = ir["backward_code"]
            
            self.stats["successful"] += 1
            self.stats["category_counts"][spec.category] += 1
            
            return result
            
        except Exception as e:
            self.stats["failed"] += 1
            print(f"  Failed to generate {spec.name}: {e}")
            return None
    
    def select_balanced_category(self) -> str:
        """Select category to maintain balance."""
        # Get current counts
        counts = self.stats["category_counts"]
        
        # If no kernels yet, random choice
        if not counts:
            return random.choice(list(ALL_GENERATORS.keys()))
        
        # Find categories with minimum count
        min_count = min(counts.values()) if counts else 0
        underrepresented = [cat for cat in ALL_GENERATORS.keys() 
                          if counts.get(cat, 0) <= min_count]
        
        return random.choice(underrepresented)
    
    def generate_batch(self, n: int, start_index: int = 0, seed: int = 42) -> list[dict[str, Any]]:
        """Generate a batch of n kernels."""
        import random
        random.seed(seed)
        
        results = []
        attempts = 0
        max_attempts = n * 3  # Allow some failures
        
        print(f"Generating batch of {n} kernels...")
        print(f"Device: {self.device}, Backward: {self.include_backward}")
        
        while len(results) < n and attempts < max_attempts:
            attempts += 1
            
            if (attempts % 10) == 0:
                print(f"  Progress: {len(results)}/{n} (attempts: {attempts})")
            
            pair = self.generate_single(seed=seed + attempts)
            if pair is not None:
                results.append(pair)
        
        return results
    
    def save_batch(self, pairs: list[dict[str, Any]], prefix: str = "kernel", start_index: int = 0):
        """Save batch to disk."""
        for i, pair in enumerate(pairs):
            filename = f"{prefix}_{start_index + i:06d}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(pair, f, indent=2)
        
        print(f"Saved {len(pairs)} kernels to {self.output_dir}")
    
    def save_stats(self):
        """Save generation statistics."""
        stats_file = self.output_dir / "stats.json"
        
        with open(stats_file, 'w') as f:
            json.dump({
                "total_generated": self.stats["total_generated"],
                "successful": self.stats["successful"],
                "failed": self.stats["failed"],
                "duplicates_skipped": self.stats["duplicates_skipped"],
                "category_distribution": dict(self.stats["category_counts"]),
                "device": self.device,
                "include_backward": self.include_backward,
            }, f, indent=2)
        
        print(f"\nStatistics saved to {stats_file}")
    
    def print_summary(self):
        """Print generation summary."""
        print("\n" + "=" * 60)
        print("Generation Summary")
        print("=" * 60)
        print(f"Total attempts:      {self.stats['total_generated']}")
        print(f"Successful:          {self.stats['successful']}")
        print(f"Failed:              {self.stats['failed']}")
        print(f"Duplicates skipped:  {self.stats['duplicates_skipped']}")
        print(f"\nCategory distribution:")
        for cat, count in sorted(self.stats['category_counts'].items()):
            print(f"  {cat:20s}: {count:4d}")


def run_production(
    n: int = 500,
    output_dir: str = "/workspace/cuda/data/production",
    device: str = "cuda",
    include_backward: bool = True,
    seed: int = 42
):
    """Run production generation."""
    print("=" * 60)
    print("CUDA Kernel Production Pipeline")
    print("=" * 60)
    print(f"Target: {n} kernels")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"Backward: {include_backward}")
    print(f"Seed: {seed}")
    print()
    
    pipeline = ProductionPipeline(output_dir, device, include_backward)
    
    # Check for existing kernels
    existing = len(list(Path(output_dir).glob("kernel_*.json")))
    if existing > 0:
        print(f"Found {existing} existing kernels, will add to them")
        start_index = existing
    else:
        start_index = 0
    
    # Generate batch
    import time
    start_time = time.time()
    
    pairs = pipeline.generate_batch(n, start_index=start_index, seed=seed)
    
    elapsed = time.time() - start_time
    
    # Save results
    pipeline.save_batch(pairs, prefix="kernel", start_index=start_index)
    pipeline.save_stats()
    pipeline.print_summary()
    
    print(f"\nGeneration time: {elapsed:.1f}s")
    print(f"Rate: {len(pairs) / elapsed:.1f} kernels/sec")
    
    return pipeline.stats


if __name__ == "__main__":
    import argparse
    import random
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=500, help="Number of kernels to generate")
    parser.add_argument("-o", "--output", default="/workspace/cuda/data/production", help="Output directory")
    parser.add_argument("-d", "--device", choices=["cpu", "cuda"], default="cuda", help="Target device")
    parser.add_argument("-b", "--backward", action="store_true", default=True, help="Include backward pass")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    run_production(args.n, args.output, args.device, args.backward, args.seed)

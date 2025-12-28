"""
Production CUDA IR Dataset Generator

Generate large-scale, high-quality CUDA IR training data without GPU hardware.
This script produces production-ready datasets for LLM training.
"""
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter

# Add paths
parent = Path(__file__).parent.parent
sys.path.insert(0, str(parent / "code" / "extraction"))
sys.path.insert(0, str(parent / "code" / "synthesis"))

import warp as wp
from generator import GENERATORS, generate_kernel
from pipeline import compile_kernel_from_source, extract_ir_from_kernel


class ProductionDatasetGenerator:
    """Generate production-quality CUDA IR dataset."""
    
    def __init__(self, output_dir: str, device: str = "cuda"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.stats = {
            "total_generated": 0,
            "successful": 0,
            "failed": 0,
            "by_category": Counter(),
            "cuda_patterns_verified": 0,
            "start_time": None,
            "end_time": None,
        }
        
    def verify_cuda_patterns(self, ir_code: str) -> bool:
        """Verify that IR code contains expected CUDA patterns."""
        required_patterns = [
            "blockIdx",
            "threadIdx",
            "blockDim",
            "gridDim"
        ]
        return any(pattern in ir_code for pattern in required_patterns)
    
    def generate_single_pair(self, category: str, seed: int) -> Dict[str, Any] | None:
        """Generate a single Python→CUDA IR pair."""
        try:
            # Generate kernel specification
            spec = generate_kernel(category, seed=seed)
            
            # Compile kernel
            kernel = compile_kernel_from_source(spec.source, spec.name)
            
            # Extract CUDA IR
            ir = extract_ir_from_kernel(kernel, device=self.device)
            
            if ir["forward_code"] is None:
                return None
            
            # Verify CUDA patterns
            has_cuda = self.verify_cuda_patterns(ir["forward_code"])
            
            pair = {
                "python_source": spec.source,
                "cuda_ir": ir["forward_code"],
                "metadata": {
                    "kernel_name": spec.name,
                    "category": spec.category,
                    "description": spec.description,
                    "device": self.device,
                    "seed": seed,
                    "cuda_patterns_verified": has_cuda,
                    **spec.metadata
                }
            }
            
            if has_cuda:
                self.stats["cuda_patterns_verified"] += 1
            
            return pair
            
        except Exception as e:
            print(f"  ✗ Failed to generate {category} kernel (seed {seed}): {e}")
            return None
    
    def generate_dataset(
        self, 
        total_pairs: int,
        categories: List[str] | None = None,
        start_seed: int = 1000
    ) -> Dict[str, Any]:
        """Generate full dataset with balanced categories."""
        
        print("=" * 70)
        print("Production CUDA IR Dataset Generation")
        print("=" * 70)
        print(f"Target: {total_pairs} pairs")
        print(f"Device: {self.device}")
        print(f"Output: {self.output_dir}")
        print(f"Categories: {categories or 'all'}")
        print()
        
        wp.init()
        
        if categories is None:
            categories = list(GENERATORS.keys())
        
        self.stats["start_time"] = time.time()
        
        # Calculate pairs per category for balanced distribution
        pairs_per_category = total_pairs // len(categories)
        remainder = total_pairs % len(categories)
        
        print(f"Generating {pairs_per_category} pairs per category")
        print(f"(+{remainder} extra pairs for balanced distribution)")
        print()
        
        generated_files = []
        seed = start_seed
        
        for cat_idx, category in enumerate(categories):
            target = pairs_per_category + (1 if cat_idx < remainder else 0)
            print(f"Category: {category} (target: {target} pairs)")
            
            cat_successful = 0
            attempts = 0
            max_attempts = target * 3  # Allow retries
            
            while cat_successful < target and attempts < max_attempts:
                pair = self.generate_single_pair(category, seed)
                seed += 1
                attempts += 1
                
                if pair:
                    # Save to file
                    filename = f"{category}_{cat_successful:04d}_seed{seed-1}.json"
                    filepath = self.output_dir / filename
                    
                    with open(filepath, 'w') as f:
                        json.dump(pair, f, indent=2)
                    
                    generated_files.append(str(filepath))
                    cat_successful += 1
                    self.stats["successful"] += 1
                    self.stats["by_category"][category] += 1
                    
                    if cat_successful % 50 == 0:
                        print(f"  Progress: {cat_successful}/{target}")
                else:
                    self.stats["failed"] += 1
            
            print(f"  ✓ Completed: {cat_successful}/{target} pairs")
            print()
        
        self.stats["end_time"] = time.time()
        self.stats["total_generated"] = len(generated_files)
        self.stats["generation_time"] = self.stats["end_time"] - self.stats["start_time"]
        self.stats["pairs_per_second"] = self.stats["total_generated"] / self.stats["generation_time"]
        
        # Save generation stats
        stats_file = self.output_dir / "generation_stats.json"
        with open(stats_file, 'w') as f:
            # Convert Counter to dict for JSON serialization
            stats_copy = self.stats.copy()
            stats_copy["by_category"] = dict(stats_copy["by_category"])
            json.dump(stats_copy, f, indent=2)
        
        # Save file list
        manifest_file = self.output_dir / "manifest.txt"
        with open(manifest_file, 'w') as f:
            for filepath in generated_files:
                f.write(f"{Path(filepath).name}\n")
        
        return self.stats
    
    def print_summary(self):
        """Print generation summary."""
        print("=" * 70)
        print("Generation Complete")
        print("=" * 70)
        print(f"Total generated: {self.stats['total_generated']}")
        print(f"Successful: {self.stats['successful']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"CUDA patterns verified: {self.stats['cuda_patterns_verified']}/{self.stats['successful']}")
        print(f"Time: {self.stats['generation_time']:.1f}s")
        print(f"Rate: {self.stats['pairs_per_second']:.1f} pairs/sec")
        print()
        print("Category distribution:")
        for category, count in sorted(self.stats["by_category"].items()):
            pct = count / self.stats["total_generated"] * 100 if self.stats["total_generated"] > 0 else 0
            print(f"  {category:15s}: {count:4d} ({pct:5.1f}%)")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate production CUDA IR dataset for LLM training"
    )
    parser.add_argument(
        "-n", "--count",
        type=int,
        default=1000,
        help="Number of pairs to generate (default: 1000)"
    )
    parser.add_argument(
        "-o", "--output",
        default="/workspace/cuda/data/cuda_production",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "-d", "--device",
        default="cuda",
        choices=["cpu", "cuda"],
        help="Target device (default: cuda)"
    )
    parser.add_argument(
        "-c", "--categories",
        nargs="+",
        choices=list(GENERATORS.keys()),
        help="Specific categories to generate (default: all)"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=1000,
        help="Starting seed value (default: 1000)"
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = ProductionDatasetGenerator(args.output, device=args.device)
    
    # Generate dataset
    stats = generator.generate_dataset(
        total_pairs=args.count,
        categories=args.categories,
        start_seed=args.seed
    )
    
    # Print summary
    generator.print_summary()
    
    # Success check
    success_rate = stats["successful"] / stats["total_generated"] if stats["total_generated"] > 0 else 0
    cuda_verification_rate = stats["cuda_patterns_verified"] / stats["successful"] if stats["successful"] > 0 else 0
    
    print()
    print("Quality Metrics:")
    print(f"  Success rate: {success_rate * 100:.1f}%")
    print(f"  CUDA verification rate: {cuda_verification_rate * 100:.1f}%")
    print()
    
    if success_rate < 0.95:
        print("⚠ Warning: Success rate below 95%")
        return 1
    
    if cuda_verification_rate < 0.99:
        print("⚠ Warning: CUDA pattern verification below 99%")
        return 1
    
    print("✓ Dataset generation successful!")
    print(f"✓ Output: {args.output}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

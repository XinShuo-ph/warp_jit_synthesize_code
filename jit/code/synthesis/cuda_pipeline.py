"""
CUDA Synthesis Pipeline: Generate Python→CUDA IR pairs without GPU hardware.

This pipeline uses warp's internal codegen APIs to generate CUDA code without
requiring actual GPU hardware. The key insight is that after loading a module
for CPU, the ModuleBuilder can generate CUDA code via codegen('cuda').

Usage:
    python cuda_pipeline.py --count 100 --output ../../data/cuda
"""
import os
import sys
import json
import hashlib
import tempfile
import importlib.util
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional, Any

# Add parent dirs to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))
sys.path.insert(0, str(Path(__file__).parent))

from generator import KernelGenerator, KernelSpec


@dataclass
class CUDASynthesisPair:
    """A Python→CUDA IR training data pair."""
    id: str
    kernel_name: str
    kernel_type: str
    python_source: str
    cuda_ir_forward: str
    cuda_ir_backward: str
    device: str
    generated_at: str
    metadata: dict


def hash_source(source: str) -> str:
    """Generate a short hash of the source code."""
    return hashlib.sha256(source.encode()).hexdigest()[:12]


def compile_and_generate_cuda(python_source: str, kernel_name: str) -> tuple:
    """
    Compile a kernel and generate CUDA IR without GPU.
    
    Args:
        python_source: Python kernel source code
        kernel_name: Name of the kernel function
    
    Returns:
        (kernel, cuda_ir_full, cuda_ir_forward, cuda_ir_backward)
    """
    import warp as wp
    from warp._src import context
    import re
    
    # Create a temporary module file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        module_code = f'''"""Temporary kernel module."""
import warp as wp

{python_source}
'''
        f.write(module_code)
        temp_path = f.name
    
    try:
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location("temp_kernel", temp_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["temp_kernel"] = module
        spec.loader.exec_module(module)
        
        # Get the kernel
        kernel = getattr(module, kernel_name)
        
        # Load for CPU first (populates internal structures)
        kernel.module.load('cpu')
        
        # Generate CUDA code using ModuleBuilder
        options = {
            'mode': 'release',
            'block_dim': 256,
            'enable_backward': True
        }
        hasher = context.ModuleHasher(kernel.module)
        builder = context.ModuleBuilder(kernel.module, options, hasher)
        builder.build_kernel(kernel)
        
        cuda_ir_full = builder.codegen('cuda')
        
        # Extract forward and backward functions
        kernel_key = kernel.key
        
        def extract_function(pattern: str, code: str) -> str:
            match = re.search(pattern, code)
            if not match:
                return ""
            start = match.start()
            brace_count = 0
            end = start
            for i, c in enumerate(code[start:]):
                if c == '{':
                    brace_count += 1
                elif c == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = start + i + 1
                        break
            return code[start:end]
        
        forward_pattern = rf'(?:extern\s+"C"\s+)?__global__\s+void\s+{re.escape(kernel_key)}_[a-f0-9]+_cuda_kernel_forward\s*\([^)]*\)\s*\{{'
        backward_pattern = rf'(?:extern\s+"C"\s+)?__global__\s+void\s+{re.escape(kernel_key)}_[a-f0-9]+_cuda_kernel_backward\s*\([^)]*\)\s*\{{'
        
        forward_ir = extract_function(forward_pattern, cuda_ir_full)
        backward_ir = extract_function(backward_pattern, cuda_ir_full)
        
        return kernel, cuda_ir_full, forward_ir, backward_ir
        
    finally:
        # Clean up temp file
        os.unlink(temp_path)
        if "temp_kernel" in sys.modules:
            del sys.modules["temp_kernel"]


class CUDASynthesisPipeline:
    """Pipeline for generating Python→CUDA IR training pairs without GPU."""
    
    def __init__(self, output_dir: str, seed: Optional[int] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generator = KernelGenerator(seed=seed)
        self.stats = {
            "total_generated": 0,
            "successful": 0,
            "failed": 0,
            "by_type": {},
            "device": "cuda"
        }
    
    def generate_pair(self, kernel_type: Optional[str] = None) -> Optional[CUDASynthesisPair]:
        """Generate a single Python→CUDA IR pair."""
        # Generate kernel spec
        spec = self.generator.generate(kernel_type)
        python_source = self.generator.to_python_source(spec)
        
        # Track type stats
        ktype = kernel_type or "random"
        self.stats["by_type"][ktype] = self.stats["by_type"].get(ktype, 0) + 1
        self.stats["total_generated"] += 1
        
        try:
            # Compile and generate CUDA IR
            kernel, cuda_full, forward_ir, backward_ir = compile_and_generate_cuda(
                python_source, spec.name
            )
            
            if not forward_ir:
                raise ValueError("Failed to extract forward CUDA IR")
            
            # Create the pair
            pair_id = hash_source(python_source)
            pair = CUDASynthesisPair(
                id=pair_id,
                kernel_name=spec.name,
                kernel_type=ktype,
                python_source=python_source,
                cuda_ir_forward=forward_ir,
                cuda_ir_backward=backward_ir,
                device="cuda",
                generated_at=datetime.now().isoformat(),
                metadata={
                    "num_params": len(spec.params),
                    "num_lines": len(spec.body_lines),
                    "cuda_ir_length": len(cuda_full),
                    "forward_ir_length": len(forward_ir),
                    "backward_ir_length": len(backward_ir),
                }
            )
            
            self.stats["successful"] += 1
            return pair
            
        except Exception as e:
            self.stats["failed"] += 1
            print(f"Failed to generate CUDA pair for {spec.name}: {e}")
            return None
    
    def save_pair(self, pair: CUDASynthesisPair) -> Path:
        """Save a pair to disk as JSON."""
        filename = f"{pair.id}_{pair.kernel_name}.json"
        filepath = self.output_dir / filename
        
        data = {
            "id": pair.id,
            "kernel_name": pair.kernel_name,
            "kernel_type": pair.kernel_type,
            "python_source": pair.python_source,
            "cuda_ir_forward": pair.cuda_ir_forward,
            "cuda_ir_backward": pair.cuda_ir_backward,
            "device": pair.device,
            "generated_at": pair.generated_at,
            "metadata": pair.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def generate_batch(self, count: int, kernel_types: Optional[List[str]] = None) -> List[CUDASynthesisPair]:
        """Generate a batch of CUDA pairs."""
        if kernel_types is None:
            kernel_types = [
                "arithmetic", "conditional", "loop", "math", 
                "vector", "atomic", "nested", "multi_cond", 
                "combined", "scalar_param"
            ]
        
        pairs = []
        for i in range(count):
            ktype = kernel_types[i % len(kernel_types)]
            pair = self.generate_pair(ktype)
            if pair:
                self.save_pair(pair)
                pairs.append(pair)
            
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{count} ({self.stats['successful']} successful)")
        
        return pairs
    
    def print_stats(self):
        """Print generation statistics."""
        print("\n=== CUDA Synthesis Statistics ===")
        print(f"Device: {self.stats['device']} (generated without GPU)")
        print(f"Total attempted: {self.stats['total_generated']}")
        print(f"Successful: {self.stats['successful']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Success rate: {self.stats['successful'] / max(1, self.stats['total_generated']) * 100:.1f}%")
        print("\nBy kernel type:")
        for ktype, count in sorted(self.stats["by_type"].items()):
            print(f"  {ktype}: {count}")


def run_cuda_pipeline(output_dir: str, count: int = 100, seed: int = 42):
    """Run the CUDA synthesis pipeline."""
    import warp as wp
    wp.init()
    
    print("=" * 60)
    print("  CUDA Synthesis Pipeline (No GPU Required)")
    print("=" * 60)
    print(f"Target: {count} CUDA pairs")
    print(f"Output: {output_dir}")
    print(f"Seed: {seed}")
    print()
    
    pipeline = CUDASynthesisPipeline(output_dir, seed=seed)
    pairs = pipeline.generate_batch(count)
    
    pipeline.print_stats()
    
    # Save stats
    stats_file = Path(output_dir) / "generation_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(pipeline.stats, f, indent=2)
    
    print(f"\nGenerated {len(pairs)} valid CUDA pairs in {output_dir}")
    print(f"Stats saved to: {stats_file}")
    
    return pairs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Python→CUDA IR pairs (no GPU required)")
    parser.add_argument("--output", "-o", type=str, default="../../data/cuda", 
                        help="Output directory")
    parser.add_argument("--count", "-n", type=int, default=100, 
                        help="Number of pairs to generate")
    parser.add_argument("--seed", "-s", type=int, default=42, 
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Resolve output path relative to script
    output_path = Path(__file__).parent / args.output
    output_path = output_path.resolve()
    
    run_cuda_pipeline(str(output_path), args.count, args.seed)

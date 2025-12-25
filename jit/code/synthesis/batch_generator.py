"""Batch generator for large-scale Python→IR pair synthesis."""
import os
import sys
import json
import hashlib
import tempfile
import importlib.util
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional
import time

sys.path.insert(0, str(Path(__file__).parent))
from generator import KernelGenerator, KernelSpec


@dataclass
class BatchConfig:
    """Configuration for batch generation."""
    output_dir: str
    kernels_per_module: int = 20  # Batch size - kernels per module compile
    total_count: int = 10000
    seed: int = 42
    kernel_types: Optional[List[str]] = None


def generate_module_source(specs: List[KernelSpec], generator: KernelGenerator) -> str:
    """Generate Python source for a module with multiple kernels."""
    header = '''"""Auto-generated warp kernels module."""
import warp as wp

'''
    kernel_sources = []
    for spec in specs:
        kernel_sources.append(generator.to_python_source(spec))
    
    return header + "\n\n".join(kernel_sources)


def hash_source(source: str) -> str:
    """Generate a short hash of source code."""
    return hashlib.sha256(source.encode()).hexdigest()[:12]


def extract_kernel_ir_from_cpp(cpp_full: str, kernel_name: str) -> tuple:
    """Extract forward and backward IR for a kernel from module C++."""
    import re
    
    def extract_function(pattern_start: str, code: str) -> str:
        match = re.search(pattern_start, code)
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
    
    forward_pattern = rf'void\s+{re.escape(kernel_name)}_[a-f0-9]+_cpu_kernel_forward\('
    backward_pattern = rf'void\s+{re.escape(kernel_name)}_[a-f0-9]+_cpu_kernel_backward\('
    
    return extract_function(forward_pattern, cpp_full), extract_function(backward_pattern, cpp_full)


class BatchGenerator:
    """Generate Python→IR pairs in batches for efficiency."""
    
    def __init__(self, config: BatchConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generator = KernelGenerator(seed=config.seed)
        
        if config.kernel_types is None:
            self.kernel_types = [
                "arithmetic", "conditional", "loop", "math", 
                "vector", "atomic", "nested", "multi_cond", 
                "combined", "scalar_param"
            ]
        else:
            self.kernel_types = config.kernel_types
        
        self.stats = {
            "total_generated": 0,
            "successful": 0,
            "failed": 0,
            "modules_compiled": 0,
            "by_type": {},
            "start_time": None,
            "end_time": None
        }
    
    def generate_batch(self, specs: List[KernelSpec]) -> List[dict]:
        """Generate a batch of kernels in a single module."""
        import warp as wp
        
        module_source = generate_module_source(specs, self.generator)
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(module_source)
            temp_path = f.name
        
        try:
            # Load and compile module
            spec = importlib.util.spec_from_file_location("batch_kernel", temp_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["batch_kernel"] = module
            spec.loader.exec_module(module)
            
            results = []
            
            for kernel_spec in specs:
                try:
                    kernel = getattr(module, kernel_spec.name)
                    wp_module = kernel.module
                    wp_module.load("cpu")
                    
                    # Get generated C++
                    module_id = wp_module.get_module_identifier()
                    cache_dir = Path(os.path.expanduser(f"~/.cache/warp/{wp.__version__}"))
                    cpp_file = cache_dir / module_id / f"{module_id}.cpp"
                    
                    if not cpp_file.exists():
                        raise FileNotFoundError(f"C++ not found: {cpp_file}")
                    
                    cpp_full = cpp_file.read_text()
                    forward_ir, backward_ir = extract_kernel_ir_from_cpp(cpp_full, kernel_spec.name)
                    
                    if not forward_ir:
                        raise ValueError("No forward IR found")
                    
                    python_source = self.generator.to_python_source(kernel_spec)
                    pair_id = hash_source(python_source)
                    
                    results.append({
                        "id": pair_id,
                        "kernel_name": kernel_spec.name,
                        "kernel_type": getattr(kernel_spec, 'type', 'unknown'),
                        "python_source": python_source,
                        "cpp_ir_forward": forward_ir,
                        "cpp_ir_backward": backward_ir,
                        "generated_at": datetime.now().isoformat(),
                        "metadata": {
                            "num_params": len(kernel_spec.params),
                            "num_lines": len(kernel_spec.body_lines),
                            "module_id": module_id
                        }
                    })
                    
                    self.stats["successful"] += 1
                    ktype = getattr(kernel_spec, 'type', 'unknown')
                    self.stats["by_type"][ktype] = self.stats["by_type"].get(ktype, 0) + 1
                    
                except Exception as e:
                    self.stats["failed"] += 1
                    # Continue with other kernels in batch
            
            self.stats["modules_compiled"] += 1
            return results
            
        finally:
            os.unlink(temp_path)
    
    def save_pairs(self, pairs: List[dict]):
        """Save pairs to disk."""
        for pair in pairs:
            filename = f"{pair['id']}_{pair['kernel_name']}.json"
            filepath = self.output_dir / filename
            with open(filepath, 'w') as f:
                json.dump(pair, f, indent=2)
    
    def run(self):
        """Run batch generation."""
        import warp as wp
        wp.init()
        
        self.stats["start_time"] = time.time()
        
        print(f"Starting batch generation (target: {self.config.total_count} pairs)")
        print(f"Batch size: {self.config.kernels_per_module} kernels/module")
        print(f"Output: {self.config.output_dir}")
        
        generated = 0
        batch_num = 0
        
        while generated < self.config.total_count:
            # Generate specs for this batch
            batch_specs = []
            for i in range(self.config.kernels_per_module):
                if generated + len(batch_specs) >= self.config.total_count:
                    break
                
                ktype = self.kernel_types[(generated + i) % len(self.kernel_types)]
                spec = self.generator.generate(ktype)
                spec.type = ktype  # Store type on spec
                batch_specs.append(spec)
            
            self.stats["total_generated"] += len(batch_specs)
            
            # Generate and save batch
            pairs = self.generate_batch(batch_specs)
            self.save_pairs(pairs)
            
            generated += len(pairs)
            batch_num += 1
            
            if batch_num % 10 == 0:
                elapsed = time.time() - self.stats["start_time"]
                rate = generated / elapsed * 3600
                print(f"Progress: {generated}/{self.config.total_count} ({rate:.0f}/hr)")
        
        self.stats["end_time"] = time.time()
        self.print_stats()
        
        return generated
    
    def print_stats(self):
        """Print generation statistics."""
        elapsed = self.stats["end_time"] - self.stats["start_time"]
        rate = self.stats["successful"] / elapsed * 3600
        
        print("\n=== Batch Generation Statistics ===")
        print(f"Total attempted: {self.stats['total_generated']}")
        print(f"Successful: {self.stats['successful']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Success rate: {self.stats['successful'] / max(1, self.stats['total_generated']) * 100:.1f}%")
        print(f"Modules compiled: {self.stats['modules_compiled']}")
        print(f"Time elapsed: {elapsed:.1f}s")
        print(f"Rate: {rate:.0f} pairs/hour")
        print("\nBy type:")
        for ktype, count in sorted(self.stats["by_type"].items()):
            print(f"  {ktype}: {count}")


def run_batch_generation(output_dir: str, count: int = 10000, 
                         batch_size: int = 20, seed: int = 42):
    """Run batch generation with given parameters."""
    config = BatchConfig(
        output_dir=output_dir,
        kernels_per_module=batch_size,
        total_count=count,
        seed=seed
    )
    
    generator = BatchGenerator(config)
    return generator.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch generate Python→IR pairs")
    parser.add_argument("--output", type=str, default="../../data/batch_10k",
                        help="Output directory")
    parser.add_argument("--count", type=int, default=10000,
                        help="Number of pairs to generate")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Kernels per module compilation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    output_path = Path(__file__).parent / args.output
    output_path = output_path.resolve()
    
    run_batch_generation(str(output_path), args.count, args.batch_size, args.seed)

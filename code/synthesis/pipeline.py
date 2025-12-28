"""End-to-end synthesis pipeline: generate kernel → compile → extract IR → save."""
import os
import sys
import json
import hashlib
import tempfile
import importlib.util
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional

# Ensure repo root is on sys.path so `code.*` imports work when executed as a script.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from code.common.device import normalize_warp_target, resolve_warp_device
from code.extraction.ir_extractor import extract_kernel_functions, get_generated_source_path
from code.extraction.offline_codegen import codegen_module_source
from code.synthesis.generator import KernelGenerator, KernelSpec


@dataclass
class SynthesisPair:
    """A Python→IR training data pair."""
    id: str
    kernel_name: str
    kernel_type: str
    python_source: str
    cpp_ir_forward: str
    cpp_ir_backward: str
    cpp_ir_full: str
    generated_at: str
    metadata: dict


def hash_source(source: str) -> str:
    """Generate a short hash of the source code."""
    return hashlib.sha256(source.encode()).hexdigest()[:12]


def compile_kernel_from_source(python_source: str, kernel_name: str, device: str) -> tuple:
    """
    Compile a kernel from source and extract its IR.
    
    Returns: (kernel, module) or raises exception
    """
    import warp as wp
    target = normalize_warp_target(device)
    
    # Create a temporary module file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        # Write full module with imports
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
        
        # Force compilation
        wp_module = kernel.module
        # For CPU, always compile. For CUDA, compile only if CUDA runtime is available.
        if target == "cpu":
            wp_module.load("cpu")
        elif wp.is_cuda_available():
            wp_module.load("cuda")
        
        return kernel, wp_module
        
    finally:
        # Clean up temp file
        os.unlink(temp_path)


class SynthesisPipeline:
    """Pipeline for generating Python→IR training pairs."""
    
    def __init__(self, output_dir: str, seed: Optional[int] = None, device: str = "cpu"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generator = KernelGenerator(seed=seed)
        # `device` here means the intended codegen/compile target. CUDA may still be
        # supported in CPU-only environments via offline codegen.
        self.device = normalize_warp_target(device)
        self.stats = {
            "total_generated": 0,
            "successful": 0,
            "failed": 0,
            "by_type": {}
        }
    
    def generate_pair(self, kernel_type: Optional[str] = None) -> Optional[SynthesisPair]:
        """Generate a single Python→IR pair."""
        import warp as wp
        
        # Generate kernel spec
        spec = self.generator.generate(kernel_type)
        python_source = self.generator.to_python_source(spec)
        
        # Track type stats
        ktype = kernel_type or "random"
        self.stats["by_type"][ktype] = self.stats["by_type"].get(ktype, 0) + 1
        self.stats["total_generated"] += 1
        
        try:
            # Compile and extract IR
            kernel, wp_module = compile_kernel_from_source(python_source, spec.name, device=self.device)

            codegen_only = False
            source_file: Path | None = None

            if self.device == "cuda" and not wp.is_cuda_available():
                # Offline CUDA codegen: produce .cu source without requiring a GPU/driver.
                codegen_only = True
                cg = codegen_module_source(kernel.module, target="cuda", enable_backward=True)
                module_id = cg.module_id
                cpp_full = cg.source
            else:
                # Runtime-backed compilation: read generated source from cache.
                module_id = wp_module.get_module_identifier()
                source_file = get_generated_source_path(module_id, self.device)
                cpp_full = source_file.read_text()

            funcs = extract_kernel_functions(cpp_full, kernel.key, device=self.device)
            forward_ir, backward_ir = funcs.get("forward", ""), funcs.get("backward", "")
            
            if not forward_ir:
                raise ValueError("Failed to extract forward IR")
            
            # Create the pair
            pair_id = hash_source(python_source)
            pair = SynthesisPair(
                id=pair_id,
                kernel_name=spec.name,
                kernel_type=ktype,
                python_source=python_source,
                cpp_ir_forward=forward_ir,
                cpp_ir_backward=backward_ir,
                cpp_ir_full=cpp_full,
                generated_at=datetime.now().isoformat(),
                metadata={
                    "num_params": len(spec.params),
                    "num_lines": len(spec.body_lines),
                    "module_id": module_id,
                    "device": self.device,
                    "source_file": str(source_file) if source_file else None,
                    "codegen_only": codegen_only,
                }
            )
            
            self.stats["successful"] += 1
            return pair
            
        except Exception as e:
            self.stats["failed"] += 1
            print(f"Failed to generate pair for {spec.name}: {e}")
            return None
    
    def save_pair(self, pair: SynthesisPair) -> Path:
        """Save a pair to disk as JSON."""
        filename = f"{pair.id}_{pair.kernel_name}.json"
        filepath = self.output_dir / filename
        
        # Convert to dict, but store only essential C++ (not full module)
        data = {
            "id": pair.id,
            "kernel_name": pair.kernel_name,
            "kernel_type": pair.kernel_type,
            "device": pair.metadata.get("device", "cpu"),
            "python_source": pair.python_source,
            "cpp_ir_forward": pair.cpp_ir_forward,
            "cpp_ir_backward": pair.cpp_ir_backward,
            "generated_at": pair.generated_at,
            "metadata": pair.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def generate_batch(self, count: int, kernel_types: Optional[List[str]] = None) -> List[SynthesisPair]:
        """Generate a batch of pairs."""
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
        print("\n=== Synthesis Statistics ===")
        print(f"Total attempted: {self.stats['total_generated']}")
        print(f"Successful: {self.stats['successful']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Success rate: {self.stats['successful'] / max(1, self.stats['total_generated']) * 100:.1f}%")
        print("\nBy type:")
        for ktype, count in sorted(self.stats["by_type"].items()):
            print(f"  {ktype}: {count}")


def run_pipeline(output_dir: str, count: int = 100, seed: int = 42, device: str = "cpu"):
    """Run the synthesis pipeline."""
    import warp as wp
    wp.init()
    
    print(f"Starting synthesis pipeline (target: {count} pairs)")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    
    pipeline = SynthesisPipeline(output_dir, seed=seed, device=device)
    pairs = pipeline.generate_batch(count)
    
    pipeline.print_stats()
    
    print(f"\nGenerated {len(pairs)} valid pairs in {output_dir}")
    return pairs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Python→IR training pairs")
    parser.add_argument("--output", type=str, default="data/samples",
                        help="Output directory")
    parser.add_argument("--count", type=int, default=100, 
                        help="Number of pairs to generate")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Compilation/extraction device")
    
    args = parser.parse_args()
    
    # Resolve output path relative to repo root (matches README usage).
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (REPO_ROOT / output_path).resolve()
    
    run_pipeline(str(output_path), args.count, args.seed, device=args.device)

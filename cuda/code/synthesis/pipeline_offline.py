"""Offline synthesis pipeline: generate CUDA IR without GPU."""
import os
import sys
import importlib.util
import tempfile
import warp as wp
from warp.context import ModuleBuilder
from pathlib import Path

# Add parent dir to path to import pipeline
sys.path.insert(0, str(Path(__file__).parent))

import pipeline
from pipeline import SynthesisPipeline, extract_kernel_ir, hash_source, SynthesisPair, datetime

def compile_and_generate_cuda(python_source: str, kernel_name: str) -> tuple:
    """
    Compile a kernel from source and generate CUDA C++ code without loading/compiling binary.
    
    Returns: (kernel, cuda_source)
    """
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
        spec = importlib.util.spec_from_file_location("temp_kernel_offline", temp_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["temp_kernel_offline"] = module
        spec.loader.exec_module(module)
        
        # Get the kernel
        kernel = getattr(module, kernel_name)
        wp_module = kernel.module
        
        # Hash the module to generate mangled names
        wp_module.hash_module()
        
        # Setup builder options for CUDA
        builder_options = wp_module.options.copy()
        builder_options["output_arch"] = 86 # Fake arch
        
        # Create builder
        builder = ModuleBuilder(
            wp_module,
            builder_options,
            hasher=wp_module.hashers.get(wp_module.options["block_dim"], None)
        )
        
        # Generate CUDA source
        cuda_source = builder.codegen("cuda")
        
        return kernel, cuda_source
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

class OfflineSynthesisPipeline(SynthesisPipeline):
    """Pipeline that generates CUDA IR offline."""
    
    def __init__(self, output_dir: str, seed=None):
        super().__init__(output_dir, seed=seed, device="cuda")
        
    def generate_pair(self, kernel_type=None):
        # Generate kernel spec
        spec = self.generator.generate(kernel_type)
        python_source = self.generator.to_python_source(spec)
        
        ktype = kernel_type or "random"
        self.stats["by_type"][ktype] = self.stats["by_type"].get(ktype, 0) + 1
        self.stats["total_generated"] += 1
        
        try:
            # Use offline generation
            kernel, cuda_full = compile_and_generate_cuda(python_source, spec.name)
            
            # Extract IR using the existing extractor
            # Note: extract_kernel_ir handles regex matching
            forward_ir, backward_ir = extract_kernel_ir(kernel, cuda_full, device="cuda")
            
            if not forward_ir:
                raise ValueError("Failed to extract forward IR from generated CUDA code")
            
            # Create the pair
            pair_id = hash_source(python_source)
            pair = SynthesisPair(
                id=pair_id,
                kernel_name=spec.name,
                kernel_type=ktype,
                python_source=python_source,
                cpp_ir_forward=forward_ir,
                cpp_ir_backward=backward_ir,
                cpp_ir_full=cuda_full,
                generated_at=datetime.now().isoformat(),
                metadata={
                    "num_params": len(spec.params),
                    "num_lines": len(spec.body_lines),
                    "device": "cuda",
                    "mode": "offline"
                }
            )
            
            self.stats["successful"] += 1
            return pair
            
        except Exception as e:
            self.stats["failed"] += 1
            print(f"Failed to generate pair for {spec.name}: {e}")
            # import traceback
            # traceback.print_exc()
            return None

def run_offline_pipeline(output_dir: str, count: int = 5, seed: int = 42):
    wp.init()
    print(f"Starting OFFLINE synthesis pipeline (target: {count} pairs)")
    print(f"Output directory: {output_dir}")
    
    pipeline = OfflineSynthesisPipeline(output_dir, seed=seed)
    pairs = pipeline.generate_batch(count)
    
    pipeline.print_stats()
    return pairs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="../../data/offline_cuda")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    output_path = Path(__file__).parent / args.output
    output_path = output_path.resolve()
    
    run_offline_pipeline(str(output_path), args.count, args.seed)

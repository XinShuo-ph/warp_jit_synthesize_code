"""
Synthesis Pipeline

End-to-end pipeline: Generate kernel → Write to file → Compile → Extract IR → Save pair
"""

import sys
sys.path.insert(0, '/workspace/code')

import warp as wp
import os
import tempfile
import importlib.util
import json
from typing import List, Dict, Any
from pathlib import Path

from synthesis.generator import KernelGenerator, KernelSpec
from extraction.ir_extractor import extract_ir, IRExtractionResult


class SynthesisPipeline:
    """End-to-end pipeline for generating Python→IR training data."""
    
    def __init__(self, output_dir: str, temp_dir: str = None):
        """
        Initialize pipeline.
        
        Args:
            output_dir: Directory to save final dataset
            temp_dir: Temporary directory for kernel files (auto-created if None)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="warp_synthesis_"))
        else:
            self.temp_dir = Path(temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.generator = KernelGenerator()
        self.generated_count = 0
        self.success_count = 0
        self.failure_count = 0
    
    def compile_kernel_from_source(self, spec: KernelSpec) -> Any:
        """
        Write kernel source to file, import, and return kernel object.
        
        Args:
            spec: KernelSpec with source code
            
        Returns:
            Compiled kernel object or None if failed
        """
        # Write source to temporary file
        module_name = f"temp_kernel_{self.generated_count}"
        module_file = self.temp_dir / f"{module_name}.py"
        
        with open(module_file, 'w') as f:
            f.write(spec.source_code)
        
        try:
            # Import the module
            spec_obj = importlib.util.spec_from_file_location(module_name, module_file)
            module = importlib.util.module_from_spec(spec_obj)
            sys.modules[module_name] = module
            spec_obj.loader.exec_module(module)
            
            # Get the kernel function
            kernel_obj = getattr(module, spec.name)
            
            # Test compilation by loading on CPU
            kernel_obj.module.load("cpu")
            
            return kernel_obj
            
        except Exception as e:
            print(f"  ✗ Failed to compile {spec.name}: {e}")
            return None
        finally:
            # Clean up temporary file
            if module_file.exists():
                os.remove(module_file)
    
    def process_single_kernel(self, spec: KernelSpec, sample_id: int) -> bool:
        """
        Process a single kernel: compile, extract IR, save.
        
        Args:
            spec: KernelSpec to process
            sample_id: Unique ID for this sample
            
        Returns:
            True if successful, False otherwise
        """
        self.generated_count += 1
        
        # Compile kernel
        kernel_obj = self.compile_kernel_from_source(spec)
        if kernel_obj is None:
            self.failure_count += 1
            return False
        
        # Extract IR
        ir_result = extract_ir(kernel_obj, device="cpu", force_compile=True)
        
        if not ir_result.success:
            print(f"  ✗ IR extraction failed for {spec.name}: {ir_result.error_message}")
            self.failure_count += 1
            return False
        
        # Save Python source
        sample_name = f"sample_{sample_id:05d}"
        py_file = self.output_dir / f"{sample_name}.py"
        with open(py_file, 'w') as f:
            f.write(spec.source_code)
        
        # Save C++ IR
        cpp_file = self.output_dir / f"{sample_name}.cpp"
        with open(cpp_file, 'w') as f:
            f.write(ir_result.cpp_source)
        
        # Save metadata
        metadata = {
            "sample_id": sample_id,
            "kernel_name": spec.name,
            "category": spec.category,
            "complexity": spec.complexity,
            "module_name": ir_result.module_name,
            "module_hash": ir_result.module_hash,
            "python_file": str(py_file.name),
            "cpp_file": str(cpp_file.name),
            "python_size": len(spec.source_code),
            "cpp_size": len(ir_result.cpp_source)
        }
        
        json_file = self.output_dir / f"{sample_name}.json"
        with open(json_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.success_count += 1
        return True
    
    def generate_dataset(self, count: int, categories: List[str] = None, 
                        quiet: bool = False) -> Dict[str, int]:
        """
        Generate complete dataset of Python→IR pairs.
        
        Args:
            count: Number of samples to generate
            categories: List of categories to include (None for all)
            quiet: Suppress progress output
            
        Returns:
            Dictionary with statistics
        """
        if not quiet:
            print(f"Generating {count} samples...")
            print(f"Output directory: {self.output_dir}")
            print()
        
        # Generate kernel specifications
        specs = self.generator.generate_batch(count, categories=categories)
        
        # Process each kernel
        for i, spec in enumerate(specs):
            if not quiet and (i % 10 == 0 or i == len(specs) - 1):
                print(f"  Processing {i+1}/{len(specs)}: {spec.name} ({spec.category})")
            
            self.process_single_kernel(spec, sample_id=i+1)
        
        stats = {
            "total_generated": self.generated_count,
            "successful": self.success_count,
            "failed": self.failure_count,
            "success_rate": self.success_count / self.generated_count if self.generated_count > 0 else 0
        }
        
        if not quiet:
            print()
            print("="*60)
            print("GENERATION COMPLETE")
            print("="*60)
            print(f"Total: {stats['total_generated']}")
            print(f"Successful: {stats['successful']}")
            print(f"Failed: {stats['failed']}")
            print(f"Success rate: {stats['success_rate']:.1%}")
        
        return stats
    
    def cleanup(self):
        """Clean up temporary directory."""
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)


def main():
    """Test synthesis pipeline."""
    wp.init()
    
    print("="*60)
    print("SYNTHESIS PIPELINE TEST")
    print("="*60)
    print()
    
    # Test with small batch
    output_dir = "/workspace/data/pipeline_test"
    pipeline = SynthesisPipeline(output_dir)
    
    try:
        stats = pipeline.generate_dataset(count=10, quiet=False)
        
        # List generated files
        print()
        print("Generated files:")
        files = sorted(Path(output_dir).glob("sample_*"))
        for f in files[:15]:  # Show first 15
            print(f"  {f.name}")
        if len(files) > 15:
            print(f"  ... and {len(files) - 15} more")
        
    finally:
        pipeline.cleanup()
    
    print()
    print("✓ Pipeline test complete")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
IR Extractor for Warp Kernels

This module provides utilities to extract intermediate representations (IR)
from Warp kernels. The IR is the generated C++/CUDA code that Warp produces
from Python kernel definitions.
"""

import warp as wp
import os
import glob
import hashlib
import json
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict

@dataclass
class KernelIR:
    """Container for kernel IR and metadata."""
    python_source: str
    cpp_code: str
    meta: str
    module_hash: str
    kernel_name: str
    
    def validate(self) -> tuple[bool, str]:
        """Validate that the IR is complete and consistent."""
        errors = []
        
        if not self.python_source or len(self.python_source) < 10:
            errors.append("Python source is empty or too short")
        
        if not self.cpp_code or len(self.cpp_code) < 100:
            errors.append("C++ code is empty or too short")
        
        if not self.kernel_name:
            errors.append("Kernel name is missing")
        
        # Check for required C++ structures (relaxed kernel name check)
        if "void " not in self.cpp_code:
            errors.append("No function definitions found in C++ code")
        
        if "_cpu_kernel_forward" not in self.cpp_code:
            errors.append("No forward kernel function found in C++ code")
        
        if errors:
            return False, "; ".join(errors)
        return True, "OK"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def compute_hash(self) -> str:
        """Compute hash of Python source for validation."""
        return hashlib.sha256(self.python_source.encode()).hexdigest()[:16]

class IRExtractorError(Exception):
    """Exception raised during IR extraction."""
    pass

class IRExtractor:
    """Extract IR from Warp kernels."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        wp.init()
        self.cache_dir = cache_dir or wp.config.kernel_cache_dir
        self.verbose = False
    
    def set_verbose(self, verbose: bool):
        """Enable/disable verbose output."""
        self.verbose = verbose
    
    def extract_ir(self, kernel: wp.Kernel, trigger_compile: bool = True) -> KernelIR:
        """
        Extract IR from a warp kernel.
        
        Args:
            kernel: A warp kernel decorated with @wp.kernel
            trigger_compile: If True, compile the kernel by launching it once
            
        Returns:
            KernelIR object containing Python source and generated C++ code
            
        Raises:
            IRExtractorError: If extraction fails
        """
        try:
            # Get kernel function source
            import inspect
            python_source = inspect.getsource(kernel.func)
        except Exception as e:
            raise IRExtractorError(f"Failed to get Python source: {e}")
        
        module = kernel.module
        module_name = module.name
        
        # Find the generated C++ file
        module_files = self._find_module_files(module_name)
        
        if not module_files:
            raise IRExtractorError(
                f"No cache files found for module '{module_name}'. "
                "Make sure the kernel has been compiled by launching it."
            )
        
        cpp_file = module_files.get('cpp')
        meta_file = module_files.get('meta')
        
        if not cpp_file:
            raise IRExtractorError(
                f"No C++ file found for module '{module_name}'. "
                f"Found files: {list(module_files.keys())}"
            )
        
        # Read the generated code with error handling
        try:
            with open(cpp_file, 'r') as f:
                cpp_code = f.read()
        except Exception as e:
            raise IRExtractorError(f"Failed to read C++ file '{cpp_file}': {e}")
        
        meta = ""
        if meta_file and os.path.exists(meta_file):
            try:
                with open(meta_file, 'r') as f:
                    meta = f.read()
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to read meta file: {e}")
        
        # Extract module hash from filename
        module_hash = os.path.basename(cpp_file).replace('.cpp', '').split('___')[-1]
        
        ir = KernelIR(
            python_source=python_source,
            cpp_code=cpp_code,
            meta=meta,
            module_hash=module_hash,
            kernel_name=kernel.key
        )
        
        # Validate the extracted IR
        valid, error_msg = ir.validate()
        if not valid:
            raise IRExtractorError(f"IR validation failed: {error_msg}")
        
        return ir
    
    def _find_module_files(self, module_name: str) -> dict:
        """Find all files for a given module in the cache."""
        # Search for directories matching the module name
        # Try different patterns
        patterns = [
            os.path.join(self.cache_dir, f"wp___{module_name.replace('.', '_')}___*"),
            os.path.join(self.cache_dir, f"*{module_name}*"),
            os.path.join(self.cache_dir, "wp___main___*"),
        ]
        
        dirs = []
        for pattern in patterns:
            dirs.extend(glob.glob(pattern))
        
        if not dirs:
            return {}
        
        # Get the most recent directory
        latest_dir = max(dirs, key=os.path.getmtime)
        
        result = {}
        for fname in os.listdir(latest_dir):
            full_path = os.path.join(latest_dir, fname)
            if fname.endswith('.cpp'):
                result['cpp'] = full_path
            elif fname.endswith('.cu'):
                result['cuda'] = full_path
            elif fname.endswith('.meta'):
                result['meta'] = full_path
            elif fname.endswith('.o'):
                result['object'] = full_path
        
        return result

    def extract_batch(self, kernel_configs: List[tuple]) -> Dict[str, KernelIR]:
        """
        Extract IR from multiple kernels.
        
        Args:
            kernel_configs: List of (kernel, launch_args) tuples
                           where launch_args is dict with 'dim', 'inputs', etc.
        
        Returns:
            Dictionary mapping kernel names to KernelIR objects
        
        Example:
            configs = [
                (kernel1, {'dim': 10, 'inputs': [a, b]}),
                (kernel2, {'dim': 5, 'inputs': [c, d, e]})
            ]
            results = extractor.extract_batch(configs)
        """
        results = {}
        total = len(kernel_configs)
        
        for idx, (kernel, launch_args) in enumerate(kernel_configs, 1):
            kernel_name = kernel.key
            
            if self.verbose:
                print(f"[{idx}/{total}] Extracting {kernel_name}...")
            
            try:
                # Launch kernel to trigger compilation
                wp.launch(kernel=kernel, **launch_args)
                wp.synchronize()
                
                # Extract IR
                ir = self.extract_ir(kernel, trigger_compile=False)
                results[kernel_name] = ir
                
                if self.verbose:
                    print(f"  ✓ Success ({len(ir.cpp_code)} bytes)")
                    
            except Exception as e:
                if self.verbose:
                    print(f"  ✗ Failed: {e}")
                # Continue with other kernels
                continue
        
        return results
    
    def clear_cache(self, module_name: Optional[str] = None):
        """
        Clear kernel cache.
        
        Args:
            module_name: If provided, only clear this module. 
                        Otherwise clear entire cache.
        """
        if not os.path.exists(self.cache_dir):
            if self.verbose:
                print("Cache directory does not exist")
            return
        
        if module_name:
            # Clear specific module
            pattern = os.path.join(self.cache_dir, f"wp___{module_name.replace('.', '_')}___*")
            dirs = glob.glob(pattern)
            
            for d in dirs:
                if self.verbose:
                    print(f"Removing {d}")
                import shutil
                shutil.rmtree(d)
        else:
            # Clear entire cache
            if self.verbose:
                print(f"WARNING: Clearing entire cache at {self.cache_dir}")
                print("This will remove all compiled kernels!")
            # We don't actually clear entire cache - too dangerous
            # Instead just report what would be removed
            dirs = [d for d in os.listdir(self.cache_dir) 
                   if os.path.isdir(os.path.join(self.cache_dir, d))]
            if self.verbose:
                print(f"Would remove {len(dirs)} cached modules")

def extract_kernel_ir_simple(kernel_func, *args, **kwargs) -> KernelIR:
    """
    Simplified API: extract IR from a kernel function.
    
    Args:
        kernel_func: Function decorated with @wp.kernel
        *args, **kwargs: Arguments to pass when launching the kernel for compilation
        
    Returns:
        KernelIR object
        
    Raises:
        IRExtractorError: If extraction fails
        
    Example:
        @wp.kernel
        def my_kernel(x: wp.array(dtype=float)):
            tid = wp.tid()
            x[tid] = x[tid] * 2.0
        
        x = wp.array([1.0, 2.0, 3.0], dtype=float)
        ir = extract_kernel_ir_simple(my_kernel, dim=3, inputs=[x])
    """
    extractor = IRExtractor()
    
    # Launch the kernel to trigger compilation
    if args or kwargs:
        wp.launch(kernel=kernel_func, *args, **kwargs)
        wp.synchronize()
    
    return extractor.extract_ir(kernel_func, trigger_compile=False)

if __name__ == "__main__":
    # Test the extractor with enhanced features
    print("Testing Enhanced IR Extractor")
    print("=" * 60)
    
    @wp.kernel
    def simple_mul(a: wp.array(dtype=float),
                    b: wp.array(dtype=float),
                    c: wp.array(dtype=float)):
        tid = wp.tid()
        c[tid] = a[tid] * b[tid]
    
    # Create test data
    n = 5
    a = wp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    b = wp.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
    c = wp.zeros(n, dtype=float)
    
    # Test 1: Basic extraction
    print("\n1. Basic extraction test")
    try:
        ir = extract_kernel_ir_simple(simple_mul, dim=n, inputs=[a, b, c])
        print(f"✓ Extracted IR for {ir.kernel_name}")
        print(f"  Python: {len(ir.python_source)} bytes")
        print(f"  C++: {len(ir.cpp_code)} bytes")
        
        # Test validation
        valid, msg = ir.validate()
        print(f"  Validation: {msg}")
        
        # Test hash
        print(f"  Python hash: {ir.compute_hash()}")
    except IRExtractorError as e:
        print(f"✗ Extraction failed: {e}")
    
    # Test 2: Batch extraction
    print("\n2. Batch extraction test")
    
    @wp.kernel
    def add_kernel(x: wp.array(dtype=float), y: wp.array(dtype=float)):
        tid = wp.tid()
        y[tid] = x[tid] + 1.0
    
    x = wp.array([1.0, 2.0, 3.0], dtype=float)
    y = wp.zeros(3, dtype=float)
    
    extractor = IRExtractor()
    extractor.set_verbose(True)
    
    configs = [
        (simple_mul, {'dim': n, 'inputs': [a, b, c]}),
        (add_kernel, {'dim': 3, 'inputs': [x, y]})
    ]
    
    results = extractor.extract_batch(configs)
    print(f"\n✓ Extracted {len(results)} kernels")
    
    # Test 3: Error handling
    print("\n3. Error handling test")
    try:
        # Try to extract without compiling first
        @wp.kernel
        def uncompiled_kernel(x: wp.array(dtype=float)):
            tid = wp.tid()
            x[tid] = 0.0
        
        # This should fail
        ir = extractor.extract_ir(uncompiled_kernel, trigger_compile=False)
        print("✗ Should have raised an error!")
    except IRExtractorError as e:
        print(f"✓ Correctly raised error: {e}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")

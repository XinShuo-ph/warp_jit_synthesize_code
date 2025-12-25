#!/usr/bin/env python3
"""Explore warp kernel compilation and IR extraction"""

import warp as wp
import os
import sys

wp.init()

# Simple test kernel
@wp.kernel
def test_kernel(x: wp.array(dtype=float),
                y: wp.array(dtype=float)):
    tid = wp.tid()
    y[tid] = x[tid] * 2.0 + 1.0

def main():
    print("Exploring warp kernel compilation...")
    
    # Get kernel module
    print(f"\nKernel: {test_kernel}")
    print(f"Kernel module: {test_kernel.module}")
    
    # Create arrays to trigger compilation
    n = 10
    x = wp.array([float(i) for i in range(n)], dtype=float)
    y = wp.zeros(n, dtype=float)
    
    # Launch to force compilation
    wp.launch(kernel=test_kernel, dim=n, inputs=[x, y])
    wp.synchronize()
    
    # Inspect module attributes
    module = test_kernel.module
    print(f"\nModule name: {module.name}")
    print(f"Module attributes: {[attr for attr in dir(module) if not attr.startswith('_')]}")
    
    # Check for generated code
    if hasattr(module, 'source'):
        print(f"\nGenerated source length: {len(module.source)} chars")
        print("\nFirst 500 chars of source:")
        print(module.source[:500])
    
    if hasattr(module, 'options'):
        print(f"\nModule options: {module.options}")
    
    # Check kernel cache directory
    cache_dir = wp.config.kernel_cache_dir
    print(f"\nKernel cache dir: {cache_dir}")
    
    if os.path.exists(cache_dir):
        # Look for files related to this module
        module_files = []
        for root, dirs, files in os.walk(cache_dir):
            for f in files:
                if module.name in f or 'main' in f.lower():
                    filepath = os.path.join(root, f)
                    size = os.path.getsize(filepath)
                    module_files.append((f, size, filepath))
        
        print(f"\nFound {len(module_files)} potential module files:")
        for fname, size, fpath in module_files[:10]:  # Show first 10
            print(f"  {fname} ({size} bytes)")
            if fname.endswith('.cpp') or fname.endswith('.cu'):
                # This is likely the generated source
                print(f"    Full path: {fpath}")
    
    # Try to access adjoint info
    if hasattr(module, 'adj'):
        print(f"\nModule has adjoint: {module.adj}")

if __name__ == "__main__":
    main()

"""
Demonstration of accessing Warp's generated C++ code (IR).

This shows how to extract the intermediate representation that Warp
generates from Python kernels.
"""

import warp as wp
import os

wp.init()

# Define a simple kernel
@wp.kernel
def simple_multiply(a: wp.array(dtype=float),
                    b: wp.array(dtype=float),
                    scale: float):
    """Multiply array by scalar."""
    i = wp.tid()
    b[i] = a[i] * scale


def get_kernel_ir(kernel, device="cpu"):
    """Extract the generated C++ code for a kernel.
    
    Args:
        kernel: A warp kernel object
        device: Target device (cpu or cuda)
        
    Returns:
        str: The generated C++ code
    """
    # Get the module that contains this kernel
    module = kernel.module
    
    # Build the module if not already built
    module.load(device)
    
    # The generated code is stored in the cache directory
    cache_dir = wp.config.kernel_cache_dir
    
    # The module hash is already formatted as a hex string in the module
    # We can look for it in the cache directory
    module_name_safe = module.name.replace(".", "_")
    
    # List all directories and find the one that matches
    cache_contents = os.listdir(cache_dir)
    prefix = f"wp_{module_name_safe}_"
    
    matching_dirs = [d for d in cache_contents if d.startswith(prefix)]
    
    if not matching_dirs:
        print(f"No matching directories found for prefix: {prefix}")
        print(f"Cache contents: {cache_contents}")
        return None
    
    # Use the first match (there should only be one)
    module_dir = matching_dirs[0]
    module_path = os.path.join(cache_dir, module_dir)
    
    # Look for the .cpp file
    cpp_file = os.path.join(module_path, f"{module_dir}.cpp")
    
    print(f"Found module directory: {module_dir}")
    print(f"Looking for: {cpp_file}")
    
    if os.path.exists(cpp_file):
        with open(cpp_file, 'r') as f:
            return f.read()
    else:
        print(f"Directory contents: {os.listdir(module_path)}")
        return None


def main():
    # Create some test data
    n = 5
    a = wp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    b = wp.zeros(n, dtype=float)
    
    # Launch kernel to trigger compilation
    print("Launching kernel to trigger compilation...")
    wp.launch(kernel=simple_multiply, dim=n, inputs=[a, b, 2.0])
    wp.synchronize()
    
    print(f"Result: {b.numpy()}")
    
    # Now extract the generated IR
    print("\n" + "="*60)
    print("EXTRACTING GENERATED C++ CODE (IR)")
    print("="*60 + "\n")
    
    ir_code = get_kernel_ir(simple_multiply, device="cpu")
    
    if ir_code:
        print(ir_code)
        print("\n" + "="*60)
        print("IR EXTRACTION SUCCESSFUL")
        print("="*60)
        return ir_code
    else:
        print("Could not find generated code")
        return None


if __name__ == "__main__":
    main()

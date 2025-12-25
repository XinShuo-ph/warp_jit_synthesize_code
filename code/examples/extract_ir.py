"""
Script to extract generated IR/C++ code from warp kernels
"""
import warp as wp
import os
import json

wp.init()

def extract_kernel_ir(kernel, kernel_name=None):
    """
    Extract the intermediate representation (C++ code) from a warp kernel
    
    Args:
        kernel: A warp kernel decorated with @wp.kernel
        kernel_name: Optional name for the kernel (defaults to kernel.key)
    
    Returns:
        dict with IR information
    """
    if kernel_name is None:
        kernel_name = kernel.key
    
    # Get the module
    module = kernel.module
    
    # Force compilation if not already compiled
    # We need to launch the kernel once to generate the code
    # This is done by the caller
    
    # Get cache directory
    cache_dir = wp.config.kernel_cache_dir
    
    # Find the cached files for this module
    module_hash = module.options.get('module_hash', None)
    
    # Search for generated files
    ir_data = {
        'kernel_name': kernel_name,
        'kernel_key': kernel.key,
        'module_name': module.name,
        'signature': kernel.sig,
        'cpp_code': None,
        'cpp_file': None,
        'metadata': None
    }
    
    # Look through cache directory
    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            if file.endswith('.cpp'):
                # Check if this file contains our kernel
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        # Look for kernel function names
                        if kernel.key in content or kernel_name in content:
                            ir_data['cpp_code'] = content
                            ir_data['cpp_file'] = filepath
                            
                            # Try to find corresponding .meta file
                            meta_file = filepath.replace('.cpp', '.meta')
                            if os.path.exists(meta_file):
                                with open(meta_file, 'r') as mf:
                                    ir_data['metadata'] = json.load(mf)
                            break
                except Exception as e:
                    pass
        if ir_data['cpp_code']:
            break
    
    return ir_data

def extract_kernel_function_from_cpp(cpp_code, kernel_name):
    """
    Extract the main kernel function from the full C++ code
    Returns the forward kernel implementation
    """
    if not cpp_code:
        return None
    
    # Look for the CPU kernel forward function
    lines = cpp_code.split('\n')
    
    # Find the function definition
    in_function = False
    function_lines = []
    brace_count = 0
    
    for i, line in enumerate(lines):
        if '_cpu_kernel_forward' in line or '_cpu_forward' in line:
            # Look backwards to find start of function
            j = i
            while j >= 0 and 'void' not in lines[j]:
                j -= 1
            if j >= 0:
                in_function = True
                function_lines = []
                i = j
        
        if in_function:
            function_lines.append(line)
            brace_count += line.count('{') - line.count('}')
            
            # If we've closed all braces, we're done
            if brace_count == 0 and '{' in ''.join(function_lines):
                break
    
    return '\n'.join(function_lines) if function_lines else None

# Test with our example kernel
@wp.kernel
def example_kernel(x: wp.array(dtype=float), 
                   y: wp.array(dtype=float),
                   alpha: float):
    tid = wp.tid()
    y[tid] = x[tid] * alpha + wp.sin(x[tid])

if __name__ == "__main__":
    import numpy as np
    
    print("=" * 80)
    print("IR EXTRACTION TEST")
    print("=" * 80)
    
    # Must launch kernel to force compilation
    n = 10
    x = wp.array(np.linspace(0, 3.14, n), dtype=wp.float32)
    y = wp.zeros(n, dtype=wp.float32)
    alpha = 2.0
    
    print("\n1. Launching kernel to force compilation...")
    wp.launch(example_kernel, dim=n, inputs=[x, y, alpha])
    print("   Done.")
    
    # Extract IR
    print("\n2. Extracting IR...")
    ir_data = extract_kernel_ir(example_kernel)
    
    print(f"\n3. Kernel Information:")
    print(f"   Name: {ir_data['kernel_name']}")
    print(f"   Key: {ir_data['kernel_key']}")
    print(f"   Module: {ir_data['module_name']}")
    print(f"   Signature: {ir_data['signature']}")
    print(f"   C++ file: {ir_data['cpp_file']}")
    
    if ir_data['cpp_code']:
        print(f"\n4. Generated C++ code length: {len(ir_data['cpp_code'])} characters")
        print("\n5. Forward kernel function:")
        print("   " + "-" * 76)
        kernel_func = extract_kernel_function_from_cpp(ir_data['cpp_code'], ir_data['kernel_key'])
        if kernel_func:
            for line in kernel_func.split('\n')[:50]:
                print("   " + line)
        else:
            # Just show first 50 lines
            for line in ir_data['cpp_code'].split('\n')[:50]:
                print("   " + line)
        print("   " + "-" * 76)
    
    if ir_data['metadata']:
        print(f"\n6. Metadata: {ir_data['metadata']}")
    
    print("\n" + "=" * 80)
    print("IR extraction successful!")

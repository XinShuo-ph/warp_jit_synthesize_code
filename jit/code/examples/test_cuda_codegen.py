import warp as wp
import warp._src.context as ctx
import warp._src.codegen
import sys

# Initialize warp (will fail to find CUDA but continue in CPU mode)
wp.init()

@wp.kernel
def simple_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]

def test_cuda_codegen():
    module = simple_kernel.module
    hasher = ctx.ModuleHasher(module)
    options = module.options.copy() if module.options else {}
    
    # Try to force CUDA codegen
    builder = ctx.ModuleBuilder(module, options, hasher)
    try:
        cuda_code = builder.codegen("cuda")
        print("CUDA Codegen successful!")
        print("First 200 chars:")
        print(cuda_code[:200])
        
        # Check if we can find the kernel function
        mangled_name = simple_kernel.get_mangled_name()
        func_name = f"{mangled_name}_cuda_kernel_forward"
        print(f"\nLooking for function: {func_name}")
        
        import re
        # Modified regex to allow for attributes before void (though we search for 'void ...')
        # The current extractor searches for 'void func_name', which should match 'extern "C" __global__ void func_name'
        pattern = rf'void {re.escape(func_name)}\s*\([^)]*\)\s*\{{'
        match = re.search(pattern, cuda_code)
        
        if match:
            print("Found kernel function definition via regex!")
        else:
            print("Failed to find kernel function definition via regex.")
            print("Searching for string directly:")
            if func_name in cuda_code:
                print(f"Found '{func_name}' string in code.")
                # Show context
                idx = cuda_code.find(func_name)
                print(f"Context:\n{cuda_code[idx-50:idx+50]}")
            else:
                print(f"'{func_name}' not found in code.")
                
    except Exception as e:
        print(f"CUDA Codegen failed: {e}")

if __name__ == "__main__":
    test_cuda_codegen()

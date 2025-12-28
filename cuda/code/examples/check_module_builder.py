import warp as wp
import sys
import os

# Import ModuleBuilder
try:
    from warp.context import ModuleBuilder
    print("Imported ModuleBuilder from warp.context")
except ImportError:
    try:
        from warp._src.context import ModuleBuilder
        print("Imported ModuleBuilder from warp._src.context")
    except ImportError:
        print("Could not import ModuleBuilder")

@wp.kernel
def test_add(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]

def generate_offline_cuda():
    print("Generating offline CUDA via ModuleBuilder...")
    
    kernel = test_add
    module = kernel.module
    
    # Hash the module (required for mangled names)
    module.hash_module()
    
    # Prepare options
    # We probably need to fake an output_arch to satisfy some checks, 
    # but for pure codegen it might not be strictly required if we set it in builder_options.
    
    builder_options = module.options.copy()
    builder_options["output_arch"] = 86 # sm_86
    
    try:
        builder = ModuleBuilder(
            module,
            builder_options,
            hasher=module.hashers.get(module.options["block_dim"], None)
        )
        
        # Generate CUDA source
        cu_source = builder.codegen("cuda")
        
        print("\n=== Generated CUDA Source (First 500 chars) ===")
        print(cu_source[:500])
        
        if "__global__ void" in cu_source:
            print("\nSUCCESS: Found CUDA kernel signature")
        
        return cu_source
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    wp.init()
    generate_offline_cuda()

import warp as wp
import warp.codegen
import warp._src.context
import sys

wp.init()

@wp.kernel
def simple_kernel(a: wp.array(dtype=float)):
    tid = wp.tid()
    a[tid] = a[tid] * 2.0

kernel_func = simple_kernel

print("Initializing Module and Builder...")
module = warp._src.context.Module("test_module")
options = {
    "enable_backward": False
}

builder = warp._src.context.ModuleBuilder(module, options)

print("Building kernel...")
try:
    builder.build_kernel(kernel_func)
    print("Kernel build successful (IR generation stage).")
    
    # Check IR blocks directly
    if hasattr(kernel_func.adj, 'blocks'):
        print(f"\nNumber of blocks: {len(kernel_func.adj.blocks)}")
        print("First block forward body:")
        for line in kernel_func.adj.blocks[0].body_forward:
            print(f"  {line.strip()}")
            
    # Try to generate C++ code (might fail due to hash)
    device = "cpu"
    try:
        cpp_source = wp.codegen.codegen_kernel(kernel_func, device, options)
        print("\n--- Generated C++ Source ---\n")
        print(cpp_source)
        print("\n----------------------------\n")
    except Exception as e:
        print(f"\nCould not generate full C++ source (expected): {e}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

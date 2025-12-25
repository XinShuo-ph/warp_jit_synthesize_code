import warp as wp
import sys
import os
from warp._src.context import Module, ModuleBuilder

# Add the current directory to path so we can import the extractor
sys.path.append(os.getcwd())

wp.init()

@wp.kernel
def simple_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float)):
    tid = wp.tid()
    b[tid] = a[tid] * 2.0

def inspect_kernel():
    print("Inspecting kernel...")
    
    kernel = simple_kernel
    adj = kernel.adj
    
    # Create necessary objects for building
    # We can use a dummy module or the kernel's module
    module = kernel.module
    
    options = {
        "max_unroll": 4,
        "enable_backward": False,
        "fast_math": False,
        "fuse_fp": True,
        "lineinfo": False,
        "cuda_output": None,
        "mode": "cpu",
        "optimization_level": 3,
        "block_dim": 256,
        "compile_time_trace": False,
        "strip_hash": False,
    }
    
    print("Building adjoint...")
    builder = ModuleBuilder(module, options)
    
    # This should populate blocks and return_var
    # Note: build() might side-effect the Adjoint object which is shared.
    # If we want to be safe we might want to reload/reparse, but for now let's modify it.
    adj.build(builder)
    
    print(f"Return var: {adj.return_var}")
    print(f"Blocks count: {len(adj.blocks)}")
    
    # Now try codegen_func again
    from warp._src.codegen import codegen_func
    
    try:
        source = codegen_func(
            adj,
            c_func_name="simple_kernel_cpu",
            device="cpu",
            options=options,
            forward_only=True
        )
        print("\n--- Generated Source (CPU) ---")
        print(source[:500] + "..." if len(source) > 500 else source)
    except Exception as e:
        print(f"Error during manual codegen: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_kernel()

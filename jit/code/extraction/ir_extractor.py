import warp as wp
import warp.codegen
import warp.context
import traceback

def get_kernel_ir(kernel: wp.Kernel, device: str = "cpu") -> str:
    """
    Extracts the generated C++/CUDA source code (IR) for a given Warp kernel.
    
    Args:
        kernel: The Warp kernel object (decorated with @wp.kernel).
        device: Target device ("cpu" or "cuda").
        
    Returns:
        The generated source code string.
    """
    # Set default options
    options = {
        "block_dim": 256,
        "mode": "release",
        "lineinfo": False,
        "enable_backward": False
    }
    
    # Ensure module options has block_dim so hash_module works
    if kernel.module:
        if "block_dim" not in kernel.module.options:
             kernel.module.options["block_dim"] = 256
        
        # Trigger hash computation to ensure kernel has a hash
        kernel.module.hash_module()
    
    # Workaround for missing line number in some environments
    if kernel.adj.fun_def_lineno is None:
        kernel.adj.fun_def_lineno = 1
        
    # Build the Adjoint object (analyzes AST)
    # This populates variables and sets builder_options
    try:
        # We pass None as builder, so it uses default_builder_options
        kernel.adj.build(None, default_builder_options=options)
    except Exception as e:
        traceback.print_exc()
        return f"Error building kernel adjacency: {e}"

    try:
        source = wp.codegen.codegen_kernel(kernel, device=device, options=options)
        return source
    except Exception as e:
        traceback.print_exc()
        return f"Error extracting IR: {e}"

@wp.kernel
def test_kernel(a: wp.array(dtype=float)):
    tid = wp.tid()
    a[tid] = a[tid] * 2.0

if __name__ == "__main__":
    wp.init()
    
    print("--- CPU IR ---")
    ir_cpu = get_kernel_ir(test_kernel, device="cpu")
    print(ir_cpu)
    
    print("\n--- CUDA IR ---")
    ir_cuda = get_kernel_ir(test_kernel, device="cuda")
    print(ir_cuda)

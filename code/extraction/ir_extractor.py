import warp as wp
import warp._src.context
import warp._src.codegen
from typing import Optional

def get_kernel_ir(kernel: wp.kernel, device: str = "cpu") -> str:
    """
    Extracts the C++/CUDA Intermediate Representation (source code) for a Warp kernel.
    
    Args:
        kernel: The Warp kernel object (decorated with @wp.kernel).
        device: The target device ("cpu" or "cuda"). Defaults to "cpu".
        
    Returns:
        str: The generated source code (IR).
    """
    if not isinstance(kernel, warp._src.context.Kernel):
        raise ValueError("Input must be a Warp kernel object.")

    # Ensure the module is built or at least prepared for codegen
    # We use a copy of options to avoid modifying the original module options unexpectedly
    options = kernel.module.options.copy()
    
    # We need to build the kernel to ensure analysis is done and hash is computed
    # The ModuleBuilder handles the heavy lifting of analyzing dependencies
    builder = warp._src.context.ModuleBuilder(kernel.module, options=options)
    builder.build_kernel(kernel)
    
    # Generate the code
    # codegen_kernel returns the source code for the specific kernel
    source = warp._src.codegen.codegen_kernel(kernel, device=device, options=options)
    
    return source

if __name__ == "__main__":
    # Simple test
    wp.init()

    @wp.kernel
    def test_kernel(x: wp.array(dtype=float)):
        tid = wp.tid()
        x[tid] = x[tid] * 2.0

    ir = get_kernel_ir(test_kernel)
    print("Extracted IR:")
    print(ir[:500] + "...") # Print first 500 chars

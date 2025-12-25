import warp as wp
import warp._src.context
import inspect
import textwrap
import uuid

def extract_ir(kernel_func):
    """
    Extracts the Intermediate Representation (IR) and Python source from a Warp kernel.

    Args:
        kernel_func: The warp kernel function (decorated with @wp.kernel).

    Returns:
        dict: A dictionary containing:
            - 'name': Kernel name.
            - 'python_source': Original Python source code.
            - 'ir': The extracted C++-like IR (list of strings/lines).
            - 'args': Argument types (as string representation).
    """
    
    # Ensure warp is initialized
    # wp.init() checks internally if it's already initialized
    try:
        if not wp.is_initialized():
             wp.init()
    except AttributeError:
        # Fallback for older versions or if is_initialized isn't exposed
        pass

    # 1. Get Python Source
    # The kernel object has an 'adj' attribute if it has been processed, 
    # but we might be the first to process it.
    # However, @wp.kernel wrapper usually creates the .adj object eagerly or we force it.
    
    # We need to build it to ensure .adj is populated with blocks.
    module_name = f"extraction_module_{uuid.uuid4().hex[:8]}"
    module = warp._src.context.Module(module_name)
    options = {
        "enable_backward": False,
        "max_unroll": 4, # Default
        "lineinfo": False
    }
    
    builder = warp._src.context.ModuleBuilder(module, options)
    
    # This populates kernel_func.adj.blocks
    builder.build_kernel(kernel_func)
    
    adj = kernel_func.adj
    
    # Extract Python Source
    python_source = adj.source
    
    # Extract IR
    # We iterate over blocks and their forward bodies.
    ir_lines = []
    
    # Add argument struct definition if useful, but maybe just body is enough for training.
    # Let's include the body instructions.
    
    for i, block in enumerate(adj.blocks):
        ir_lines.append(f"// Block {i}")
        for line in block.body_forward:
            ir_lines.append(line.rstrip())
            
    # Extract Args info
    args_info = {}
    for arg in adj.args:
        args_info[arg.label] = str(arg.type)

    return {
        "name": kernel_func.key,
        "python_source": python_source,
        "ir": ir_lines,
        "args": args_info
    }

if __name__ == "__main__":
    # Self-test
    wp.init()
    
    @wp.kernel
    def test_k(x: wp.array(dtype=float)):
        tid = wp.tid()
        x[tid] = x[tid] + 1.0
        
    data = extract_ir(test_k)
    print(f"Kernel: {data['name']}")
    print("Python Source:")
    print(data['python_source'])
    print("\nIR:")
    print("\n".join(data['ir']))

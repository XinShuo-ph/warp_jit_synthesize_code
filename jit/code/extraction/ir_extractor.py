"""IR Extractor - Extract C++/CUDA IR from warp kernels."""
import warp as wp
import numpy as np
import os
import glob as globlib

# Initialize warp once
wp.init()


def _get_cache_dir():
    """Get the warp cache directory."""
    return os.path.join(os.path.expanduser("~"), ".cache", "warp", wp.__version__)


def _force_compile(kernel, device: str = "cpu"):
    """Force kernel compilation by launching with minimal data."""
    args = []
    for arg in kernel.adj.args:
        arg_type = arg.type
        if hasattr(arg_type, 'dtype'):
            arr = wp.zeros(1, dtype=arg_type.dtype, device=device)
            args.append(arr)
        else:
            args.append(arg_type())
    
    try:
        wp.launch(kernel, dim=1, inputs=args, device=device)
        wp.synchronize_device(wp.get_device(device))
    except Exception:
        pass


def extract_ir(kernel, device: str = "cpu") -> str:
    """Extract the generated IR (C++/CUDA code) from a warp kernel.
    
    Args:
        kernel: A warp kernel (decorated with @wp.kernel)
        device: Target device - "cpu" or "cuda"
        
    Returns:
        Generated C++/CUDA source code as string
    """
    # Force compilation first
    _force_compile(kernel, device)
    
    # Find the cache file
    cache_dir = _get_cache_dir()
    module = kernel.module
    
    # Get the module hash from execs
    exec_id = module.cpu_exec_id if device == "cpu" else module.cuda_exec_id
    exec_key = (None, exec_id)
    
    if exec_key in module.execs:
        exec_obj = module.execs[exec_key]
        module_hash = exec_obj.module_hash.hex()
        
        # Look for the source file
        cache_subdir = os.path.join(cache_dir, f"wp_{module.name}_{module_hash[:7]}")
        ext = ".cpp" if device == "cpu" else ".cu"
        
        for fname in os.listdir(cache_subdir) if os.path.isdir(cache_subdir) else []:
            if fname.endswith(ext):
                with open(os.path.join(cache_subdir, fname), 'r') as f:
                    return f.read()
    
    return ""


def get_kernel_source(kernel) -> str:
    """Get the original Python source code of a kernel.
    
    Args:
        kernel: A warp kernel
        
    Returns:
        Python source code as string
    """
    return kernel.adj.source


def extract_pair(kernel, device: str = "cpu") -> dict:
    """Extract a Python→IR pair from a kernel.
    
    Args:
        kernel: A warp kernel
        device: Target device
        
    Returns:
        Dictionary with 'python_source' and 'ir_code' keys
    """
    return {
        "python_source": get_kernel_source(kernel),
        "ir_code": extract_ir(kernel, device),
        "kernel_name": kernel.adj.fun_name,
        "device": device,
    }


if __name__ == "__main__":
    # Test the extractor
    @wp.kernel
    def test_add(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
        i = wp.tid()
        c[i] = a[i] + b[i]
    
    print("=== Python Source ===")
    print(get_kernel_source(test_add))
    
    print("\n=== Generated IR ===")
    ir = extract_ir(test_add, "cpu")
    print(ir[:1500] if len(ir) > 1500 else ir)
    print(f"\n... ({len(ir)} chars total)")

"""IR Extractor - Extracts generated HLO code from JAX kernels."""
from dataclasses import dataclass
from typing import Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np
import tempfile
import importlib.util
import os
import sys
import inspect

@dataclass
class ExtractedIR:
    """Container for extracted IR from a jax kernel."""
    kernel_name: str
    python_source: str
    hlo_fwd: str  # Forward pass HLO
    hlo_bwd: Optional[str] = None  # Backward pass HLO


def create_module_from_source(source: str, module_name: str):
    """Create a Python module from source code string."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(source)
        temp_path = f.name
    
    try:
        spec = importlib.util.spec_from_file_location(module_name, temp_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module, temp_path
    except Exception as e:
        os.unlink(temp_path)
        raise e


def extract_ir(kernel_source: str, kernel_name: str, module_id: int = 0) -> ExtractedIR:
    """
    Extract HLO from a JAX kernel source.
    """
    module_name = f"synth_module_jax_{module_id}"
    module = None
    temp_path = None
    
    try:
        module, temp_path = create_module_from_source(kernel_source, module_name)
        
        if not hasattr(module, kernel_name):
             # Try to find any function if name mismatch (though regex should handle it)
             funcs = [n for n, o in inspect.getmembers(module, inspect.isfunction) if o.__module__ == module_name]
             if funcs:
                 kernel_name = funcs[0]
             else:
                 raise ValueError(f"Kernel {kernel_name} not found in generated source")

        func = getattr(module, kernel_name)
        
        # Determine args
        sig = inspect.signature(func)
        args = []
        N = 128
        
        for param in sig.parameters:
            name = param
            if name in ['alpha', 'scale', 'threshold', 't1', 't2', 'const1', 'const2']:
                args.append(0.5) # float scalar
            elif name in ['n']:
                args.append(N) # int scalar
            else:
                # Default to array
                args.append(jnp.ones((N,), dtype=jnp.float32))
        
        # Compile Forward
        lowered_fwd = jax.jit(func).lower(*args)
        hlo_fwd = lowered_fwd.as_text()
        
        # Compile Backward
        hlo_bwd = None
        try:
            # Check output shape
            out_shape_struct = jax.eval_shape(func, *args)
            # handle tuple output or single output
            if isinstance(out_shape_struct, tuple):
                 # For multiple outputs, backward is complex, skip for now or sum them
                 pass 
            else:
                out_shape = out_shape_struct.shape
                
                if len(out_shape) == 0:
                    # Scalar output
                    grad_func = jax.grad(func)
                    lowered_bwd = jax.jit(grad_func).lower(*args)
                    hlo_bwd = lowered_bwd.as_text()
                else:
                    # Array output - use vjp with dummy cotangents
                    cotangents = jnp.ones(out_shape, dtype=jnp.float32)
                    
                    def bwd_wrapper(cotangents, *primals):
                        # standard vector-jacobian product: grad(sum(f(x) * v))
                        loss = lambda *a: jnp.sum(func(*a) * cotangents)
                        return jax.grad(loss)(*primals)
                    
                    lowered_bwd = jax.jit(lambda *p: bwd_wrapper(cotangents, *p)).lower(*args)
                    hlo_bwd = lowered_bwd.as_text()
                    
        except Exception as e:
            hlo_bwd = f"Error extracting backward: {e}"
            
        return ExtractedIR(
            kernel_name=kernel_name,
            python_source=kernel_source,
            hlo_fwd=hlo_fwd,
            hlo_bwd=hlo_bwd
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        if module_name in sys.modules:
            del sys.modules[module_name]

if __name__ == "__main__":
    # Test
    code = """
import jax
import jax.numpy as jnp

@jax.jit
def test_kernel(a, b):
    return a + b
"""
    ir = extract_ir(code, "test_kernel")
    print("=== Forward HLO ===")
    print(ir.hlo_fwd[:500])
    print("=== Backward HLO ===")
    print(ir.hlo_bwd[:500] if ir.hlo_bwd else "None")

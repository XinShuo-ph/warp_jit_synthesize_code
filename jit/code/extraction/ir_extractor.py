"""IR Extractor - Extracts generated HLO/jaxpr code from JAX functions."""
from dataclasses import dataclass
from typing import Optional, Callable, Any
import jax
import jax.numpy as jnp
from jax import make_jaxpr
from jax._src import core as jax_core


@dataclass
class ExtractedIR:
    """Container for extracted IR from a JAX function."""
    function_name: str
    python_source: str
    jaxpr: str  # JAX intermediate representation
    hlo: str  # HLO (High-Level Optimizer) IR
    stablehlo: Optional[str] = None  # StableHLO/MLIR representation


def get_function_source(func: Callable) -> str:
    """Get the source code of a function."""
    import inspect
    try:
        return inspect.getsource(func)
    except (OSError, TypeError):
        return ""


def extract_ir(
    func: Callable,
    sample_inputs: tuple,
    enable_backward: bool = True
) -> ExtractedIR:
    """
    Extract IR (jaxpr/HLO) from a JAX function.
    
    Args:
        func: A JAX-compatible function
        sample_inputs: Sample inputs to trace the function
        enable_backward: Whether to include backward (gradient) code
        
    Returns:
        ExtractedIR containing Python source and generated IR
    """
    function_name = getattr(func, '__name__', 'anonymous')
    python_source = get_function_source(func)
    
    # Extract jaxpr (JAX intermediate representation)
    jaxpr_obj = make_jaxpr(func)(*sample_inputs)
    jaxpr_str = str(jaxpr_obj)
    
    # Extract HLO via XLA compilation
    jitted_func = jax.jit(func)
    lowered = jitted_func.lower(*sample_inputs)
    
    # Get HLO text representation
    hlo_str = lowered.as_text()
    
    # Get StableHLO/MLIR representation if available
    stablehlo_str = None
    try:
        # Try to get StableHLO representation
        stablehlo_str = lowered.as_text(dialect='stablehlo')
    except Exception:
        try:
            # Alternative: get the MLIR module
            stablehlo_str = str(lowered.compiler_ir(dialect='stablehlo'))
        except Exception:
            pass
    
    # If backward is enabled, also extract gradient function IR
    if enable_backward:
        try:
            # Create gradient function
            grad_func = jax.grad(lambda *args: jnp.sum(func(*args)))
            grad_jaxpr = make_jaxpr(grad_func)(*sample_inputs)
            
            jaxpr_str += "\n\n=== BACKWARD (GRADIENT) ===\n"
            jaxpr_str += str(grad_jaxpr)
            
            # Get HLO for gradient
            grad_lowered = jax.jit(grad_func).lower(*sample_inputs)
            hlo_str += "\n\n=== BACKWARD (GRADIENT) HLO ===\n"
            hlo_str += grad_lowered.as_text()
            
            if stablehlo_str:
                try:
                    grad_stablehlo = grad_lowered.as_text(dialect='stablehlo')
                    stablehlo_str += "\n\n=== BACKWARD (GRADIENT) STABLEHLO ===\n"
                    stablehlo_str += grad_stablehlo
                except Exception:
                    pass
                    
        except Exception as e:
            # Some functions may not be differentiable
            jaxpr_str += f"\n\n=== BACKWARD: Not available ({e}) ===\n"
    
    return ExtractedIR(
        function_name=function_name,
        python_source=python_source,
        jaxpr=jaxpr_str,
        hlo=hlo_str,
        stablehlo=stablehlo_str,
    )


def extract_ir_pair(func: Callable, sample_inputs: tuple, ir_type: str = "hlo") -> tuple[str, str]:
    """
    Extract Python→IR pair suitable for LLM training.
    
    Args:
        func: A JAX-compatible function
        sample_inputs: Sample inputs to trace the function
        ir_type: "jaxpr", "hlo", or "stablehlo"
        
    Returns:
        Tuple of (python_source, ir_code)
    """
    ir = extract_ir(func, sample_inputs)
    
    if ir_type == "jaxpr":
        return (ir.python_source, ir.jaxpr)
    elif ir_type == "stablehlo" and ir.stablehlo:
        return (ir.python_source, ir.stablehlo)
    else:
        return (ir.python_source, ir.hlo)


def extract_vmap_ir(func: Callable, sample_input: Any, batch_size: int = 32) -> ExtractedIR:
    """
    Extract IR from a vmapped (vectorized) function.
    
    Args:
        func: A JAX-compatible function that operates on single elements
        sample_input: A single sample input
        batch_size: Batch size for vectorization
        
    Returns:
        ExtractedIR with vectorized code
    """
    # Create batched version
    vmapped_func = jax.vmap(func)
    
    # Create batched sample input
    if isinstance(sample_input, tuple):
        batched_inputs = tuple(
            jnp.stack([inp] * batch_size) if hasattr(inp, 'shape') else inp
            for inp in sample_input
        )
    else:
        batched_inputs = (jnp.stack([sample_input] * batch_size),)
    
    return extract_ir(vmapped_func, batched_inputs)


if __name__ == "__main__":
    # Test with a simple function
    print("=== JAX IR Extractor Demo ===\n")
    
    def test_kernel(a, b):
        return a * 2.0 + b
    
    # Create sample inputs
    a = jnp.ones((8,), dtype=jnp.float32)
    b = jnp.ones((8,), dtype=jnp.float32)
    
    ir = extract_ir(test_kernel, (a, b))
    
    print("=== Function Name ===")
    print(ir.function_name)
    
    print("\n=== Python Source ===")
    print(ir.python_source)
    
    print("\n=== Jaxpr ===")
    print(ir.jaxpr[:1500] if len(ir.jaxpr) > 1500 else ir.jaxpr)
    
    print("\n=== HLO (first 1500 chars) ===")
    print(ir.hlo[:1500] if len(ir.hlo) > 1500 else ir.hlo)
    
    if ir.stablehlo:
        print("\n=== StableHLO available ===")
        print("Yes")
        print("\n=== StableHLO (first 1000 chars) ===")
        print(ir.stablehlo[:1000])
    else:
        print("\n=== StableHLO available ===")
        print("No")
    
    # Test vmap
    print("\n\n=== Testing vmap extraction ===")
    
    def element_func(x):
        return jnp.sin(x) + jnp.cos(x)
    
    vmap_ir = extract_vmap_ir(element_func, jnp.array(1.0))
    print(f"Vmapped function jaxpr:\n{vmap_ir.jaxpr[:800]}")

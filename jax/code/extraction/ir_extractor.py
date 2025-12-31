"""IR Extractor - Extracts HLO/XLA IR from JAX functions."""
from dataclasses import dataclass
from typing import Optional, Callable, Any, Dict
import jax
import jax.numpy as jnp
from functools import partial


@dataclass
class ExtractedIR:
    """Container for extracted IR from a JAX function."""
    kernel_name: str
    python_source: str
    hlo_text: str  # HLO text representation (forward)
    hlo_optimized: Optional[str] = None  # Optimized HLO
    hlo_backward: Optional[str] = None  # HLO for backward pass (gradient)
    stablehlo: Optional[str] = None  # StableHLO representation


def get_sample_inputs(fn: Callable, input_spec: Optional[Dict] = None) -> tuple:
    """Generate sample inputs for a function based on its signature or spec."""
    if input_spec:
        return tuple(input_spec.values())
    
    # Default: try to infer from function
    import inspect
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    
    inputs = []
    for param in params:
        # Default to (10,) float32 array
        if param in ['key', 'rng_key']:
            inputs.append(jax.random.PRNGKey(0))
        elif param in ['training', 'train']:
            inputs.append(True)
        elif param in ['n', 'num', 'count', 'size']:
            inputs.append(5)
        elif param in ['alpha', 'beta', 'gamma', 'scale', 'rate']:
            inputs.append(jnp.array(1.0))
        elif param in ['q', 'k', 'v']:
            inputs.append(jnp.ones((4, 8, 16)))  # batch, seq, dim
        elif param == 'kernel':
            inputs.append(jnp.ones((3,)))  # small 1D kernel
        else:
            inputs.append(jnp.ones((10,)))
    
    return tuple(inputs)


def extract_hlo_text(fn: Callable, inputs: tuple) -> str:
    """Extract HLO text representation from a JIT-compiled function."""
    lowered = jax.jit(fn).lower(*inputs)
    return lowered.as_text()


def extract_hlo_optimized(fn: Callable, inputs: tuple) -> str:
    """Extract optimized HLO after compilation."""
    lowered = jax.jit(fn).lower(*inputs)
    compiled = lowered.compile()
    return compiled.as_text()


def extract_stablehlo(fn: Callable, inputs: tuple) -> Optional[str]:
    """Extract StableHLO representation if available."""
    try:
        lowered = jax.jit(fn).lower(*inputs)
        # Try to get StableHLO module
        if hasattr(lowered, 'compiler_ir'):
            stablehlo_module = lowered.compiler_ir(dialect='stablehlo')
            return str(stablehlo_module)
        return None
    except Exception:
        return None


def make_grad_fn(fn: Callable, inputs: tuple) -> Optional[Callable]:
    """Create a gradient function for the given function."""
    try:
        # Determine which args to differentiate (must be float arrays)
        argnums = []
        for i, inp in enumerate(inputs):
            if isinstance(inp, jnp.ndarray) and jnp.issubdtype(inp.dtype, jnp.floating):
                argnums.append(i)
        
        if not argnums:
            return None
        
        # Handle multiple argnums
        if len(argnums) == 1:
            argnums = argnums[0]
        else:
            argnums = tuple(argnums)
        
        # Create a scalar loss function for grad
        def loss_fn(*args):
            result = fn(*args)
            if isinstance(result, jnp.ndarray):
                return jnp.sum(result)
            return result
        
        return jax.grad(loss_fn, argnums=argnums)
    except Exception:
        return None


def extract_backward_hlo(fn: Callable, inputs: tuple) -> Optional[str]:
    """Extract HLO for the backward pass (gradient computation)."""
    grad_fn = make_grad_fn(fn, inputs)
    if grad_fn is None:
        return None
    
    try:
        lowered = jax.jit(grad_fn).lower(*inputs)
        return lowered.as_text()
    except Exception:
        return None


def extract_ir(
    fn: Callable,
    inputs: Optional[tuple] = None,
    input_spec: Optional[Dict] = None,
    enable_backward: bool = True
) -> ExtractedIR:
    """
    Extract IR (HLO/XLA) from a JAX function.
    
    Args:
        fn: A JAX-compatible function
        inputs: Optional tuple of input values
        input_spec: Optional dict of input specifications
        enable_backward: Whether to include backward (gradient) HLO
        
    Returns:
        ExtractedIR containing Python source and generated HLO
    """
    import inspect
    
    # Get function name and source
    kernel_name = fn.__name__ if hasattr(fn, '__name__') else str(fn)
    try:
        python_source = inspect.getsource(fn)
    except (OSError, TypeError):
        python_source = f"# Source not available for {kernel_name}"
    
    # Generate sample inputs if not provided
    if inputs is None:
        inputs = get_sample_inputs(fn, input_spec)
    
    # Extract forward HLO
    hlo_text = extract_hlo_text(fn, inputs)
    
    # Extract optimized HLO
    try:
        hlo_optimized = extract_hlo_optimized(fn, inputs)
    except Exception:
        hlo_optimized = None
    
    # Extract backward HLO
    hlo_backward = None
    if enable_backward:
        hlo_backward = extract_backward_hlo(fn, inputs)
    
    # Extract StableHLO
    stablehlo = extract_stablehlo(fn, inputs)
    
    return ExtractedIR(
        kernel_name=kernel_name,
        python_source=python_source,
        hlo_text=hlo_text,
        hlo_optimized=hlo_optimized,
        hlo_backward=hlo_backward,
        stablehlo=stablehlo,
    )


def extract_ir_pair(fn: Callable, inputs: Optional[tuple] = None, mode: str = "hlo") -> tuple[str, str]:
    """
    Extract Python→HLO pair suitable for LLM training.
    
    Args:
        fn: A JAX function
        inputs: Optional input values
        mode: "hlo", "optimized", or "stablehlo"
        
    Returns:
        Tuple of (python_source, ir_code)
    """
    ir = extract_ir(fn, inputs)
    
    if mode == "optimized" and ir.hlo_optimized:
        return (ir.python_source, ir.hlo_optimized)
    elif mode == "stablehlo" and ir.stablehlo:
        return (ir.python_source, ir.stablehlo)
    return (ir.python_source, ir.hlo_text)


if __name__ == "__main__":
    # Test with a simple function
    print("=== Testing JAX IR Extractor ===\n")
    
    def test_kernel(a, b):
        """Simple test kernel."""
        return a * 2.0 + b
    
    # Generate sample inputs
    inputs = (jnp.ones((10,)), jnp.ones((10,)))
    
    ir = extract_ir(test_kernel, inputs)
    print("=== Kernel Name ===")
    print(ir.kernel_name)
    print("\n=== Python Source ===")
    print(ir.python_source)
    print("\n=== HLO Text (first 1500 chars) ===")
    print(ir.hlo_text[:1500] if len(ir.hlo_text) > 1500 else ir.hlo_text)
    print("\n=== Optimized HLO available ===")
    print("Yes" if ir.hlo_optimized else "No")
    print("\n=== Backward HLO available ===")
    print("Yes" if ir.hlo_backward else "No")
    if ir.hlo_backward:
        print("\n=== Backward HLO (first 1500 chars) ===")
        print(ir.hlo_backward[:1500] if len(ir.hlo_backward) > 1500 else ir.hlo_backward)
    print("\n=== StableHLO available ===")
    print("Yes" if ir.stablehlo else "No")
    
    # Test with a more complex function
    print("\n\n=== Testing with reduction kernel ===")
    
    def reduction_kernel(a):
        """Reduction with mean."""
        return jnp.mean(a)
    
    ir2 = extract_ir(reduction_kernel, (jnp.ones((100,)),))
    print("\n=== Kernel Name ===")
    print(ir2.kernel_name)
    print("\n=== HLO Text ===")
    print(ir2.hlo_text[:1000] if len(ir2.hlo_text) > 1000 else ir2.hlo_text)
    print("\n=== Backward HLO ===")
    if ir2.hlo_backward:
        print(ir2.hlo_backward[:1000] if len(ir2.hlo_backward) > 1000 else ir2.hlo_backward)

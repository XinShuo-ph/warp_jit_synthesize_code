import jax
import jax.numpy as jnp
from typing import Callable, Any, Optional

def get_ir(func: Callable, *args: Any, backend: Optional[str] = None) -> str:
    """
    Extracts the StableHLO IR for a given JAX function and arguments.
    
    Args:
        func: The python function to compile.
        *args: Example arguments to trace the function with.
        backend: 'cpu', 'gpu', or 'tpu'. Defaults to JAX's default.
        
    Returns:
        String containing the HLO text.
    """
    try:
        # Lower the function using JAX's JIT compiler
        lowered = jax.jit(func, backend=backend).lower(*args)
        
        # Return the textual representation (usually StableHLO)
        return lowered.as_text()
    except Exception as e:
        raise RuntimeError(f"Failed to extract IR: {e}")

def get_executable_text(func: Callable, *args: Any, backend: Optional[str] = None) -> str:
    """
    Extracts the compiled executable text (e.g. assembly) if available.
    """
    try:
        lowered = jax.jit(func, backend=backend).lower(*args)
        compiled = lowered.compile()
        return compiled.as_text()
    except Exception as e:
        return f"Could not get executable text: {e}"

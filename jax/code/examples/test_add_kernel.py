"""Example: Simple addition kernel with JAX."""
import jax
import jax.numpy as jnp
import sys
from pathlib import Path

# Add extraction module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))

from ir_extractor import extract_ir


def add_kernel(a, b):
    """Simple element-wise addition."""
    return a + b


if __name__ == "__main__":
    print("=== JAX Add Kernel Example ===\n")
    
    # Create sample inputs
    a = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = jnp.array([5.0, 4.0, 3.0, 2.0, 1.0])
    
    # Run the kernel
    result = add_kernel(a, b)
    print(f"Input a: {a}")
    print(f"Input b: {b}")
    print(f"Result:  {result}")
    
    # Extract HLO
    ir = extract_ir(add_kernel, (a, b))
    
    print("\n=== HLO Forward ===")
    print(ir.hlo_text)
    
    print("\n=== HLO Backward ===")
    if ir.hlo_backward:
        print(ir.hlo_backward)
    else:
        print("No backward pass available")

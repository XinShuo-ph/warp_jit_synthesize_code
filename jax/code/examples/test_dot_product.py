"""Example: Dot product kernel with JAX."""
import jax
import jax.numpy as jnp
import sys
from pathlib import Path

# Add extraction module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))

from ir_extractor import extract_ir


def dot_product(a, b):
    """Compute dot product of two vectors."""
    return jnp.sum(a * b)


if __name__ == "__main__":
    print("=== JAX Dot Product Example ===\n")
    
    # Create sample inputs
    a = jnp.array([1.0, 2.0, 3.0])
    b = jnp.array([4.0, 5.0, 6.0])
    
    # Run the kernel
    result = dot_product(a, b)
    print(f"Input a: {a}")
    print(f"Input b: {b}")
    print(f"Dot product: {result}")
    print(f"Expected: {1*4 + 2*5 + 3*6} = 32.0")
    
    # Extract HLO
    ir = extract_ir(dot_product, (a, b))
    
    print("\n=== HLO Forward ===")
    print(ir.hlo_text)
    
    print("\n=== HLO Backward ===")
    if ir.hlo_backward:
        print(ir.hlo_backward[:1500] if len(ir.hlo_backward) > 1500 else ir.hlo_backward)

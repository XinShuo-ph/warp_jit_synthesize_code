"""Example: Dot product function with JAX IR extraction."""
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))
sys.path.insert(0, str(Path(__file__).parent.parent / "synthesis"))

import jax
import jax.numpy as jnp
from ir_extractor import extract_ir


def dot_product(a, b):
    """Compute dot product of two vectors."""
    return jnp.dot(a, b)


def main():
    print("=== JAX Dot Product Example ===\n")
    
    # Create sample inputs
    a = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    b = jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
    
    # Test the function
    result = dot_product(a, b)
    print(f"Input a: {a}")
    print(f"Input b: {b}")
    print(f"Result: {result}")
    print(f"Expected: 10.0 (1+2+3+4)")
    
    # Extract IR
    print("\n=== Extracting IR ===\n")
    ir = extract_ir(dot_product, (a, b))
    
    print("--- Jaxpr ---")
    print(ir.jaxpr)
    
    print("\n--- HLO ---")
    print(ir.hlo[:1000] if len(ir.hlo) > 1000 else ir.hlo)
    
    # Test gradient
    print("\n=== Gradient Test ===")
    grad_fn = jax.grad(lambda a, b: dot_product(a, b), argnums=0)
    grad_a = grad_fn(a, b)
    print(f"Gradient w.r.t. a: {grad_a}")
    print(f"Expected: {b} (derivative of a·b w.r.t. a is b)")


if __name__ == "__main__":
    main()

"""Example: SAXPY (Scalar A*X Plus Y) function with JAX IR extraction."""
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))
sys.path.insert(0, str(Path(__file__).parent.parent / "synthesis"))

import jax
import jax.numpy as jnp
from ir_extractor import extract_ir


def saxpy(alpha, x, y):
    """SAXPY: alpha * x + y (Single-precision A*X Plus Y)."""
    return alpha * x + y


def main():
    print("=== JAX SAXPY Example ===\n")
    
    # Create sample inputs
    alpha = jnp.array(2.0, dtype=jnp.float32)
    x = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    y = jnp.array([10.0, 20.0, 30.0, 40.0], dtype=jnp.float32)
    
    # Test the function
    result = saxpy(alpha, x, y)
    print(f"alpha: {alpha}")
    print(f"x: {x}")
    print(f"y: {y}")
    print(f"Result (alpha*x + y): {result}")
    print(f"Expected: [12. 24. 36. 48.]")
    
    # Extract IR
    print("\n=== Extracting IR ===\n")
    ir = extract_ir(saxpy, (alpha, x, y))
    
    print("--- Jaxpr ---")
    print(ir.jaxpr)
    
    print("\n--- HLO (first 1200 chars) ---")
    print(ir.hlo[:1200] if len(ir.hlo) > 1200 else ir.hlo)
    
    # Test gradient
    print("\n=== Gradient Test ===")
    # Gradient w.r.t. x (should be alpha broadcast)
    grad_x_fn = jax.grad(lambda a, x, y: jnp.sum(saxpy(a, x, y)), argnums=1)
    grad_x = grad_x_fn(alpha, x, y)
    print(f"Gradient w.r.t. x: {grad_x}")
    print(f"Expected: [2. 2. 2. 2.] (alpha broadcast)")
    
    # Gradient w.r.t. alpha (should be sum(x))
    grad_alpha_fn = jax.grad(lambda a, x, y: jnp.sum(saxpy(a, x, y)), argnums=0)
    grad_alpha = grad_alpha_fn(alpha, x, y)
    print(f"Gradient w.r.t. alpha: {grad_alpha}")
    print(f"Expected: 10.0 (sum of x)")


if __name__ == "__main__":
    main()

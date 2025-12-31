"""Example: SAXPY (Single-precision A*X Plus Y) kernel with JAX."""
import jax
import jax.numpy as jnp
import sys
from pathlib import Path

# Add extraction module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))

from ir_extractor import extract_ir


def saxpy(alpha, x, y):
    """SAXPY: result = alpha * x + y"""
    return alpha * x + y


if __name__ == "__main__":
    print("=== JAX SAXPY Example ===\n")
    
    # Create sample inputs
    alpha = jnp.array(2.0)
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    y = jnp.array([0.5, 0.5, 0.5, 0.5])
    
    # Run the kernel
    result = saxpy(alpha, x, y)
    print(f"alpha: {alpha}")
    print(f"x: {x}")
    print(f"y: {y}")
    print(f"Result (alpha * x + y): {result}")
    
    # Extract HLO
    ir = extract_ir(saxpy, (alpha, x, y))
    
    print("\n=== HLO Forward ===")
    print(ir.hlo_text)
    
    print("\n=== HLO Backward ===")
    if ir.hlo_backward:
        print(ir.hlo_backward[:2000] if len(ir.hlo_backward) > 2000 else ir.hlo_backward)

"""Example: Simple addition function with JAX IR extraction."""
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))
sys.path.insert(0, str(Path(__file__).parent.parent / "synthesis"))

import jax
import jax.numpy as jnp
from ir_extractor import extract_ir


def add_arrays(a, b):
    """Simple elementwise addition."""
    return a + b


def main():
    print("=== JAX Add Arrays Example ===\n")
    
    # Create sample inputs
    a = jnp.ones((64,), dtype=jnp.float32)
    b = jnp.ones((64,), dtype=jnp.float32) * 2.0
    
    # Test the function
    result = add_arrays(a, b)
    print(f"Input a: {a[:5]}...")
    print(f"Input b: {b[:5]}...")
    print(f"Result: {result[:5]}...")
    print(f"Expected: [3. 3. 3. 3. 3.]")
    
    # Extract IR
    print("\n=== Extracting IR ===\n")
    ir = extract_ir(add_arrays, (a, b))
    
    print("--- Jaxpr ---")
    print(ir.jaxpr[:500] if len(ir.jaxpr) > 500 else ir.jaxpr)
    
    print("\n--- HLO (first 800 chars) ---")
    print(ir.hlo[:800] if len(ir.hlo) > 800 else ir.hlo)
    
    print("\n--- StableHLO available ---")
    print("Yes" if ir.stablehlo else "No")


if __name__ == "__main__":
    main()

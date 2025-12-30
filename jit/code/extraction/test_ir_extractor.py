"""Test IR extraction with JAX."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ir_extractor import extract_ir
import jax.numpy as jnp


def simple_test_kernel(a, b):
    """Simple test kernel for IR extraction."""
    return a * 2.0 + b


if __name__ == "__main__":
    # Create sample inputs
    n = 10
    sample_a = jnp.ones(n, dtype=jnp.float32)
    sample_b = jnp.zeros(n, dtype=jnp.float32)
    
    print("Testing IR extraction...")
    ir = extract_ir(simple_test_kernel, (sample_a, sample_b))
    
    print("\n=== Kernel Name ===")
    print(ir.kernel_name)
    print("\n=== Python Source ===")
    print(ir.python_source)
    print("\n=== HLO Code (first 1000 chars) ===")
    print(ir.hlo_text[:1000])
    print("\n=== Optimized HLO available ===")
    print("Yes" if ir.optimized_hlo else "No")
    
    print("\nTest completed successfully!")

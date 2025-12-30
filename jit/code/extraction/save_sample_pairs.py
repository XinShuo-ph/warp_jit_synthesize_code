"""
Save sample Python→IR pairs to JSON files
"""

import jax.numpy as jnp
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from extraction.ir_extractor import IRExtractor


def save_samples():
    """Generate and save sample Python→IR pairs."""
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    extractor = IRExtractor(dialect='stablehlo')
    
    # Sample 1: Simple addition
    def add_vectors(x, y):
        """Add two vectors element-wise."""
        return x + y
    
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    result = extractor.extract_with_metadata(add_vectors, x, y)
    
    filepath = os.path.join(data_dir, 'sample_01_add.json')
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {filepath}")
    
    # Sample 2: Matrix multiplication
    def matmul(A, B):
        """Matrix multiplication."""
        return jnp.dot(A, B)
    
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    B = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    result = extractor.extract_with_metadata(matmul, A, B)
    
    filepath = os.path.join(data_dir, 'sample_02_matmul.json')
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {filepath}")
    
    # Sample 3: Math operations
    def math_ops(x):
        """Combined math operations."""
        return jnp.tanh(jnp.sin(x) + jnp.exp(x))
    
    x = jnp.array([1.0, 2.0, 3.0])
    result = extractor.extract_with_metadata(math_ops, x)
    
    filepath = os.path.join(data_dir, 'sample_03_math.json')
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {filepath}")
    
    # Sample 4: Conditional
    def conditional(x):
        """Conditional operation using where."""
        return jnp.where(x > 0, x ** 2, -x)
    
    x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = extractor.extract_with_metadata(conditional, x)
    
    filepath = os.path.join(data_dir, 'sample_04_conditional.json')
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {filepath}")
    
    # Sample 5: Reduction
    def sum_squares(x):
        """Sum of squares."""
        return jnp.sum(x ** 2)
    
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    result = extractor.extract_with_metadata(sum_squares, x)
    
    filepath = os.path.join(data_dir, 'sample_05_reduction.json')
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {filepath}")
    
    # Sample 6: Linear layer
    def linear_layer(W, x, b):
        """Linear transformation: Wx + b"""
        return jnp.dot(W, x) + b
    
    W = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x = jnp.array([1.0, 2.0, 3.0])
    b = jnp.array([0.5, 1.5])
    result = extractor.extract_with_metadata(linear_layer, W, x, b)
    
    filepath = os.path.join(data_dir, 'sample_06_linear.json')
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {filepath}")
    
    print(f"\nTotal samples saved: 6")


if __name__ == "__main__":
    save_samples()

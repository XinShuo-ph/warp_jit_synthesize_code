"""Tests for JAX IR Extractor."""
import jax
import jax.numpy as jnp
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from ir_extractor import extract_ir, extract_ir_pair, get_sample_inputs


def test_simple_elementwise():
    """Test simple elementwise function."""
    def add_fn(a, b):
        return a + b
    
    inputs = (jnp.ones((10,)), jnp.ones((10,)))
    ir = extract_ir(add_fn, inputs)
    
    assert ir.kernel_name == "add_fn"
    assert "add" in ir.hlo_text.lower() or "+" in ir.hlo_text
    assert ir.hlo_backward is not None
    print("✓ test_simple_elementwise passed")


def test_reduction():
    """Test reduction function."""
    def mean_fn(a):
        return jnp.mean(a)
    
    inputs = (jnp.ones((100,)),)
    ir = extract_ir(mean_fn, inputs)
    
    assert ir.kernel_name == "mean_fn"
    assert "reduce" in ir.hlo_text.lower()
    print("✓ test_reduction passed")


def test_matmul():
    """Test matrix multiplication."""
    def matmul_fn(a, b):
        return jnp.matmul(a, b)
    
    inputs = (jnp.ones((8, 8)), jnp.ones((8, 8)))
    ir = extract_ir(matmul_fn, inputs)
    
    assert ir.kernel_name == "matmul_fn"
    assert "dot" in ir.hlo_text.lower()
    assert ir.hlo_backward is not None
    print("✓ test_matmul passed")


def test_conditional():
    """Test conditional function."""
    def cond_fn(a):
        return jnp.where(a > 0, a * 2, a * 0.5)
    
    inputs = (jnp.array([-1.0, 0.0, 1.0]),)
    ir = extract_ir(cond_fn, inputs)
    
    assert ir.kernel_name == "cond_fn"
    assert "select" in ir.hlo_text.lower()
    print("✓ test_conditional passed")


def test_backward_pass():
    """Test that backward pass is correctly extracted."""
    def loss_fn(x, y):
        return jnp.sum((x - y) ** 2)
    
    inputs = (jnp.ones((10,)), jnp.zeros((10,)))
    ir = extract_ir(loss_fn, inputs)
    
    assert ir.hlo_backward is not None
    # Backward pass should have gradients
    assert len(ir.hlo_backward) > 0
    print("✓ test_backward_pass passed")


def test_extract_pair():
    """Test extract_ir_pair function."""
    def fn(a):
        return jnp.sqrt(a)
    
    inputs = (jnp.ones((10,)),)
    python_src, hlo = extract_ir_pair(fn, inputs, mode="hlo")
    
    assert "sqrt" in python_src
    assert len(hlo) > 0
    print("✓ test_extract_pair passed")


def test_sample_inputs():
    """Test automatic input generation."""
    def fn(a, alpha, n):
        pass
    
    inputs = get_sample_inputs(fn)
    assert len(inputs) == 3
    assert isinstance(inputs[0], jnp.ndarray)
    assert isinstance(inputs[1], jnp.ndarray)
    assert isinstance(inputs[2], int)
    print("✓ test_sample_inputs passed")


def run_all_tests():
    """Run all tests."""
    print("=== Running IR Extractor Tests ===\n")
    
    test_simple_elementwise()
    test_reduction()
    test_matmul()
    test_conditional()
    test_backward_pass()
    test_extract_pair()
    test_sample_inputs()
    
    print("\n=== All tests passed! ===")


if __name__ == "__main__":
    run_all_tests()

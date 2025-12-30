"""Tests for JAX IR Extractor."""
import jax
import jax.numpy as jnp
from jax_ir_extractor import extract_ir, extract_ir_with_grad, ExtractedIR


def test_simple_function():
    """Test extraction from a simple function."""
    @jax.jit
    def add_arrays(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return a + b
    
    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (8,))
    b = jax.random.normal(jax.random.PRNGKey(1), (8,))
    
    ir = extract_ir(add_arrays, (a, b))
    
    assert isinstance(ir, ExtractedIR)
    assert ir.function_name == "add_arrays"
    assert "add_arrays" in ir.python_source
    assert "stablehlo.add" in ir.hlo_text or "add" in ir.hlo_text
    print("✓ test_simple_function passed")


def test_unary_function():
    """Test extraction from a unary math function."""
    @jax.jit
    def sqrt_array(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sqrt(x)
    
    x = jnp.abs(jax.random.normal(jax.random.PRNGKey(0), (16,))) + 0.1
    
    ir = extract_ir(sqrt_array, (x,))
    
    assert ir.function_name == "sqrt_array"
    assert "sqrt" in ir.hlo_text.lower()
    print("✓ test_unary_function passed")


def test_gradient_extraction():
    """Test extraction of gradient functions."""
    def loss_fn(x: jnp.ndarray) -> float:
        return jnp.sum(x ** 2)
    
    x = jax.random.normal(jax.random.PRNGKey(0), (8,))
    
    forward_ir, grad_ir = extract_ir_with_grad(loss_fn, (x,))
    
    assert forward_ir.function_name == "loss_fn"
    assert "grad" in grad_ir.function_name
    assert grad_ir.hlo_text is not None
    assert len(grad_ir.hlo_text) > 0
    print("✓ test_gradient_extraction passed")


def test_reduction():
    """Test extraction from reduction operations."""
    @jax.jit
    def sum_array(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(x)
    
    x = jax.random.normal(jax.random.PRNGKey(0), (16,))
    
    ir = extract_ir(sum_array, (x,))
    
    assert "reduce" in ir.hlo_text.lower()
    print("✓ test_reduction passed")


def test_conditional():
    """Test extraction from conditional operations."""
    @jax.jit
    def conditional_fn(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(x > 0, x * 2, x * 0.5)
    
    x = jax.random.normal(jax.random.PRNGKey(0), (16,))
    
    ir = extract_ir(conditional_fn, (x,))
    
    assert "select" in ir.hlo_text.lower() or "where" in ir.python_source
    print("✓ test_conditional passed")


def test_loop():
    """Test extraction from loop operations."""
    @jax.jit
    def loop_fn(x: jnp.ndarray, n: int) -> jnp.ndarray:
        def body(i, acc):
            return acc + x
        return jax.lax.fori_loop(0, n, body, jnp.zeros_like(x))
    
    x = jax.random.normal(jax.random.PRNGKey(0), (8,))
    
    ir = extract_ir(loop_fn, (x, 5))
    
    assert "while" in ir.hlo_text.lower() or "loop" in ir.python_source
    print("✓ test_loop passed")


def test_matmul():
    """Test extraction from matrix multiplication."""
    @jax.jit
    def matmul_fn(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return jnp.matmul(a, b)
    
    a = jax.random.normal(jax.random.PRNGKey(0), (4, 4))
    b = jax.random.normal(jax.random.PRNGKey(1), (4, 4))
    
    ir = extract_ir(matmul_fn, (a, b))
    
    assert "dot" in ir.hlo_text.lower()
    print("✓ test_matmul passed")


if __name__ == "__main__":
    print("=== Running JAX IR Extractor Tests ===\n")
    
    test_simple_function()
    test_unary_function()
    test_gradient_extraction()
    test_reduction()
    test_conditional()
    test_loop()
    test_matmul()
    
    print("\n=== All tests passed! ===")

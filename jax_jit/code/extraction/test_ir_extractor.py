"""Test IR extractor with various JAX functions."""
import jax
import jax.numpy as jnp
from jax import jit
import sys
sys.path.insert(0, '/workspace/jax_jit/code/extraction')
from ir_extractor import extract_ir, extract_with_grad, save_ir_pair


# Test functions
def simple_add(x, y):
    """Add two arrays."""
    return x + y


def saxpy(alpha, x, y):
    """Compute alpha * x + y."""
    return alpha * x + y


def dot_product(x, y):
    """Compute dot product."""
    return jnp.dot(x, y)


def vector_norm(x):
    """Compute L2 norm."""
    return jnp.sqrt(jnp.sum(x * x))


@jit
def conditional_scale(x, threshold=0.0):
    """Scale values based on threshold."""
    return jnp.where(x > threshold, x * 2.0, x * 0.5)


@jit
def matrix_multiply(A, B):
    """Matrix multiplication."""
    return jnp.matmul(A, B)


def reduction_sum(x):
    """Sum reduction."""
    return jnp.sum(x)


def test_basic_extraction():
    """Test basic IR extraction."""
    print("=" * 80)
    print("TEST 1: Basic IR Extraction")
    print("=" * 80)
    
    x = jnp.array([1., 2., 3., 4.])
    y = jnp.array([5., 6., 7., 8.])
    
    pair = extract_ir(simple_add, x, y)
    
    print(f"\nFunction: {pair.function_name}")
    print(f"Cost Analysis: {pair.cost_analysis}")
    print(f"\nPython Source ({len(pair.python_source.splitlines())} lines):")
    print(pair.python_source)
    print(f"\nStableHLO IR ({len(pair.stablehlo_ir.splitlines())} lines):")
    print(pair.stablehlo_ir[:500] + "..." if len(pair.stablehlo_ir) > 500 else pair.stablehlo_ir)
    
    assert pair.function_name == "simple_add"
    assert "def simple_add" in pair.python_source
    assert "stablehlo" in pair.stablehlo_ir
    assert len(pair.stablehlo_ir) > 0
    print("\n✓ Test passed")


def test_scalar_and_vector():
    """Test with scalar and vector inputs."""
    print("\n" + "=" * 80)
    print("TEST 2: Scalar and Vector Inputs (saxpy)")
    print("=" * 80)
    
    alpha = 2.5
    x = jnp.array([1., 2., 3.])
    y = jnp.array([4., 5., 6.])
    
    pair = extract_ir(saxpy, alpha, x, y)
    
    print(f"\nFunction: {pair.function_name}")
    print(f"Cost Analysis: {pair.cost_analysis}")
    print(f"\nStableHLO IR ({len(pair.stablehlo_ir.splitlines())} lines):")
    print(pair.stablehlo_ir)
    
    assert "broadcast" in pair.stablehlo_ir.lower() or "broadcast_in_dim" in pair.stablehlo_ir
    assert "multiply" in pair.stablehlo_ir.lower()
    assert "add" in pair.stablehlo_ir.lower()
    print("\n✓ Test passed")


def test_reduction():
    """Test reduction operations."""
    print("\n" + "=" * 80)
    print("TEST 3: Reduction Operations")
    print("=" * 80)
    
    x = jnp.array([1., 2., 3., 4., 5.])
    
    pair = extract_ir(reduction_sum, x)
    
    print(f"\nFunction: {pair.function_name}")
    print(f"Cost Analysis: {pair.cost_analysis}")
    print(f"\nStableHLO IR:")
    print(pair.stablehlo_ir)
    
    assert "reduce" in pair.stablehlo_ir.lower()
    print("\n✓ Test passed")


def test_matrix_ops():
    """Test matrix operations."""
    print("\n" + "=" * 80)
    print("TEST 4: Matrix Operations")
    print("=" * 80)
    
    A = jnp.array([[1., 2.], [3., 4.]])
    B = jnp.array([[5., 6.], [7., 8.]])
    
    pair = extract_ir(matrix_multiply, A, B)
    
    print(f"\nFunction: {pair.function_name}")
    print(f"Cost Analysis: {pair.cost_analysis}")
    print(f"\nStableHLO IR:")
    print(pair.stablehlo_ir)
    
    assert "dot" in pair.stablehlo_ir.lower() or "matmul" in pair.stablehlo_ir.lower()
    print("\n✓ Test passed")


def test_conditional():
    """Test conditional operations."""
    print("\n" + "=" * 80)
    print("TEST 5: Conditional Operations")
    print("=" * 80)
    
    x = jnp.array([-2., -1., 0., 1., 2.])
    
    pair = extract_ir(conditional_scale, x)
    
    print(f"\nFunction: {pair.function_name}")
    print(f"Cost Analysis: {pair.cost_analysis}")
    print(f"\nStableHLO IR:")
    print(pair.stablehlo_ir)
    
    assert "compare" in pair.stablehlo_ir.lower() or "select" in pair.stablehlo_ir.lower()
    print("\n✓ Test passed")


def test_gradient_extraction():
    """Test gradient IR extraction."""
    print("\n" + "=" * 80)
    print("TEST 6: Gradient Extraction")
    print("=" * 80)
    
    def loss_fn(x):
        return jnp.sum(x * x)
    
    x = jnp.array([1., 2., 3.])
    
    fwd_pair, bwd_pair = extract_with_grad(loss_fn, x)
    
    print(f"\nForward Function: {fwd_pair.function_name}")
    print(f"Forward Cost: {fwd_pair.cost_analysis}")
    print(f"Forward IR lines: {len(fwd_pair.stablehlo_ir.splitlines())}")
    
    print(f"\nBackward Function: {bwd_pair.function_name}")
    print(f"Backward Cost: {bwd_pair.cost_analysis}")
    print(f"Backward IR lines: {len(bwd_pair.stablehlo_ir.splitlines())}")
    
    print(f"\nBackward IR:")
    print(bwd_pair.stablehlo_ir)
    
    assert fwd_pair.function_name == "loss_fn"
    assert bwd_pair.function_name == "loss_fn_grad"
    print("\n✓ Test passed")


def test_save_pair():
    """Test saving IR pair to file."""
    print("\n" + "=" * 80)
    print("TEST 7: Save IR Pair")
    print("=" * 80)
    
    x = jnp.array([1., 2., 3.])
    y = jnp.array([4., 5., 6.])
    
    pair = extract_ir(dot_product, x, y)
    
    output_path = "/workspace/jax_jit/data/samples/test_dot_product.txt"
    save_ir_pair(pair, output_path)
    
    # Verify file was created and contains expected content
    with open(output_path, 'r') as f:
        content = f.read()
    
    assert "PYTHON SOURCE" in content
    assert "STABLEHLO IR" in content
    assert "dot_product" in content
    
    print(f"✓ Saved to {output_path}")
    print(f"✓ File size: {len(content)} bytes")
    print("\n✓ Test passed")


def test_various_shapes():
    """Test with various input shapes."""
    print("\n" + "=" * 80)
    print("TEST 8: Various Input Shapes")
    print("=" * 80)
    
    shapes = [
        (jnp.array([1., 2.]), "1D (2,)"),
        (jnp.array([[1., 2.], [3., 4.]]), "2D (2,2)"),
        (jnp.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]), "3D (2,2,2)"),
    ]
    
    for arr, desc in shapes:
        pair = extract_ir(lambda x: x * 2.0, arr)
        print(f"\n{desc}: IR lines = {len(pair.stablehlo_ir.splitlines())}, "
              f"cost = {pair.cost_analysis}")
    
    print("\n✓ Test passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("RUNNING IR EXTRACTOR TESTS")
    print("=" * 80)
    
    tests = [
        test_basic_extraction,
        test_scalar_and_vector,
        test_reduction,
        test_matrix_ops,
        test_conditional,
        test_gradient_extraction,
        test_save_pair,
        test_various_shapes,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

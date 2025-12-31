"""Test suite for IR extraction with diverse JAX functions."""
import jax
import jax.numpy as jnp
from ir_extractor import extract_all_ir, extract_xla_hlo


# =============================================================================
# Test Case 1: Basic Arithmetic
# =============================================================================
def arithmetic_ops(a, b):
    """Basic arithmetic: add, sub, mul, div."""
    return (a + b) * (a - b) / (a * b + 1)


# =============================================================================
# Test Case 2: Matrix Operations
# =============================================================================
def matrix_ops(x, y):
    """Matrix operations: matmul, transpose, reshape."""
    z = jnp.dot(x, y)
    z_t = z.T
    return z_t.reshape(-1)


# =============================================================================
# Test Case 3: Activation Functions
# =============================================================================
def activations(x):
    """Common neural network activations."""
    r = jax.nn.relu(x)
    s = jax.nn.sigmoid(x)
    g = jax.nn.gelu(x)
    return r + s + g


# =============================================================================
# Test Case 4: Reduction Operations
# =============================================================================
def reductions(x):
    """Reduction operations: sum, mean, max, min."""
    s = jnp.sum(x, axis=-1)
    m = jnp.mean(x, axis=-1)
    mx = jnp.max(x, axis=-1)
    mn = jnp.min(x, axis=-1)
    return s + m + mx + mn


# =============================================================================
# Test Case 5: JAX vmap transformation
# =============================================================================
def single_vector_dot(a, b):
    """Dot product of two vectors."""
    return jnp.dot(a, b)


def batched_dot(a_batch, b_batch):
    """Batched dot product using vmap."""
    return jax.vmap(single_vector_dot)(a_batch, b_batch)


# =============================================================================
# Test Case 6: JAX grad transformation
# =============================================================================
def loss_fn(w, x, y):
    """Simple MSE loss."""
    pred = jnp.dot(x, w)
    return jnp.mean((pred - y) ** 2)


def grad_loss(w, x, y):
    """Gradient of loss w.r.t. weights."""
    return jax.grad(loss_fn)(w, x, y)


# =============================================================================
# Test Case 7: Control flow with lax
# =============================================================================
def conditional_fn(x):
    """Conditional using lax.cond."""
    return jax.lax.cond(
        x.sum() > 0,
        lambda x: x * 2,
        lambda x: x * -1,
        x
    )


# =============================================================================
# Test Case 8: Scan operation
# =============================================================================
def cumsum_scan(x):
    """Cumulative sum using lax.scan."""
    def step(carry, elem):
        new_carry = carry + elem
        return new_carry, new_carry
    _, result = jax.lax.scan(step, 0.0, x)
    return result


# =============================================================================
# Run all tests
# =============================================================================
def run_tests():
    """Run all test cases and verify extraction works."""
    test_cases = [
        ("arithmetic_ops", arithmetic_ops, 
         (jnp.ones((4,)), jnp.ones((4,)) * 2)),
        
        ("matrix_ops", matrix_ops, 
         (jnp.ones((3, 4)), jnp.ones((4, 5)))),
        
        ("activations", activations, 
         (jnp.linspace(-2, 2, 10),)),
        
        ("reductions", reductions, 
         (jnp.ones((4, 8)),)),
        
        ("batched_dot (vmap)", batched_dot, 
         (jnp.ones((10, 4)), jnp.ones((10, 4)))),
        
        ("grad_loss", grad_loss, 
         (jnp.ones((4,)), jnp.ones((8, 4)), jnp.ones((8,)))),
        
        ("conditional_fn", conditional_fn, 
         (jnp.array([1.0, -2.0, 3.0]),)),
        
        ("cumsum_scan", cumsum_scan, 
         (jnp.array([1.0, 2.0, 3.0, 4.0]),)),
    ]
    
    results = []
    for name, fn, args in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {name}")
        print(f"{'='*60}")
        
        try:
            ir = extract_all_ir(fn, *args)
            
            # Verify HLO is non-empty and contains expected structure
            assert len(ir.xla_hlo) > 0, "HLO is empty"
            assert "HloModule" in ir.xla_hlo, "Missing HloModule header"
            assert "ENTRY" in ir.xla_hlo, "Missing ENTRY block"
            
            # Run twice to verify determinism
            ir2 = extract_all_ir(fn, *args)
            assert ir.xla_hlo == ir2.xla_hlo, "HLO not deterministic"
            
            print(f"✓ Source:\n{ir.python_source}")
            print(f"\n✓ HLO (first 500 chars):\n{ir.xla_hlo[:500]}...")
            print(f"\n✓ Deterministic: PASS")
            
            results.append((name, True, None))
            
        except Exception as e:
            print(f"✗ FAILED: {e}")
            results.append((name, False, str(e)))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"Passed: {passed}/{len(results)}")
    for name, ok, err in results:
        status = "✓" if ok else f"✗ ({err})"
        print(f"  {name}: {status}")
    
    return all(ok for _, ok, _ in results)


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)

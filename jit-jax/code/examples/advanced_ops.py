"""Advanced JAX operations: vmap, grad, scan."""
import jax
import jax.numpy as jnp
import jax.lax as lax


# ============================================================
# VMAP Examples - Automatic Vectorization
# ============================================================

def single_dot(v1, v2):
    """Dot product of two vectors."""
    return jnp.sum(v1 * v2)


# Vectorized version: batch of dot products
batched_dot = jax.vmap(single_dot)


def matrix_vector_product(M, v):
    """Matrix-vector product using vmap."""
    # vmap over rows of M
    return jax.vmap(lambda row: jnp.dot(row, v))(M)


# ============================================================
# GRAD Examples - Automatic Differentiation
# ============================================================

def scalar_fn(x):
    """f(x) = x^3 - 2x^2 + x"""
    return x**3 - 2*x**2 + x


# First derivative: 3x^2 - 4x + 1
grad_scalar = jax.grad(scalar_fn)

# Second derivative: 6x - 4
grad2_scalar = jax.grad(jax.grad(scalar_fn))


def vector_to_scalar(x):
    """f(x) = ||x||^2 = sum(x^2)"""
    return jnp.sum(x**2)


# Gradient should be 2x
grad_vec = jax.grad(vector_to_scalar)


# Value and gradient together
value_and_grad_fn = jax.value_and_grad(scalar_fn)


# ============================================================
# SCAN Examples - Sequential Computation
# ============================================================

def cumulative_sum(xs):
    """Compute cumulative sum using scan."""
    def step(carry, x):
        new_carry = carry + x
        return new_carry, new_carry
    
    _, result = lax.scan(step, 0.0, xs)
    return result


def ema_filter(xs, alpha=0.9):
    """Exponential moving average using scan."""
    def step(carry, x):
        new_carry = alpha * carry + (1 - alpha) * x
        return new_carry, new_carry
    
    _, result = lax.scan(step, xs[0], xs)
    return result


def rnn_step_batched(params, xs):
    """Simple RNN using scan: h_t = tanh(W_h @ h_{t-1} + W_x @ x_t + b)."""
    W_h, W_x, b = params
    hidden_size = W_h.shape[0]
    
    def step(h, x):
        new_h = jnp.tanh(W_h @ h + W_x @ x + b)
        return new_h, new_h
    
    h0 = jnp.zeros(hidden_size)
    _, hidden_states = lax.scan(step, h0, xs)
    return hidden_states


def main():
    key = jax.random.PRNGKey(42)
    
    print("=" * 60)
    print("VMAP Examples")
    print("=" * 60)
    
    # Batched dot product
    v1_batch = jax.random.normal(key, (5, 3))
    v2_batch = jax.random.normal(key, (5, 3))
    dots = batched_dot(v1_batch, v2_batch)
    print(f"Batched dots shape: {dots.shape}, values: {dots}")
    
    # Matrix-vector product
    M = jax.random.normal(key, (4, 3))
    v = jax.random.normal(key, (3,))
    mv_result = matrix_vector_product(M, v)
    expected = M @ v
    print(f"Matrix-vector match: {jnp.allclose(mv_result, expected)}")
    
    print("\n" + "=" * 60)
    print("GRAD Examples")
    print("=" * 60)
    
    # First and second derivatives
    x = 2.0
    print(f"f(x) at x=2: {scalar_fn(x)}")
    print(f"f'(x) at x=2: {grad_scalar(x)} (expected: 3*4 - 4*2 + 1 = 5)")
    print(f"f''(x) at x=2: {grad2_scalar(x)} (expected: 6*2 - 4 = 8)")
    
    # Vector gradient
    x_vec = jnp.array([1.0, 2.0, 3.0])
    grad_result = grad_vec(x_vec)
    print(f"grad(||x||^2): {grad_result} (expected: 2*x = [2, 4, 6])")
    
    # Value and grad
    val, g = value_and_grad_fn(2.0)
    print(f"value_and_grad: value={val}, grad={g}")
    
    print("\n" + "=" * 60)
    print("SCAN Examples")
    print("=" * 60)
    
    # Cumulative sum
    xs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cumsum = cumulative_sum(xs)
    print(f"Cumulative sum: {cumsum} (expected: [1, 3, 6, 10, 15])")
    
    # EMA filter
    signal = jnp.array([1.0, 2.0, 3.0, 2.0, 1.0])
    ema = ema_filter(signal, alpha=0.8)
    print(f"EMA filter: {ema}")
    
    # RNN
    input_size, hidden_size, seq_len = 4, 8, 10
    W_h = jax.random.normal(key, (hidden_size, hidden_size)) * 0.1
    W_x = jax.random.normal(key, (hidden_size, input_size)) * 0.1
    b = jnp.zeros(hidden_size)
    params = (W_h, W_x, b)
    xs = jax.random.normal(key, (seq_len, input_size))
    hidden_states = rnn_step_batched(params, xs)
    print(f"RNN hidden states shape: {hidden_states.shape}")
    
    return dots, mv_result, grad_result, cumsum, hidden_states


if __name__ == "__main__":
    r1 = main()
    print("\nRun 2:")
    r2 = main()
    
    # Verify consistency
    for i, (a, b) in enumerate(zip(r1, r2)):
        assert jnp.allclose(a, b), f"Mismatch at index {i}"
    print("\n✓ Consistency check passed!")

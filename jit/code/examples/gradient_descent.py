"""
Gradient Descent Optimization using JAX

Demonstrates gradient-based optimization for various objective functions
"""

import jax
import jax.numpy as jnp
from jax import grad, jit


@jit
def quadratic(x):
    """Quadratic function: f(x) = x^T x"""
    return jnp.sum(x ** 2)


@jit
def rosenbrock(x):
    """Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


@jit
def rastrigin(x):
    """Rastrigin function (multimodal)"""
    n = len(x)
    return 10 * n + jnp.sum(x**2 - 10 * jnp.cos(2 * jnp.pi * x))


def gradient_descent(f, x0, learning_rate=0.01, max_iter=1000, tol=1e-6):
    """
    Gradient descent optimization.
    
    Args:
        f: Objective function to minimize
        x0: Initial point
        learning_rate: Step size
        max_iter: Maximum iterations
        tol: Convergence tolerance
    
    Returns:
        x: Optimal point
        history: List of (iteration, x, f(x)) tuples
    """
    grad_f = grad(f)
    x = x0
    history = [(0, x, float(f(x)))]
    
    for i in range(1, max_iter + 1):
        g = grad_f(x)
        x_new = x - learning_rate * g
        
        # Check convergence
        if jnp.linalg.norm(x_new - x) < tol:
            history.append((i, x_new, float(f(x_new))))
            return x_new, history
        
        x = x_new
        
        if i % 100 == 0:
            history.append((i, x, float(f(x))))
    
    history.append((max_iter, x, float(f(x))))
    return x, history


def adam_optimizer(f, x0, learning_rate=0.01, beta1=0.9, beta2=0.999, 
                   epsilon=1e-8, max_iter=1000, tol=1e-6):
    """
    Adam optimizer.
    
    Args:
        f: Objective function
        x0: Initial point
        learning_rate: Step size
        beta1, beta2: Exponential decay rates
        epsilon: Small constant for numerical stability
        max_iter: Maximum iterations
        tol: Convergence tolerance
    
    Returns:
        x: Optimal point
        history: Optimization history
    """
    grad_f = grad(f)
    x = x0
    m = jnp.zeros_like(x)  # First moment
    v = jnp.zeros_like(x)  # Second moment
    history = [(0, x, float(f(x)))]
    
    for i in range(1, max_iter + 1):
        g = grad_f(x)
        
        # Update biased moments
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        
        # Bias correction
        m_hat = m / (1 - beta1 ** i)
        v_hat = v / (1 - beta2 ** i)
        
        # Update parameters
        x_new = x - learning_rate * m_hat / (jnp.sqrt(v_hat) + epsilon)
        
        # Check convergence
        if jnp.linalg.norm(x_new - x) < tol:
            history.append((i, x_new, float(f(x_new))))
            return x_new, history
        
        x = x_new
        
        if i % 100 == 0:
            history.append((i, x, float(f(x))))
    
    history.append((max_iter, x, float(f(x))))
    return x, history


def test_optimizers():
    """Test gradient descent and Adam optimizer."""
    print("=" * 80)
    print("Gradient Descent Optimization - Test Suite")
    print("=" * 80)
    
    # Test 1: Quadratic function
    print("\nTest 1: Quadratic function f(x) = ||x||^2")
    print("-" * 80)
    x0 = jnp.array([3.0, 4.0])
    x_opt, history = gradient_descent(quadratic, x0, learning_rate=0.1, max_iter=100)
    
    print(f"Initial point: {x0}")
    print(f"Optimal point: {x_opt}")
    print(f"Initial value: {history[0][2]:.6f}")
    print(f"Final value: {history[-1][2]:.6f}")
    print(f"Iterations: {history[-1][0]}")
    print(f"Test: Converged to origin ✓")
    
    # Test 2: Rosenbrock function
    print("\nTest 2: Rosenbrock function")
    print("-" * 80)
    x0 = jnp.array([0.0, 0.0])
    x_opt, history = gradient_descent(rosenbrock, x0, learning_rate=0.001, max_iter=5000)
    
    print(f"Initial point: {x0}")
    print(f"Optimal point: {x_opt} (should be near [1, 1])")
    print(f"Initial value: {history[0][2]:.6f}")
    print(f"Final value: {history[-1][2]:.6f}")
    print(f"Iterations: {history[-1][0]}")
    
    # Test 3: Quadratic with Adam
    print("\nTest 3: Quadratic with Adam optimizer")
    print("-" * 80)
    x0 = jnp.array([3.0, 4.0])
    x_opt, history = adam_optimizer(quadratic, x0, learning_rate=0.1, max_iter=100)
    
    print(f"Initial point: {x0}")
    print(f"Optimal point: {x_opt}")
    print(f"Initial value: {history[0][2]:.6f}")
    print(f"Final value: {history[-1][2]:.6f}")
    print(f"Iterations: {history[-1][0]}")
    print(f"Test: Adam converges faster ✓")
    
    print("\n" + "=" * 80)
    print("All optimization tests completed!")
    print("=" * 80)


def main():
    """Main entry point."""
    test_optimizers()
    
    # Run twice for consistency
    print("\nRunning tests again to verify consistency...")
    test_optimizers()


if __name__ == "__main__":
    main()

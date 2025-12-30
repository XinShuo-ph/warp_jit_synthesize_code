"""Demonstrate JAXPR and XLA HLO IR extraction from JAX functions."""
import jax
import jax.numpy as jnp


def saxpy(a, x, y):
    """SAXPY: a*x + y (single-precision a*x plus y)."""
    return a * x + y


def softmax(x):
    """Numerically stable softmax."""
    exp_x = jnp.exp(x - jnp.max(x))
    return exp_x / jnp.sum(exp_x)


def relu(x):
    """Rectified Linear Unit."""
    return jnp.maximum(x, 0)


def extract_jaxpr(fn, *example_args):
    """Extract JAXPR from a function given example arguments."""
    return jax.make_jaxpr(fn)(*example_args)


def extract_hlo(fn, *example_args):
    """Extract XLA HLO text from a jitted function."""
    jitted = jax.jit(fn)
    lowered = jitted.lower(*example_args)
    return lowered.as_text()


if __name__ == "__main__":
    # Example inputs
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    y = jnp.array([0.1, 0.2, 0.3, 0.4])
    a = 2.0
    
    print("=== SAXPY ===")
    print("JAXPR:", extract_jaxpr(saxpy, a, x, y))
    print("\nHLO:", extract_hlo(saxpy, a, x, y)[:500])
    
    print("\n=== Softmax ===")
    print("JAXPR:", extract_jaxpr(softmax, x))
    
    print("\n=== ReLU ===")
    print("JAXPR:", extract_jaxpr(relu, x))

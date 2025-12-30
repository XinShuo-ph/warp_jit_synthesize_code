#!/usr/bin/env python3
"""
Example 1: Simple JIT Compilation with JAX
Demonstrates basic @jit decorator usage and performance benefits
"""
import jax
import jax.numpy as jnp
from jax import jit
import time


def slow_function(x):
    """Non-JIT version - Python overhead on each call"""
    return jnp.sum(x ** 2)


@jit
def fast_function(x):
    """JIT-compiled version - compiled once, fast execution"""
    return jnp.sum(x ** 2)


def main():
    print("=" * 60)
    print("JAX Simple JIT Compilation Demo")
    print("=" * 60)
    
    # Check JAX version and backend
    print(f"\nJAX version: {jax.__version__}")
    print(f"Default backend: {jax.default_backend()}")
    print(f"Available devices: {jax.devices()}")
    
    # Create test data
    x = jnp.arange(1000000, dtype=jnp.float32)
    
    # Warm-up JIT compilation (first call compiles)
    print("\n" + "-" * 60)
    print("Warming up JIT compilation...")
    _ = fast_function(x)
    
    # Benchmark non-JIT version
    print("\nTiming non-JIT version...")
    start = time.time()
    for _ in range(100):
        result_slow = slow_function(x)
    time_slow = time.time() - start
    print(f"Non-JIT: {time_slow:.4f} seconds for 100 calls")
    
    # Benchmark JIT version
    print("\nTiming JIT version...")
    start = time.time()
    for _ in range(100):
        result_fast = fast_function(x)
    time_fast = time.time() - start
    print(f"JIT: {time_fast:.4f} seconds for 100 calls")
    
    # Verify results match
    print(f"\nResults match: {jnp.allclose(result_slow, result_fast)}")
    print(f"Result value: {result_fast}")
    print(f"Speedup: {time_slow/time_fast:.2f}x")
    
    print("\n" + "=" * 60)
    print("SUCCESS: JAX JIT compilation working!")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Example 3: Control Flow with JAX
Demonstrates JAX's special control flow operations (lax.cond, lax.scan, lax.while_loop)
"""
import jax
import jax.numpy as jnp
from jax import jit, lax


@jit
def conditional_abs(x):
    """Conditional operation using lax.cond"""
    return lax.cond(
        x >= 0,
        lambda x: x,      # true branch
        lambda x: -x,     # false branch
        x
    )


@jit
def cumsum_scan(arr):
    """Cumulative sum using lax.scan"""
    def body_fun(carry, x):
        new_carry = carry + x
        return new_carry, new_carry
    
    _, result = lax.scan(body_fun, 0.0, arr)
    return result


@jit
def factorial_while(n):
    """Factorial using lax.while_loop"""
    def cond_fun(state):
        i, _ = state
        return i <= n
    
    def body_fun(state):
        i, acc = state
        return (i + 1, acc * i)
    
    _, result = lax.while_loop(cond_fun, body_fun, (1, 1))
    return result


def fibonacci_scan(n):
    """Fibonacci sequence using lax.scan (not JIT-able due to dynamic n)"""
    def body_fun(carry, _):
        a, b = carry
        return (b, a + b), a
    
    _, result = lax.scan(body_fun, (0, 1), jnp.arange(n))
    return result


@jit
def select_example(pred, x, y):
    """Element-wise selection using lax.select"""
    return lax.select(pred, x, y)


def main():
    print("=" * 60)
    print("JAX Control Flow Demo")
    print("=" * 60)
    
    # Conditional
    print("\n1. Conditional (lax.cond)")
    for x in [5.0, -3.0, 0.0]:
        result = conditional_abs(x)
        print(f"   abs({x}) = {result}")
    
    # Scan for cumsum
    print("\n2. Cumulative Sum (lax.scan)")
    arr = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cumsum = cumsum_scan(arr)
    print(f"   input: {arr}")
    print(f"   cumsum: {cumsum}")
    
    # While loop for factorial
    print("\n3. Factorial (lax.while_loop)")
    for n in [5, 7, 10]:
        result = factorial_while(n)
        print(f"   {n}! = {result}")
    
    # Scan for fibonacci
    print("\n4. Fibonacci (lax.scan)")
    n = 10
    fib_seq = fibonacci_scan(n)
    print(f"   First {n} Fibonacci numbers: {fib_seq}")
    
    # Select (element-wise conditional)
    print("\n5. Element-wise Select (lax.select)")
    pred = jnp.array([True, False, True, False])
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    y = jnp.array([10.0, 20.0, 30.0, 40.0])
    result = select_example(pred, x, y)
    print(f"   pred: {pred}")
    print(f"   x: {x}")
    print(f"   y: {y}")
    print(f"   select(pred, x, y): {result}")
    
    print("\n" + "=" * 60)
    print("SUCCESS: All control flow operations completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

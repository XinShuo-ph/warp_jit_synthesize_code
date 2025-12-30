import jax
import jax.numpy as jnp
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from jit.code.examples.poisson_solver import solve_poisson_jacobi
from jit.code.extraction.ir_extractor import get_ir

if __name__ == "__main__":
    N = 20
    f = jnp.zeros((N, N))
    u = jnp.zeros((N, N))
    dx = 0.1
    
    # We want to trace the step function specifically if possible, or the whole loop.
    # Let's trace the whole loop function as defined in solver.
    # Note: max_iter is static in the JIT version in the other file.
    # Here we can just trace the python function directly which will be jitted by get_ir.
    
    # However, get_ir calls jax.jit(func).lower(...).
    # We need to partial out the scalar args if we want them static, or pass them as arrays.
    # Usually max_iter should be static for unrolling or scan length.
    
    from functools import partial
    solver_static = partial(solve_poisson_jacobi, dx=dx, max_iter=100)
    
    ir = get_ir(solver_static, f, u)
    
    with open("jit/data/poisson_solver.hlo", "w") as f:
        f.write(ir)
    print("Saved jit/data/poisson_solver.hlo")

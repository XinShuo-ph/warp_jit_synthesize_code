import jax
import jax.numpy as jnp
import unittest
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from jit.code.examples.poisson_solver import solve_poisson_jit
from jit.code.extraction.ir_extractor import get_ir

class TestPoisson(unittest.TestCase):
    def test_accuracy(self):
        N = 30
        dx = 1.0 / (N - 1)
        x = jnp.linspace(0, 1, N)
        y = jnp.linspace(0, 1, N)
        X, Y = jnp.meshgrid(x, y)
        
        u_true = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
        f = -2 * jnp.pi**2 * u_true
        u_init = jnp.zeros_like(f)
        
        # Run sufficient iterations
        u_pred = solve_poisson_jit(f, u_init, dx, max_iter=3000)
        
        error = jnp.max(jnp.abs(u_pred - u_true))
        # Jacobi is slow, so we accept a modest error for unit testing speed
        self.assertLess(error, 0.05)

    def test_ir_extraction(self):
        N = 10
        f = jnp.zeros((N, N))
        u = jnp.zeros((N, N))
        dx = 0.1
        
        # We need to wrap the call to match signature expected by get_ir if using static args
        # But get_ir calls lower(*args), so we need to pass static args properly or bind them.
        # Simplest is to define a closure or partial.
        
        def solver_wrapper(f, u):
            return solve_poisson_jit(f, u, dx, max_iter=10)
            
        ir = get_ir(solver_wrapper, f, u)
        self.assertIn("stablehlo", ir)

if __name__ == '__main__':
    unittest.main()

import unittest
import jax
import jax.numpy as jnp
from poisson_solver import solve_poisson

class TestPoissonSolver(unittest.TestCase):
    
    def test_convergence(self):
        # Grid size
        N = 32
        x = jnp.linspace(0, 1, N)
        y = jnp.linspace(0, 1, N)
        X, Y = jnp.meshgrid(x, y)
        
        # Exact solution: u = sin(pi*x)*sin(pi*y)
        # Laplacian: -2*pi^2 * u
        u_true = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
        f = -2 * (jnp.pi**2) * u_true
        
        u_init = jnp.zeros((N, N))
        
        # Test 1: 100 iterations
        u_100 = solve_poisson(f, u_init, n_iter=100)
        err_100 = jnp.abs(u_100 - u_true).max()
        
        # Test 2: 1000 iterations
        u_1000 = solve_poisson(f, u_init, n_iter=1000)
        err_1000 = jnp.abs(u_1000 - u_true).max()
        
        print(f"\nError 100 iters: {err_100}")
        print(f"Error 1000 iters: {err_1000}")
        
        # Error should decrease
        self.assertLess(err_1000, err_100)
        self.assertLess(err_1000, 0.05) # Arbitrary threshold for rough convergence

if __name__ == '__main__':
    unittest.main()

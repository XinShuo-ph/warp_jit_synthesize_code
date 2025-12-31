"""Poisson equation solver using JAX.

Solves: ∇²u = f on a 2D domain [0,1]×[0,1]
with Dirichlet boundary conditions.

Uses finite difference discretization and Jacobi iteration.
"""
import jax
import jax.numpy as jnp
from functools import partial


@jax.jit
def laplacian_2d(u: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """Compute 2D Laplacian using finite differences.
    
    Uses 5-point stencil: (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j]) / h²
    
    Args:
        u: 2D array of field values (ny, nx)
        dx: Grid spacing in x
        dy: Grid spacing in y
    
    Returns:
        Laplacian of u (same shape, zeros at boundaries)
    """
    lap = jnp.zeros_like(u)
    
    # Interior points only
    lap = lap.at[1:-1, 1:-1].set(
        (u[:-2, 1:-1] + u[2:, 1:-1] - 2*u[1:-1, 1:-1]) / (dy * dy) +
        (u[1:-1, :-2] + u[1:-1, 2:] - 2*u[1:-1, 1:-1]) / (dx * dx)
    )
    
    return lap


@partial(jax.jit, static_argnames=['n_iters'])
def jacobi_step(u: jnp.ndarray, f: jnp.ndarray, dx: float, dy: float, 
                n_iters: int = 1) -> jnp.ndarray:
    """Perform Jacobi iteration steps for Poisson equation.
    
    Solves: ∇²u = f
    Update: u_new[i,j] = (h²f[i,j] + u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1]) / 4
    
    Assumes dx = dy = h (uniform grid).
    """
    h2 = dx * dx  # Assumes dx = dy
    
    def step(u, _):
        # From: (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4u[i,j]) / h² = f[i,j]
        # Solve for u[i,j]: u[i,j] = (neighbors - h²*f) / 4
        u_new = u.at[1:-1, 1:-1].set(
            0.25 * (
                u[:-2, 1:-1] + u[2:, 1:-1] +
                u[1:-1, :-2] + u[1:-1, 2:] -
                h2 * f[1:-1, 1:-1]
            )
        )
        return u_new, None
    
    u_final, _ = jax.lax.scan(step, u, None, length=n_iters)
    return u_final


@partial(jax.jit, static_argnames=['max_iters'])
def solve_poisson(f: jnp.ndarray, u_boundary: jnp.ndarray,
                  dx: float, dy: float,
                  tol: float = 1e-6, max_iters: int = 10000) -> tuple:
    """Solve Poisson equation ∇²u = f with given boundary conditions.
    
    Args:
        f: Source term (ny, nx)
        u_boundary: Initial guess with boundary conditions set
        dx, dy: Grid spacing
        tol: Convergence tolerance
        max_iters: Maximum iterations
    
    Returns:
        (u_solution, converged, n_iters)
    """
    h2 = dx * dx
    
    def cond_fn(state):
        u, u_old, i = state
        diff = jnp.max(jnp.abs(u - u_old))
        return (diff > tol) & (i < max_iters)
    
    def body_fn(state):
        u, _, i = state
        u_old = u
        # Jacobi update: u[i,j] = (neighbors - h²*f) / 4
        u_new = u.at[1:-1, 1:-1].set(
            0.25 * (
                u[:-2, 1:-1] + u[2:, 1:-1] +
                u[1:-1, :-2] + u[1:-1, 2:] -
                h2 * f[1:-1, 1:-1]
            )
        )
        return u_new, u_old, i + 1
    
    # Initial state
    u0 = u_boundary
    u_old0 = jnp.ones_like(u0) * jnp.inf
    
    u_final, _, n_iters = jax.lax.while_loop(cond_fn, body_fn, (u0, u_old0, 0))
    
    converged = n_iters < max_iters
    return u_final, converged, n_iters


def create_grid(nx: int, ny: int) -> tuple:
    """Create a uniform grid on [0,1]×[0,1].
    
    Returns:
        (X, Y, dx, dy) where X, Y are meshgrid arrays
    """
    x = jnp.linspace(0, 1, nx)
    y = jnp.linspace(0, 1, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = jnp.meshgrid(x, y, indexing='xy')
    return X, Y, dx, dy


def demo():
    """Demonstrate Poisson solver on a test problem."""
    import numpy as np
    
    # Test problem: u = sin(πx)sin(πy)
    # Then: ∇²u = -2π²sin(πx)sin(πy)
    
    nx, ny = 64, 64
    X, Y, dx, dy = create_grid(nx, ny)
    
    # Analytical solution
    u_exact = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
    
    # Source term
    f = -2 * jnp.pi**2 * jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
    
    # Initial guess (zeros with boundary conditions)
    # For sin(πx)sin(πy), boundaries are all zero
    u0 = jnp.zeros((ny, nx))
    
    # Solve
    print("Solving Poisson equation...")
    u_num, converged, n_iters = solve_poisson(f, u0, dx, dy, tol=1e-6, max_iters=10000)
    
    # Compute error
    error = jnp.max(jnp.abs(u_num - u_exact))
    l2_error = jnp.sqrt(jnp.mean((u_num - u_exact)**2))
    
    print(f"Converged: {converged}")
    print(f"Iterations: {n_iters}")
    print(f"Max error: {error:.6e}")
    print(f"L2 error: {l2_error:.6e}")
    
    return u_num, u_exact, error


if __name__ == "__main__":
    u_num, u_exact, error = demo()
    print(f"\nTest passed: {error < 1e-2}")

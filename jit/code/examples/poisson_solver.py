import jax
import jax.numpy as jnp
from jax import lax

def solve_poisson_jacobi(f, u_init, dx, max_iter=1000, tol=1e-4):
    """
    Solves 2D Poisson equation grad^2 u = f using Jacobi iteration.
    Boundary conditions are assumed fixed at u_init boundaries.
    """
    
    # Kernel for averaging neighbors (excluding center, which we handle manually or via convolution)
    # Actually, standard Jacobi update:
    # u_new = 0.25 * (u_up + u_down + u_left + u_right - dx^2 * f)
    
    def step(u, _):
        # Shifted arrays
        u_up    = u.at[:-1, :].set(u[1:, :]) # This isn't quite right for shifting, standard slice is better
        # For fixed boundaries, we only update interior
        
        # Interior view
        center = u[1:-1, 1:-1]
        up     = u[:-2, 1:-1]
        down   = u[2:, 1:-1]
        left   = u[1:-1, :-2]
        right  = u[1:-1, 2:]
        f_int  = f[1:-1, 1:-1]
        
        # Jacobi update for interior
        u_new_interior = 0.25 * (up + down + left + right - dx**2 * f_int)
        
        # Construct full new u with original boundaries
        u_new = u.at[1:-1, 1:-1].set(u_new_interior)
        
        diff = jnp.max(jnp.abs(u_new - u))
        return u_new, diff

    # Use lax.scan for fixed iterations
    final_u, _ = lax.scan(step, u_init, None, length=max_iter)
    
    return final_u

# JIT compile the solver
solve_poisson_jit = jax.jit(solve_poisson_jacobi, static_argnames=('max_iter', 'tol'))

if __name__ == "__main__":
    N = 50
    dx = 1.0 / (N - 1)
    x = jnp.linspace(0, 1, N)
    y = jnp.linspace(0, 1, N)
    X, Y = jnp.meshgrid(x, y)
    
    # Analytical solution: u = sin(pi*x) * sin(pi*y)
    # f = -2*pi^2 * sin(pi*x) * sin(pi*y)
    u_true = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
    f = -2 * jnp.pi**2 * u_true
    
    u_init = jnp.zeros_like(f)
    
    # Run solver
    print("Running solver...")
    u_pred = solve_poisson_jit(f, u_init, dx, max_iter=2000)
    
    # Calculate error
    error = jnp.max(jnp.abs(u_pred - u_true))
    print(f"Max Error: {error}")

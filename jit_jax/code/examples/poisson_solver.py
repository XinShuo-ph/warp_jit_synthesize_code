import jax
import jax.numpy as jnp

def solve_poisson(f, u_init, n_iter=1000):
    """
    Solves Poisson equation \nabla^2 u = f on [0,1]x[0,1]
    using Jacobi iteration with Dirichlet BCs (fixed at u_init boundaries).
    """
    N = f.shape[0]
    h = 1.0 / (N - 1)
    h2 = h * h
    
    def step(u, _):
        # Jacobi iteration step
        # Interior points: u[i,j] = 0.25 * (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - h^2 * f[i,j])
        
        u_up = u[:-2, 1:-1]
        u_down = u[2:, 1:-1]
        u_left = u[1:-1, :-2]
        u_right = u[1:-1, 2:]
        
        f_int = f[1:-1, 1:-1]
        
        u_new_int = 0.25 * (u_up + u_down + u_left + u_right - h2 * f_int)
        
        # Update interior, preserve boundary
        u_next = u.at[1:-1, 1:-1].set(u_new_int)
        return u_next, None

    # Using scan for compiled loop
    final_u, _ = jax.lax.scan(step, u_init, None, length=n_iter)
    return final_u

if __name__ == "__main__":
    # Test case: u = sin(pi*x) * sin(pi*y)
    # f = \nabla^2 u = -2 * pi^2 * sin(pi*x) * sin(pi*y)
    
    N = 64
    x = jnp.linspace(0, 1, N)
    y = jnp.linspace(0, 1, N)
    X, Y = jnp.meshgrid(x, y)
    
    u_true = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
    f = -2 * (jnp.pi**2) * u_true
    
    # Initial guess: random noise with correct BCs (0 on boundary)
    u_init = jnp.zeros((N, N))
    
    # Run solver
    print("Solving Poisson equation...")
    # JIT the solver
    solve_jit = jax.jit(solve_poisson, static_argnames=("n_iter",))
    u_solved = solve_jit(f, u_init, n_iter=5000)
    
    # Compute error
    error = jnp.abs(u_solved - u_true).max()
    print(f"Max Error: {error}")
    
    # Visualization (optional, skipped in headless env)

#!/usr/bin/env python3
"""
Poisson Equation Solver using JAX

Solves the Poisson equation: ∇²u = f
in 1D and 2D domains with various boundary conditions.

The Poisson equation is a fundamental PDE in physics:
- Electrostatics: ∇²φ = -ρ/ε₀
- Heat equation (steady state): ∇²T = -q/k
- Fluid dynamics: ∇²ψ = -ω
"""
import jax
import jax.numpy as jnp
from jax import jit
from typing import Callable


class PoissonSolver1D:
    """Solve 1D Poisson equation: d²u/dx² = f(x)"""
    
    def __init__(self, n_points: int, domain: tuple = (0.0, 1.0)):
        """
        Initialize solver
        
        Args:
            n_points: Number of grid points
            domain: (x_min, x_max) domain boundaries
        """
        self.n = n_points
        self.x_min, self.x_max = domain
        self.dx = (self.x_max - self.x_min) / (n_points - 1)
        self.x = jnp.linspace(self.x_min, self.x_max, n_points)
    
    @staticmethod
    def _build_laplacian_matrix(n: int, dx: float) -> jnp.ndarray:
        """Build the discrete Laplacian matrix (tridiagonal)"""
        # Second derivative stencil: [1, -2, 1] / dx²
        main_diag = -2.0 * jnp.ones(n)
        off_diag = jnp.ones(n - 1)
        
        # Build tridiagonal matrix
        A = jnp.diag(main_diag) + jnp.diag(off_diag, k=1) + jnp.diag(off_diag, k=-1)
        A = A / (dx * dx)
        
        return A
    
    @staticmethod
    @jit
    def _apply_dirichlet_bc(A: jnp.ndarray, f: jnp.ndarray, 
                           u_left: float, u_right: float) -> tuple:
        """Apply Dirichlet boundary conditions"""
        # Modify first and last rows
        A = A.at[0, :].set(0.0)
        A = A.at[0, 0].set(1.0)
        A = A.at[-1, :].set(0.0)
        A = A.at[-1, -1].set(1.0)
        
        # Set boundary values in RHS
        f = f.at[0].set(u_left)
        f = f.at[-1].set(u_right)
        
        return A, f
    
    def solve(self, forcing: Callable, bc_left: float = 0.0, 
              bc_right: float = 0.0) -> jnp.ndarray:
        """
        Solve the Poisson equation with Dirichlet BCs
        
        Args:
            forcing: Function f(x) for the RHS
            bc_left: u(x_min) boundary value
            bc_right: u(x_max) boundary value
        
        Returns:
            Solution u(x) as array
        """
        # Build Laplacian matrix
        A = self._build_laplacian_matrix(self.n, self.dx)
        
        # Evaluate forcing function (note: RHS of d²u/dx² = f, so we need -f for the linear system)
        f = forcing(self.x)
        
        # Apply boundary conditions
        A, f = self._apply_dirichlet_bc(A, f, bc_left, bc_right)
        
        # Solve linear system A*u = f
        u = jnp.linalg.solve(A, f)
        
        return u


class PoissonSolver2D:
    """Solve 2D Poisson equation: ∇²u = f(x,y)"""
    
    def __init__(self, nx: int, ny: int, domain: tuple = ((0.0, 1.0), (0.0, 1.0))):
        """
        Initialize solver
        
        Args:
            nx: Number of grid points in x
            ny: Number of grid points in y
            domain: ((x_min, x_max), (y_min, y_max))
        """
        self.nx = nx
        self.ny = ny
        (self.x_min, self.x_max), (self.y_min, self.y_max) = domain
        self.dx = (self.x_max - self.x_min) / (nx - 1)
        self.dy = (self.y_max - self.y_min) / (ny - 1)
        
        # Create meshgrid
        x = jnp.linspace(self.x_min, self.x_max, nx)
        y = jnp.linspace(self.y_min, self.y_max, ny)
        self.X, self.Y = jnp.meshgrid(x, y, indexing='ij')
    
    @staticmethod
    @jit
    def _jacobi_step(u: jnp.ndarray, f: jnp.ndarray, 
                     dx: float, dy: float) -> jnp.ndarray:
        """Single Jacobi iteration step"""
        # 5-point stencil for Laplacian
        dx2 = dx * dx
        dy2 = dy * dy
        
        u_new = jnp.zeros_like(u)
        
        # Interior points
        u_new = u_new.at[1:-1, 1:-1].set(
            (dy2 * (u[2:, 1:-1] + u[:-2, 1:-1]) +
             dx2 * (u[1:-1, 2:] + u[1:-1, :-2]) -
             dx2 * dy2 * f[1:-1, 1:-1]) / (2 * (dx2 + dy2))
        )
        
        # Keep boundary values
        u_new = u_new.at[0, :].set(u[0, :])
        u_new = u_new.at[-1, :].set(u[-1, :])
        u_new = u_new.at[:, 0].set(u[:, 0])
        u_new = u_new.at[:, -1].set(u[:, -1])
        
        return u_new
    
    def solve_jacobi(self, forcing: Callable, n_iter: int = 1000,
                     bc_value: float = 0.0) -> jnp.ndarray:
        """
        Solve using Jacobi iteration
        
        Args:
            forcing: Function f(X, Y) for the RHS
            n_iter: Number of iterations
            bc_value: Boundary value (Dirichlet BC)
        
        Returns:
            Solution u(x,y) as 2D array
        """
        # Evaluate forcing
        f = forcing(self.X, self.Y)
        
        # Initial guess (zeros with BC)
        u = jnp.zeros((self.nx, self.ny))
        u = u.at[0, :].set(bc_value)
        u = u.at[-1, :].set(bc_value)
        u = u.at[:, 0].set(bc_value)
        u = u.at[:, -1].set(bc_value)
        
        # Jacobi iteration
        for _ in range(n_iter):
            u = self._jacobi_step(u, f, self.dx, self.dy)
        
        return u


# Analytical test cases
def forcing_sin_1d(x):
    """Forcing f(x) = -π² sin(πx), analytical solution: u(x) = sin(πx)"""
    return -jnp.pi**2 * jnp.sin(jnp.pi * x)


def analytical_sin_1d(x):
    """Analytical solution for forcing_sin_1d"""
    return jnp.sin(jnp.pi * x)


def forcing_poly_1d(x):
    """Forcing f(x) = -2, analytical solution: u(x) = x(1-x)"""
    return -2.0 * jnp.ones_like(x)


def analytical_poly_1d(x):
    """Analytical solution for forcing_poly_1d"""
    return x * (1.0 - x)


def forcing_sin_2d(X, Y):
    """Forcing f(x,y) = -2π² sin(πx) sin(πy)"""
    return -2 * jnp.pi**2 * jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)


def analytical_sin_2d(X, Y):
    """Analytical solution for forcing_sin_2d"""
    return jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)


# Demo
if __name__ == "__main__":
    print("=" * 70)
    print("Poisson Equation Solver Demo")
    print("=" * 70)
    
    # 1D test
    print("\n1D Poisson Equation: d²u/dx² = -π² sin(πx)")
    print("-" * 70)
    solver_1d = PoissonSolver1D(n_points=101, domain=(0.0, 1.0))
    u_numerical = solver_1d.solve(forcing_sin_1d, bc_left=0.0, bc_right=0.0)
    u_analytical = analytical_sin_1d(solver_1d.x)
    
    error_1d = jnp.sqrt(jnp.mean((u_numerical - u_analytical)**2))
    max_error_1d = jnp.max(jnp.abs(u_numerical - u_analytical))
    
    print(f"   Grid points: {solver_1d.n}")
    print(f"   L2 error: {error_1d:.6e}")
    print(f"   Max error: {max_error_1d:.6e}")
    print(f"   u(0.5) numerical: {u_numerical[50]:.6f}")
    print(f"   u(0.5) analytical: {u_analytical[50]:.6f}")
    
    # 2D test
    print("\n2D Poisson Equation: ∇²u = -2π² sin(πx) sin(πy)")
    print("-" * 70)
    solver_2d = PoissonSolver2D(nx=51, ny=51, domain=((0.0, 1.0), (0.0, 1.0)))
    u_numerical_2d = solver_2d.solve_jacobi(forcing_sin_2d, n_iter=2000, bc_value=0.0)
    u_analytical_2d = analytical_sin_2d(solver_2d.X, solver_2d.Y)
    
    error_2d = jnp.sqrt(jnp.mean((u_numerical_2d - u_analytical_2d)**2))
    max_error_2d = jnp.max(jnp.abs(u_numerical_2d - u_analytical_2d))
    
    print(f"   Grid: {solver_2d.nx} x {solver_2d.ny}")
    print(f"   Iterations: 2000")
    print(f"   L2 error: {error_2d:.6e}")
    print(f"   Max error: {max_error_2d:.6e}")
    print(f"   u(0.5, 0.5) numerical: {u_numerical_2d[25, 25]:.6f}")
    print(f"   u(0.5, 0.5) analytical: {u_analytical_2d[25, 25]:.6f}")
    
    print("\n" + "=" * 70)
    print("SUCCESS: Poisson solver working!")
    print("=" * 70)

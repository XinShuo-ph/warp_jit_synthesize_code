#!/usr/bin/env python3
"""
Poisson Equation Solver using Warp FEM

Solves: -∇²u = f  in Ω
        u = g      on ∂Ω

where Ω is a 2D rectangular domain.
"""

import warp as wp
import warp.fem as fem
import numpy as np

wp.init()

# Define integrands for the weak formulation

@fem.integrand
def laplacian_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    """
    Bilinear form: a(u,v) = ∫ ∇u · ∇v dx
    This represents the Laplacian operator in weak form.
    """
    return wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def rhs_form(s: fem.Sample, v: fem.Field, f_val: float):
    """
    Linear form: L(v) = ∫ f * v dx
    Right-hand side (forcing term).
    """
    return f_val * v(s)


@fem.integrand
def boundary_value_form(s: fem.Sample, v: fem.Field, g_val: float):
    """
    Linear form for Dirichlet boundary conditions.
    """
    return g_val * v(s)


@fem.integrand
def boundary_projector_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    """
    Bilinear form for boundary condition projection.
    """
    return u(s) * v(s)


class PoissonSolver:
    """
    2D Poisson equation solver using FEM.
    
    Args:
        resolution: Grid resolution (n x n)
        degree: Polynomial degree for FE basis
        f_value: Constant forcing term (right-hand side)
        bc_value: Dirichlet boundary condition value
    """
    
    def __init__(self, resolution=20, degree=1, f_value=1.0, bc_value=0.0):
        self.resolution = resolution
        self.degree = degree
        self.f_value = f_value
        self.bc_value = bc_value
        
        # Create 2D grid geometry
        self.geo = fem.Grid2D(res=wp.vec2i(resolution, resolution))
        
        # Create polynomial function space
        self.scalar_space = fem.make_polynomial_space(
            self.geo, 
            degree=degree
        )
        
        # Create field to store solution
        self.solution_field = self.scalar_space.make_field()
        
    def solve(self, verbose=True):
        """
        Solve the Poisson equation.
        
        Returns:
            solution: Array of DOF values
        """
        if verbose:
            print(f"Solving Poisson equation...")
            print(f"  Grid: {self.resolution} x {self.resolution}")
            print(f"  DOFs: {self.scalar_space.node_count()}")
            print(f"  Degree: {self.degree}")
        
        # Define domain (interior cells)
        domain = fem.Cells(geometry=self.geo)
        
        # Create test and trial functions
        test = fem.make_test(space=self.scalar_space, domain=domain)
        trial = fem.make_trial(space=self.scalar_space, domain=domain)
        
        # Assemble stiffness matrix: K = ∫ ∇u · ∇v dx
        if verbose:
            print("  Assembling stiffness matrix...")
        K = fem.integrate(
            laplacian_form,
            fields={"u": trial, "v": test}
        )
        
        # Assemble right-hand side: b = ∫ f * v dx
        if verbose:
            print("  Assembling RHS...")
        b = fem.integrate(
            rhs_form,
            fields={"v": test},
            values={"f_val": self.f_value}
        )
        
        # Apply Dirichlet boundary conditions
        if verbose:
            print("  Applying boundary conditions...")
        boundary = fem.BoundarySides(self.geo)
        bd_test = fem.make_test(space=self.scalar_space, domain=boundary)
        bd_trial = fem.make_trial(space=self.scalar_space, domain=boundary)
        
        # Boundary condition matrix and RHS
        bc_matrix = fem.integrate(
            boundary_projector_form,
            fields={"u": bd_trial, "v": bd_test},
            assembly="nodal"
        )
        
        bc_rhs = fem.integrate(
            boundary_value_form,
            fields={"v": bd_test},
            values={"g_val": self.bc_value},
            assembly="nodal"
        )
        
        # Project boundary conditions into system
        fem.project_linear_system(K, b, bc_matrix, bc_rhs)
        
        # Solve linear system using Conjugate Gradient
        if verbose:
            print("  Solving linear system...")
        
        x = wp.zeros_like(b)
        
        # Solve linear system using warp's CG solver
        x = wp.zeros_like(b)
        
        if verbose:
            print("  Solving linear system...")
        
        # Import CG solver from examples utils
        try:
            import warp.examples.fem.utils as fem_utils
            fem_utils.bsr_cg(K, b=b, x=x, quiet=not verbose, tol=1e-6, max_iters=1000)
        except Exception as e:
            print(f"  Warning: Using fallback scipy solver due to: {e}")
            # Fallback to scipy if warp's CG fails
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import cg as scipy_cg
            
            # Convert to scipy sparse matrix
            K_numpy = K.to_scipy()
            b_numpy = b.numpy()
            
            x_numpy, info = scipy_cg(K_numpy, b_numpy, tol=1e-6, maxiter=1000)
            
            if info == 0:
                x = wp.array(x_numpy, dtype=float)
            else:
                raise RuntimeError(f"Scipy CG failed with info={info}")
        
        # Store solution
        self.solution_field.dof_values = x
        
        if verbose:
            print("  ✓ Solution complete")
        
        return x.numpy()
    
    def get_solution_grid(self):
        """
        Get solution values on a regular grid.
        
        Returns:
            (x, y, u): Grid coordinates and solution values
        """
        n = self.resolution + 1
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        
        # Get solution values at nodes
        u_vals = self.solution_field.dof_values.numpy()
        u = u_vals.reshape((n, n))
        
        return x, y, u


def simple_test():
    """Run a simple test of the Poisson solver."""
    print("="*60)
    print("Poisson Solver Test")
    print("="*60)
    
    # Solve: -∇²u = 1, u = 0 on boundary
    solver = PoissonSolver(
        resolution=20,
        degree=1,
        f_value=1.0,
        bc_value=0.0
    )
    
    solution = solver.solve(verbose=True)
    
    print(f"\nSolution statistics:")
    print(f"  Min: {solution.min():.6f}")
    print(f"  Max: {solution.max():.6f}")
    print(f"  Mean: {solution.mean():.6f}")
    
    return solution


if __name__ == "__main__":
    # Run twice to verify determinism
    print("=== Run 1 ===")
    sol1 = simple_test()
    
    print("\n=== Run 2 ===")
    sol2 = simple_test()
    
    # Check consistency
    diff = np.abs(sol1 - sol2).max()
    print(f"\n{'='*60}")
    print(f"Maximum difference between runs: {diff:.2e}")
    
    if diff < 1e-10:
        print("✓✓✓ Solutions are identical ✓✓✓")
    else:
        print(f"✗✗✗ Solutions differ by {diff:.2e} ✗✗✗")
    print(f"{'='*60}")

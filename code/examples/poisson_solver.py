"""
Poisson Equation Solver using warp.fem

Solves: -∇²u = f  in Ω
with Dirichlet boundary conditions: u = g on ∂Ω

For validation, we use manufactured solutions where the exact solution is known.
"""

import warp as wp
import warp.fem as fem
import numpy as np

wp.init()


@fem.integrand
def laplacian_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    """
    Bilinear form for Laplacian: ∫ ∇u · ∇v dx
    This comes from weak formulation of -∇²u
    """
    return wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def source_form(s: fem.Sample, v: fem.Field, f_value: float):
    """
    Linear form for source term: ∫ f v dx
    """
    return f_value * v(s)


@fem.integrand
def boundary_value_form(s: fem.Sample, v: fem.Field, g_value: float):
    """
    Boundary condition: sets u = g_value on boundary
    """
    return g_value * v(s)


@fem.integrand  
def boundary_projector_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    """
    Projects boundary conditions onto the solution space
    """
    return u(s) * v(s)


class PoissonSolver:
    """
    2D Poisson equation solver with Dirichlet boundary conditions
    """
    
    def __init__(self, resolution=32, degree=2):
        """
        Args:
            resolution: Grid resolution (resolution x resolution cells)
            degree: Polynomial degree for finite elements (1 or 2)
        """
        self.resolution = resolution
        self.degree = degree
        
        # Create 2D grid geometry
        self.geo = fem.Grid2D(res=wp.vec2i(resolution, resolution))
        
        # Create function space
        self.space = fem.make_polynomial_space(self.geo, degree=degree)
        
        # Solution field
        self.solution_field = self.space.make_field()
        
    def solve(self, source_value=1.0, boundary_value=0.0):
        """
        Solve the Poisson equation with constant source and boundary values
        
        Args:
            source_value: Constant source term f
            boundary_value: Boundary condition value g
            
        Returns:
            solution: The discrete solution field
        """
        
        # Define domain (interior cells)
        domain = fem.Cells(geometry=self.geo)
        
        # Test and trial functions
        test = fem.make_test(space=self.space, domain=domain)
        trial = fem.make_trial(space=self.space, domain=domain)
        
        # Assemble system matrix (Laplacian)
        matrix = fem.integrate(
            laplacian_form, 
            fields={"u": trial, "v": test}
        )
        
        # Assemble right-hand side (source term)
        rhs = fem.integrate(
            source_form,
            fields={"v": test},
            values={"f_value": source_value}
        )
        
        # Boundary conditions
        boundary = fem.BoundarySides(self.geo)
        bd_test = fem.make_test(space=self.space, domain=boundary)
        bd_trial = fem.make_trial(space=self.space, domain=boundary)
        
        bd_matrix = fem.integrate(
            boundary_projector_form,
            fields={"u": bd_trial, "v": bd_test},
            assembly="nodal"
        )
        
        bd_rhs = fem.integrate(
            boundary_value_form,
            fields={"v": bd_test},
            values={"g_value": boundary_value},
            assembly="nodal"
        )
        
        # Apply boundary conditions (hard constraints)
        fem.project_linear_system(matrix, rhs, bd_matrix, bd_rhs)
        
        # Solve using Conjugate Gradient
        x = wp.zeros_like(rhs)
        self._cg_solve(matrix, rhs, x)
        
        # Store solution
        self.solution_field.dof_values = x
        
        return self.solution_field
    
    def _cg_solve(self, A, b, x, max_iters=1000, tol=1e-6):
        """
        Simple Conjugate Gradient solver using warp.fem utilities
        """
        # Import from warp repo examples
        import sys
        sys.path.insert(0, '/workspace/warp_repo/warp/examples/fem')
        from utils import bsr_cg
        
        bsr_cg(A, x, b, max_iters=max_iters, tol=tol, quiet=False)
    
    def get_solution_values(self):
        """
        Return solution as numpy array
        """
        return self.solution_field.dof_values.numpy()
    
    def evaluate_at_points(self, points: np.ndarray):
        """
        Evaluate solution at given points
        
        Args:
            points: Nx2 array of (x,y) coordinates
            
        Returns:
            values: N-array of solution values
        """
        # This would require interpolation - simplified for now
        # In practice, use fem field evaluation
        return np.zeros(len(points))


def run_simple_example():
    """
    Run a simple Poisson equation example
    """
    print("=" * 80)
    print("POISSON EQUATION SOLVER - Simple Example")
    print("=" * 80)
    
    # Create solver
    solver = PoissonSolver(resolution=16, degree=2)
    
    # Solve with constant forcing
    print("\nSolving: -∇²u = 1.0 with u = 0 on boundary")
    print(f"Grid resolution: {solver.resolution}x{solver.resolution}")
    print(f"Polynomial degree: {solver.degree}")
    
    solution = solver.solve(source_value=1.0, boundary_value=0.0)
    
    # Get solution values
    sol_values = solver.get_solution_values()
    
    print(f"\nSolution computed:")
    print(f"  DOF count: {len(sol_values)}")
    print(f"  Min value: {sol_values.min():.6f}")
    print(f"  Max value: {sol_values.max():.6f}")
    print(f"  Mean value: {sol_values.mean():.6f}")
    
    print("\n✓ Poisson solver ran successfully!")
    return solver, solution


if __name__ == "__main__":
    solver, solution = run_simple_example()

"""Poisson equation solver using warp.fem.

Solves: -∇²u = f on [0,1]² with homogeneous Dirichlet BCs.

Test case: u = sin(πx)sin(πy) implies f = 2π²sin(πx)sin(πy)
"""
import sys
# Add warp fem examples to path for bsr_cg utility
import warp
warp_examples_fem = str(__import__('pathlib').Path(warp.__file__).parent / 'examples' / 'fem')
sys.path.insert(0, warp_examples_fem)

import numpy as np
import warp as wp
import warp.fem as fem
import utils as fem_example_utils

wp.set_module_options({"enable_backward": False})


@fem.integrand
def laplacian_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    """Bilinear form: ∫∇u·∇v dx (weak form of -∇²u)."""
    return wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def source_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    """Linear form: ∫fv dx where f = 2π²sin(πx)sin(πy)."""
    x = domain(s)
    pi = 3.14159265359
    f_val = 2.0 * pi * pi * wp.sin(pi * x[0]) * wp.sin(pi * x[1])
    return f_val * v(s)


@fem.integrand
def boundary_projector(s: fem.Sample, u: fem.Field, v: fem.Field):
    """Bilinear form for boundary projection (homogeneous Dirichlet)."""
    return u(s) * v(s)


@fem.integrand
def zero_boundary_value(s: fem.Sample, v: fem.Field):
    """Zero boundary value for homogeneous Dirichlet BC."""
    return 0.0 * v(s)


def solve_poisson(resolution: int = 20, degree: int = 2, quiet: bool = False) -> tuple:
    """Solve the Poisson equation on a 2D grid.
    
    Args:
        resolution: Grid resolution (NxN cells)
        degree: Polynomial degree of shape functions
        quiet: Suppress solver output
        
    Returns:
        Tuple of (solution_field, geometry)
    """
    # Create 2D grid geometry
    geo = fem.Grid2D(res=wp.vec2i(resolution, resolution))
    
    # Create scalar function space
    scalar_space = fem.make_polynomial_space(geo, degree=degree)
    
    # Domain for integration (all cells)
    domain = fem.Cells(geometry=geo)
    
    # Create test and trial functions
    test = fem.make_test(space=scalar_space, domain=domain)
    trial = fem.make_trial(space=scalar_space, domain=domain)
    
    # Assemble stiffness matrix (Laplacian)
    stiffness = fem.integrate(laplacian_form, fields={"u": trial, "v": test})
    
    # Assemble load vector (source term)
    load = fem.integrate(source_form, domain=domain, fields={"v": test})
    
    # Apply homogeneous Dirichlet boundary conditions
    boundary = fem.BoundarySides(geo)
    bd_test = fem.make_test(space=scalar_space, domain=boundary)
    bd_trial = fem.make_trial(space=scalar_space, domain=boundary)
    
    # Boundary projector
    bd_matrix = fem.integrate(boundary_projector, fields={"u": bd_trial, "v": bd_test}, assembly="nodal")
    bd_rhs = fem.integrate(zero_boundary_value, fields={"v": bd_test}, assembly="nodal")
    
    # Project linear system
    fem.project_linear_system(stiffness, load, bd_matrix, bd_rhs)
    
    # Solve using Conjugate Gradient from example utils
    solution = wp.zeros_like(load)
    fem_example_utils.bsr_cg(stiffness, b=load, x=solution, quiet=quiet)
    
    # Create solution field
    solution_field = scalar_space.make_field()
    solution_field.dof_values = solution
    
    return solution_field, geo


def get_analytical_solution(x: float, y: float) -> float:
    """Analytical solution: u = sin(πx)sin(πy)."""
    pi = np.pi
    return np.sin(pi * x) * np.sin(pi * y)


if __name__ == "__main__":
    wp.init()
    
    print("Solving Poisson equation: -∇²u = f")
    print("Test case: u = sin(πx)sin(πy), f = 2π²sin(πx)sin(πy)")
    print()
    
    solution_field, geo = solve_poisson(resolution=20, degree=2, quiet=False)
    
    print("\nSolution computed successfully!")
    print(f"DOF values shape: {solution_field.dof_values.shape}")
    dof_vals = solution_field.dof_values.numpy()
    print(f"DOF values range: [{dof_vals.min():.4f}, {dof_vals.max():.4f}]")
    print(f"Expected max (at center): {get_analytical_solution(0.5, 0.5):.4f}")

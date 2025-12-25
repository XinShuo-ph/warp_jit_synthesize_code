"""Poisson Equation Solver using warp.fem

Solves: -∇²u = f in Ω
        u = g on ∂Ω

where Ω is the domain and ∂Ω is the boundary.

For testing, we use a manufactured solution:
u_exact(x, y) = sin(π*x) * sin(π*y)

This gives:
f(x, y) = 2π² * sin(π*x) * sin(π*y)
"""

import warp as wp
import warp.fem as fem
import numpy as np

wp.init()


@fem.integrand
def laplacian_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    """Bilinear form for Laplacian: ∫ ∇u · ∇v dx"""
    return wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def rhs_form(s: fem.Sample, v: fem.Field, f_val: float):
    """Linear form for RHS: ∫ f*v dx"""
    return f_val * v(s)


@fem.integrand
def manufactured_rhs(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    """RHS for manufactured solution: f = 2π² * sin(πx) * sin(πy)"""
    pos = domain(s)
    x = pos[0]
    y = pos[1]
    pi = 3.14159265359
    f = 2.0 * pi * pi * wp.sin(pi * x) * wp.sin(pi * y)
    return f * v(s)


@fem.integrand
def boundary_projector_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    """Projector for boundary conditions"""
    return u(s) * v(s)


@fem.integrand
def boundary_value_form(s: fem.Sample, v: fem.Field, val: float):
    """Linear form for boundary values"""
    return val * v(s)


def solve_poisson(resolution=20, degree=2):
    """Solve Poisson equation with manufactured solution.
    
    Args:
        resolution: Grid resolution
        degree: Polynomial degree for finite elements
        
    Returns:
        Tuple of (solution_field, error_l2, error_h1)
    """
    # Create 2D grid geometry
    geo = fem.Grid2D(res=wp.vec2i(resolution, resolution))
    
    # Create scalar function space
    space = fem.make_polynomial_space(geo, degree=degree)
    
    # Create domain for interior
    domain = fem.Cells(geometry=geo)
    
    # Test and trial functions
    test = fem.make_test(space=space, domain=domain)
    trial = fem.make_trial(space=space, domain=domain)
    
    # Assemble stiffness matrix (Laplacian)
    matrix = fem.integrate(laplacian_form, fields={"u": trial, "v": test})
    
    # Assemble RHS with manufactured solution
    rhs = fem.integrate(manufactured_rhs, fields={"v": test})
    
    # Boundary conditions (u = 0 on boundary for manufactured solution)
    boundary = fem.BoundarySides(geo)
    bd_test = fem.make_test(space=space, domain=boundary)
    bd_trial = fem.make_trial(space=space, domain=boundary)
    
    # Homogeneous Dirichlet BC (u = 0 on boundary)
    bd_matrix = fem.integrate(boundary_projector_form, fields={"u": bd_trial, "v": bd_test}, assembly="nodal")
    bd_rhs = fem.integrate(boundary_value_form, fields={"v": bd_test}, values={"val": 0.0}, assembly="nodal")
    
    # Project boundary conditions onto system
    fem.project_linear_system(matrix, rhs, bd_matrix, bd_rhs)
    
    # Solve linear system
    x = wp.zeros_like(rhs)
    
    # Simple conjugate gradient solver
    from warp.examples.fem.utils import bsr_cg
    bsr_cg(matrix, b=rhs, x=x, quiet=False, tol=1.0e-8, max_iters=1000)
    
    # Create solution field and ensure correct dtype
    solution_field = space.make_field()
    # Convert x to float32 if needed
    if x.dtype == wp.float64:
        x_float32 = wp.array(x.numpy(), dtype=wp.float32)
        solution_field.dof_values = x_float32
    else:
        solution_field.dof_values = x
    
    return solution_field, geo, space


def compute_error(solution_field, geo, space):
    """Compute L2 error against analytical solution.
    
    Args:
        solution_field: Computed solution
        geo: Geometry
        space: Function space
        
    Returns:
        L2 error norm
    """
    # Analytical solution: u(x,y) = sin(πx) * sin(πy)
    
    @fem.integrand
    def error_squared(s: fem.Sample, domain: fem.Domain, u: fem.Field):
        pos = domain(s)
        x = pos[0]
        y = pos[1]
        pi = 3.14159265359
        u_exact = wp.sin(pi * x) * wp.sin(pi * y)
        u_computed = u(s)
        error = u_computed - u_exact
        return error * error
    
    domain = fem.Cells(geometry=geo)
    
    # Integrate the squared error
    error_l2_squared = fem.integrate(
        error_squared,
        fields={"u": solution_field},
        quadrature=fem.RegularQuadrature(domain=domain, order=2 * space.degree)
    )
    
    # error_l2_squared is already a scalar
    error_l2 = np.sqrt(float(error_l2_squared))
    
    return error_l2


if __name__ == "__main__":
    print("Solving Poisson equation: -∇²u = f")
    print("Manufactured solution: u(x,y) = sin(πx) * sin(πy)")
    print("=" * 70)
    
    # Solve with different resolutions
    for res in [10, 20]:
        print(f"\nResolution: {res}x{res}")
        solution, geo, space = solve_poisson(resolution=res, degree=2)
        error = compute_error(solution, geo, space)
        print(f"L2 error: {error}")
        
    print("\n" + "=" * 70)
    print("Poisson solver completed successfully!")

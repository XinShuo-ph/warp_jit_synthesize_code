import warp as wp
import warp.fem as fem
import warp.optim.linear
import math

# Analytical solution: u = sin(pi*x)*sin(pi*y)
# f = 2*pi^2 * sin(pi*x)*sin(pi*y)

@fem.integrand
def linear_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    pos = fem.position(domain, s)
    x = pos[0]
    y = pos[1]
    pi = wp.float32(math.pi)
    f = 2.0 * pi * pi * wp.sin(pi * x) * wp.sin(pi * y)
    return f * v(s)

@fem.integrand
def diffusion_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(
        fem.grad(u, s),
        fem.grad(v, s),
    )

@fem.integrand
def boundary_projector(s: fem.Sample, u: fem.Field, v: fem.Field):
    return u(s) * v(s)

@fem.integrand
def boundary_value(s: fem.Sample, v: fem.Field):
    return 0.0 * v(s)

def solve_poisson(resolution=32, degree=2, quiet=False):
    wp.init()
    
    geo = fem.Grid2D(res=wp.vec2i(resolution), bounds_lo=wp.vec2(0.0, 0.0), bounds_hi=wp.vec2(1.0, 1.0))
    
    space = fem.make_polynomial_space(geo, degree=degree)
    domain = fem.Cells(geometry=geo)
    
    # Fields
    u = space.make_field() # Solution field (initialized to 0)
    test = fem.make_test(space=space, domain=domain)
    trial = fem.make_trial(space=space, domain=domain)
    
    # Assembly
    # domain arg is implicitly passed if requested in integrand signature
    rhs = fem.integrate(linear_form, fields={"v": test}, output_dtype=wp.float32)
    matrix = fem.integrate(diffusion_form, fields={"u": trial, "v": test}, output_dtype=wp.float32)
    
    # Boundary Conditions
    boundary = fem.BoundarySides(geo)
    bd_test = fem.make_test(space=space, domain=boundary)
    bd_trial = fem.make_trial(space=space, domain=boundary)
    
    bd_matrix = fem.integrate(boundary_projector, fields={"u": bd_trial, "v": bd_test}, assembly="nodal", output_dtype=wp.float32)
    bd_rhs = fem.integrate(boundary_value, fields={"v": bd_test}, assembly="nodal", output_dtype=wp.float32)
    
    # Project BCs
    fem.project_linear_system(matrix, rhs, bd_matrix, bd_rhs)
    
    # Solve
    # u.dof_values is the underlying array
    wp.optim.linear.cg(matrix, rhs, u.dof_values, maxiter=2000, tol=1e-6, use_cuda_graph=False)
    
    if not quiet:
        print(f"Solved with resolution {resolution}, degree {degree}")
    
    return u, geo, space

if __name__ == "__main__":
    solve_poisson()

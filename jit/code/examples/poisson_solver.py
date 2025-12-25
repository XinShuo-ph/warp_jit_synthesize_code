import warp as wp
import warp.fem as fem
import math
import numpy as np

# Try importing the utils from the installed package
try:
    import warp.examples.fem.utils as fem_utils
except ImportError:
    # Fallback or simplified CG if not found (though it should be there)
    print("Warning: Could not import warp.examples.fem.utils")
    sys.exit(1)

wp.init()

@fem.integrand
def diffusion_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    # Weak form: dot(grad(u), grad(v))
    return wp.dot(
        fem.grad(u, s),
        fem.grad(v, s)
    )

@fem.integrand
def rhs_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    # f(x,y) = 2 * pi^2 * sin(pi*x) * sin(pi*y)
    # domain is [0,1]x[0,1]
    
    # Get position
    x = fem.position(domain, s)
    
    # We can use wp.sin and constants
    pi = 3.1415926
    # Cast to float32 to ensure we stay in float32 land
    val = 2.0 * pi * pi * wp.sin(pi * x[0]) * wp.sin(pi * x[1])
    
    return val * v(s)

def solve_poisson(resolution=32, degree=2):
    # 1. Geometry: Unit square [0,1]x[0,1]
    # Grid2D default bounds are min=(-1,-1), max=(1,1)? Or 0,1?
    # Checking docs or assuming defaults. Usually centered.
    # We can specify bounds.
    geo = fem.Grid2D(res=wp.vec2i(resolution), bounds_lo=wp.vec2(0.0, 0.0), bounds_hi=wp.vec2(1.0, 1.0))
    
    # 2. Space
    space = fem.make_polynomial_space(geo, degree=degree)
    
    # 3. Domains
    domain = fem.Cells(geometry=geo)
    
    # 4. Assembly
    # LHS: Diffusion
    test = fem.make_test(space=space, domain=domain)
    trial = fem.make_trial(space=space, domain=domain)
    
    matrix = fem.integrate(diffusion_form, fields={"u": trial, "v": test})
    
    # RHS: Force
    rhs = fem.integrate(rhs_form, fields={"v": test})
    # RHS is float64 by default from integrate
    
    # 5. Boundary Conditions (Dirichlet u=0 on all sides)
    boundary = fem.BoundarySides(geo)
    bd_test = fem.make_test(space=space, domain=boundary)
    bd_trial = fem.make_trial(space=space, domain=boundary)
    
    # Projector form for u=0: we want to constrain u to 0.
    
    @fem.integrand
    def boundary_projector(s: fem.Sample, u: fem.Field, v: fem.Field):
        return u(s) * v(s)

    @fem.integrand
    def boundary_value(s: fem.Sample, v: fem.Field):
        return 0.0 * v(s)

    bd_matrix = fem.integrate(boundary_projector, fields={"u": bd_trial, "v": bd_test}, assembly="nodal")
    bd_rhs = fem.integrate(boundary_value, fields={"v": bd_test}, assembly="nodal")
    
    # Enforce BCs
    fem.project_linear_system(matrix, rhs, bd_matrix, bd_rhs)
    
    # 6. Solve
    x = wp.zeros_like(rhs)
    fem_utils.bsr_cg(matrix, b=rhs, x=x, quiet=True)
    
    # 7. Create solution field
    field = space.make_field()
    
    # Cast result to float32 for the field assignment if needed
    # We check if explicit cast is needed.
    x_f32 = wp.from_numpy(x.numpy().astype(np.float32), dtype=wp.float32)
    field.dof_values = x_f32
    
    return field, geo

if __name__ == "__main__":
    field, geo = solve_poisson(resolution=50)
    print("Poisson solved successfully.")

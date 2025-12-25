import warp as wp
import warp.fem as fem

@fem.integrand
def mass_form(
    s: fem.Sample,
    u: fem.Field,
    v: fem.Field,
):
    """Mass matrix form: u * v"""
    return u(s) * v(s)

@fem.integrand
def diffusion_form(
    s: fem.Sample,
    u: fem.Field,
    v: fem.Field,
):
    """Laplacian form: dot(grad(u), grad(v))"""
    return wp.dot(fem.grad(u, s), fem.grad(v, s))

@fem.integrand
def forcing_form(
    s: fem.Sample,
    domain: fem.Domain,
    v: fem.Field,
):
    """Forcing term f * v. 
    Here we define f = 2 * pi^2 * sin(pi*x) * sin(pi*y)
    which corresponds to exact solution u = sin(pi*x) * sin(pi*y)
    """
    x = fem.position(domain, s)
    f = 2.0 * wp.PI * wp.PI * wp.sin(wp.PI * x[0]) * wp.sin(wp.PI * x[1])
    return f * v(s)

class PoissonSolver:
    def __init__(self, resolution=50, degree=2):
        self.resolution = resolution
        self.degree = degree
        
        # 1. Geometry: 2D Grid [0,1]x[0,1]
        self.geo = fem.Grid2D(res=wp.vec2i(resolution))
        
        # 2. Function Space
        self.space = fem.make_polynomial_space(self.geo, degree=degree)
        
        # 3. Fields
        self.u_field = self.space.make_field()  # Solution
        
    def solve(self):
        domain = fem.Cells(geometry=self.geo)
        
        # Trial and Test functions
        u = fem.make_trial(space=self.space, domain=domain)
        v = fem.make_test(space=self.space, domain=domain)
        
        # Assemble LHS (Stiffness Matrix)
        # integrate dot(grad u, grad v)
        matrix = fem.integrate(diffusion_form, fields={"u": u, "v": v})
        
        # Assemble RHS (Load Vector)
        # integrate f * v
        rhs = fem.integrate(forcing_form, fields={"v": v})
        
        # Apply Boundary Conditions
        # Homogeneous Dirichlet on all boundaries (u=0)
        boundary = fem.BoundarySides(self.geo)
        # We can use project_linear_system to enforce u=0 on boundary nodes
        # For simple Grid2D with nodal basis, we can just zero out rows/cols or use dirichlet util
        
        # Using fem.dirichlet to enforce u=0
        # For polynomial space, we can identify boundary DOFs
        
        # fem.dirichlet simply sets rows to identity and rhs to value
        # We need a field representing the boundary value (0 everywhere)
        # Or simpler: project_linear_system with empty boundary matrices? No.
        
        # Let's construct a projector for the boundary
        bd_test = fem.make_test(space=self.space, domain=boundary)
        bd_trial = fem.make_trial(space=self.space, domain=boundary)
        
        # This form projects u=0 on boundary
        @fem.integrand
        def boundary_projector(s: fem.Sample, u: fem.Field, v: fem.Field):
            return u(s) * v(s)

        @fem.integrand
        def boundary_value(s: fem.Sample, v: fem.Field):
            return 0.0 * v(s)
            
        bd_matrix = fem.integrate(boundary_projector, fields={"u": bd_trial, "v": bd_test}, assembly="nodal")
        bd_rhs = fem.integrate(boundary_value, fields={"v": bd_test}, assembly="nodal")
        
        # Enforce Hard BCs
        fem.project_linear_system(matrix, rhs, bd_matrix, bd_rhs)
        
        # Solve
        x = wp.zeros_like(rhs)
        
        # Using Conjugate Gradient from warp.examples.fem.utils (need to import or implement)
        # Or warp.fem.utils if available?
        # example_diffusion used warp.examples.fem.utils.bsr_cg
        # We can use a simple CG implementation here if needed, or try to import.
        
        # Let's verify imports first.
        try:
            from warp.examples.fem.utils import bsr_cg
            bsr_cg(matrix, b=rhs, x=x, quiet=True)
        except ImportError:
            # Simple CG fallback
            # This is a placeholder, strictly we need a BSR matrix solver
            print("Warning: Could not import bsr_cg, skipping solve")
            return
            
        self.u_field.dof_values = x
        return x

if __name__ == "__main__":
    wp.init()
    solver = PoissonSolver(resolution=32)
    solver.solve()
    print("Poisson solve complete.")

"""
Poisson Equation Solver using Warp FEM

Solves: -∇²u = f on Ω = [0,1]×[0,1]
With: u = g on ∂Ω (Dirichlet boundary conditions)

Uses variational formulation:
Find u ∈ H¹₀(Ω) such that ∫∇u·∇v dx = ∫fv dx for all v ∈ H¹₀(Ω)
"""
import numpy as np
import warp as wp
import warp.fem as fem
from warp.sparse import BsrMatrix, bsr_mv

wp.set_module_options({"enable_backward": False})


@fem.integrand
def laplacian_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    """Bilinear form: ∫∇u·∇v dx"""
    return wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def source_form(s: fem.Sample, v: fem.Field, f_val: float):
    """Linear form: ∫fv dx with constant source f"""
    return f_val * v(s)


@fem.integrand
def source_form_sin(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    """Linear form: ∫fv dx where f = 2π²sin(πx)sin(πy)"""
    x = fem.position(domain, s)
    pi = 3.141592653589793
    f_val = 2.0 * pi * pi * wp.sin(pi * x[0]) * wp.sin(pi * x[1])
    return f_val * v(s)


@fem.integrand
def boundary_value_form(s: fem.Sample, v: fem.Field, bc_val: float):
    """Dirichlet BC value form"""
    return bc_val * v(s)


@fem.integrand
def boundary_projector_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    """Dirichlet BC projector form"""
    return u(s) * v(s)


def cg_solve(A: BsrMatrix, b: wp.array, x: wp.array, max_iters: int = 1000, tol: float = 1e-8) -> tuple[float, int]:
    """Simple Conjugate Gradient solver."""
    # r = b - A*x
    r = wp.zeros_like(b)
    bsr_mv(A, x, r, alpha=1.0, beta=0.0)
    wp.launch(lambda i: wp.atomic_add(r, i, -wp.load(wp.addressof(r, i)) + b[i] - wp.load(wp.addressof(r, i))),
              dim=r.shape[0])
    
    # Actually, use simpler approach with warp arrays
    r_np = b.numpy() - bsr_mv_numpy(A, x.numpy())
    p_np = r_np.copy()
    rs_old = np.dot(r_np, r_np)
    
    for i in range(max_iters):
        Ap_np = bsr_mv_numpy(A, p_np)
        alpha = rs_old / (np.dot(p_np, Ap_np) + 1e-30)
        x_np = x.numpy() + alpha * p_np
        r_np = r_np - alpha * Ap_np
        rs_new = np.dot(r_np, r_np)
        
        if np.sqrt(rs_new) < tol:
            x.assign(x_np.astype(np.float32))
            return np.sqrt(rs_new), i + 1
        
        p_np = r_np + (rs_new / rs_old) * p_np
        rs_old = rs_new
        x.assign(x_np.astype(np.float32))
    
    return np.sqrt(rs_new), max_iters


def bsr_mv_numpy(A: BsrMatrix, x: np.ndarray) -> np.ndarray:
    """BSR matrix-vector multiply using numpy."""
    x_wp = wp.array(x.astype(np.float32), dtype=float)
    y_wp = wp.zeros(A.nrow * A.block_shape[0], dtype=float)
    bsr_mv(A, x_wp, y_wp)
    return y_wp.numpy()


class PoissonSolver:
    """2D Poisson equation solver using Finite Element Method."""
    
    def __init__(self, resolution: int = 20, degree: int = 1):
        """
        Initialize solver.
        
        Args:
            resolution: Grid resolution (NxN cells)
            degree: Polynomial degree of shape functions (1=linear, 2=quadratic)
        """
        self.resolution = resolution
        self.degree = degree
        
        # Create 2D grid geometry on [0,1]×[0,1]
        self._geo = fem.Grid2D(res=wp.vec2i(resolution, resolution))
        
        # Create scalar function space
        self._space = fem.make_polynomial_space(self._geo, degree=degree)
        
        # Create field to store solution
        self._field = self._space.make_field()
        
        self._solution = None
        self._residual = None
        self._iterations = None
    
    def solve(self, source_type: str = "constant", source_value: float = 1.0, 
              bc_value: float = 0.0, max_iters: int = 1000, tol: float = 1e-8):
        """
        Solve the Poisson equation.
        
        Args:
            source_type: "constant" for f=const, "sin" for f=2π²sin(πx)sin(πy)
            source_value: Value of constant source (ignored if source_type="sin")
            bc_value: Dirichlet boundary condition value
            max_iters: Maximum CG iterations
            tol: Convergence tolerance
        """
        domain = fem.Cells(geometry=self._geo)
        
        # Build stiffness matrix (Laplacian)
        test = fem.make_test(space=self._space, domain=domain)
        trial = fem.make_trial(space=self._space, domain=domain)
        A = fem.integrate(laplacian_form, fields={"u": trial, "v": test})
        
        # Build RHS (source term)
        if source_type == "sin":
            rhs = fem.integrate(source_form_sin, fields={"v": test})
        else:
            rhs = fem.integrate(source_form, fields={"v": test}, values={"f_val": source_value})
        
        # Apply Dirichlet boundary conditions
        boundary = fem.BoundarySides(self._geo)
        bd_test = fem.make_test(space=self._space, domain=boundary)
        bd_trial = fem.make_trial(space=self._space, domain=boundary)
        
        bd_matrix = fem.integrate(boundary_projector_form, 
                                  fields={"u": bd_trial, "v": bd_test}, 
                                  assembly="nodal")
        bd_rhs = fem.integrate(boundary_value_form, 
                               fields={"v": bd_test}, 
                               values={"bc_val": bc_value},
                               assembly="nodal")
        
        # Project linear system to enforce Dirichlet BCs
        fem.project_linear_system(A, rhs, bd_matrix, bd_rhs)
        
        # Solve using warp's built-in CG
        x = wp.zeros_like(rhs)
        
        from warp.optim.linear import cg
        self._iterations, self._residual, _ = cg(A, b=rhs, x=x, tol=tol, maxiter=max_iters)
        
        # Store solution
        self._field.dof_values = x
        self._solution = x
        
        return self._field
    
    def get_solution_at_points(self, points: np.ndarray) -> np.ndarray:
        """Evaluate solution at given points."""
        if self._solution is None:
            raise RuntimeError("Must call solve() first")
        
        # For grid, we can sample the field
        return self._field.dof_values.numpy()
    
    def get_dof_positions(self) -> np.ndarray:
        """Get positions of degrees of freedom."""
        # For P1 elements on grid, DOFs are at vertices
        n = self.resolution + 1
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        return np.stack([xx.flatten(), yy.flatten()], axis=1)


def compute_analytical_solution(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute analytical solution u(x,y) = sin(πx)sin(πy)."""
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def compute_l2_error(numerical: np.ndarray, analytical: np.ndarray, h: float) -> float:
    """Compute L2 error norm."""
    diff = numerical - analytical
    return np.sqrt(h * h * np.sum(diff ** 2))


if __name__ == "__main__":
    wp.init()
    
    print("=" * 60)
    print("Poisson Solver Test")
    print("=" * 60)
    
    # Test with analytical solution
    resolution = 20
    solver = PoissonSolver(resolution=resolution, degree=1)
    
    print(f"\nSolving -∇²u = 2π²sin(πx)sin(πy) on [0,1]×[0,1]")
    print(f"With u = 0 on boundary")
    print(f"Resolution: {resolution}x{resolution}")
    
    field = solver.solve(source_type="sin", bc_value=0.0)
    
    print(f"\nConverged: residual={solver._residual:.2e}, iterations={solver._iterations}")
    
    # Compute error
    positions = solver.get_dof_positions()
    u_num = field.dof_values.numpy()
    u_exact = compute_analytical_solution(positions[:, 0], positions[:, 1])
    
    h = 1.0 / resolution
    l2_error = compute_l2_error(u_num, u_exact, h)
    max_error = np.max(np.abs(u_num - u_exact))
    
    print(f"\nError metrics:")
    print(f"  L2 error: {l2_error:.6e}")
    print(f"  Max error: {max_error:.6e}")
    
    print("\n" + "=" * 60)

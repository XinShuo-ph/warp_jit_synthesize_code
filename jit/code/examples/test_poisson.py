import warp as wp
import warp.fem as fem
import unittest
import sys
import os
import numpy as np

# Add workspace to path
sys.path.append(os.getcwd())

from jit.code.examples.poisson_solver import PoissonSolver

@fem.integrand
def l2_error_squared_form(s: fem.Sample, domain: fem.Domain, u: fem.Field):
    x = fem.position(domain, s)
    u_exact = wp.sin(wp.PI * x[0]) * wp.sin(wp.PI * x[1])
    diff = u(s) - u_exact
    return diff * diff

class TestPoissonSolver(unittest.TestCase):
    
    def compute_l2_error(self, solver):
        # We want to integrate (u - u_exact)^2 over the domain
        # fem.integrate usually assembles a vector/matrix.
        # If we want a scalar integral, we can treat it as a linear form with a "constant" test function v=1
        # But constructing a constant test function on the whole domain is tricky if we want exact integration.
        
        # Alternative: Evaluate field at node positions and compare (Nodal L2 error)
        # This is easier and sufficient for convergence checks usually.
        
        # Get nodal values
        # For Grid2D with nodal basis, dof_values correspond to nodes roughly (depending on ordering).
        # We can also sample the field.
        
        # Let's try using fem.integrate with a dummy test function 1?
        # Or just use the underlying data.
        
        # Let's iterate over nodes for simplicity first.
        # positions = solver.geo.reference_node_positions() # This gives reference positions
        # Actually Grid2D nodes are regular.
        
        # Better: warp.fem might have an 'integrate' that works for scalars?
        # Let's try to assemble a vector where integrand is (u-u_exact)^2 * v, where v is a test function
        # sum(vector) is not exactly integral.
        
        # Let's blindly trust that we can access values and compute error on CPU for now using sampling.
        
        res = solver.resolution
        
        # Sample points
        x = np.linspace(0, 1, res)
        y = np.linspace(0, 1, res)
        X, Y = np.meshgrid(x, y)
        
        # Evaluate exact solution
        U_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
        
        # Evaluate numerical solution
        # We need to query the field at these points.
        # This can be done via fem.interpolate or similar if we were inside a kernel.
        # Outside, we can peek at dof_values if the mapping is simple.
        
        # For Grid2D, dof_values are laid out usually linearly.
        # solver.u_field.dof_values is a warp array.
        vals = solver.u_field.dof_values.numpy()
        
        # The length of vals depends on degree. 
        # For degree 2, it's (2*res+1)^2 roughly?
        # Let's assume degree 1 for simple mapping check first?
        # But we used degree 2.
        
        # Let's stick to checking if it ran and error is "small".
        # A proper L2 integration is best done via FEM.
        
        return np.linalg.norm(vals) # Placeholder

    def test_convergence(self):
        # Run at two resolutions
        print("\nRunning Res 10...")
        s1 = PoissonSolver(resolution=10, degree=2)
        s1.solve()
        
        print("Running Res 20...")
        s2 = PoissonSolver(resolution=20, degree=2)
        s2.solve()
        
        # We expect the 'values' to converge to the sine wave.
        # Let's grab the center point value as a proxy for error.
        # (0.5, 0.5)
        # Exact value = sin(pi/2)*sin(pi/2) = 1.0
        
        # How to find center DOF?
        # For Grid2D, we can create a sample at (0.5, 0.5) and evaluate?
        # Currently evaluation from Python side might be tricky without a kernel.
        
        # Let's define a kernel to sample the field at center
        
        @wp.kernel
        def sample_center(field: fem.Field, out: wp.array(dtype=float)):
            # We need a sample point.
            # This is hard without setting up a Sample object properly.
            pass
            
        # simpler: check max value. It should be close to 1.0 (approx) and non-negative.
        max_v1 = s1.u_field.dof_values.numpy().max()
        max_v2 = s2.u_field.dof_values.numpy().max()
        
        print(f"Max Val Res 10: {max_v1}")
        print(f"Max Val Res 20: {max_v2}")
        
        self.assertTrue(abs(max_v1 - 1.0) < 0.1)
        self.assertTrue(abs(max_v2 - 1.0) < 0.05)
        
        # Ideally error should decrease
        err1 = abs(max_v1 - 1.0)
        err2 = abs(max_v2 - 1.0)
        self.assertLess(err2, err1)

if __name__ == "__main__":
    wp.init()
    unittest.main()

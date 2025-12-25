import warp as wp
import warp.fem as fem
import math
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from jit.code.examples.poisson_solver import solve_poisson

@fem.integrand
def error_form(s: fem.Sample, domain: fem.Domain, u: fem.Field):
    pos = fem.position(domain, s)
    x = pos[0]
    y = pos[1]
    pi = wp.float32(math.pi)
    u_exact = wp.sin(pi * x) * wp.sin(pi * y)
    u_approx = u(s)
    diff = u_approx - u_exact
    return diff * diff

def test_poisson_convergence():
    wp.init()
    
    # Run with increasing resolution and check error
    resolutions = [16, 32]
    errors = []
    
    for res in resolutions:
        u, geo, space = solve_poisson(resolution=res, degree=2, quiet=True)
        
        domain = fem.Cells(geometry=geo)
        
        # Integrate error
        # Note: output_dtype=wp.float32 to match u
        l2_sq_array = fem.integrate(error_form, domain=domain, fields={"u": u}, output_dtype=wp.float32)
        
        # fem.integrate returns a warp array for scalar result? Or scalar?
        # It usually returns a warp array of shape (1,) or similar for scalar reduction on GPU
        # If accumulate on CPU, it might return float.
        # But let's assume it returns a warp array or scalar.
        
        if hasattr(l2_sq_array, "numpy"):
            l2_sq = float(l2_sq_array.numpy()[0])
        else:
             # If it's a scalar (unlikely from integrate which usually returns array)
             l2_sq = float(l2_sq_array)
             
        error = math.sqrt(l2_sq)
        errors.append(error)
        print(f"Res {res}: L2 Error = {error}")
        
    # Check convergence
    # Error should decrease.
    if errors[1] < errors[0]:
        print("Convergence test passed.")
    else:
        print(f"Convergence test FAILED: {errors[1]} >= {errors[0]}")
        sys.exit(1)

if __name__ == "__main__":
    test_poisson_convergence()

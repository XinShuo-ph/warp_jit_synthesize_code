import unittest
import warp as wp
import warp.fem as fem
from poisson_solver import solve_poisson

wp.init()

@fem.integrand
def l2_error_form(s: fem.Sample, domain: fem.Domain, u: fem.Field):
    x = fem.position(domain, s)
    pi = 3.1415926
    
    u_exact = wp.sin(pi * x[0]) * wp.sin(pi * x[1])
    u_num = u(s)
    
    diff = u_num - u_exact
    return diff * diff

class TestPoisson(unittest.TestCase):
    def test_convergence(self):
        # Run solver with low resolution
        field_lo, geo_lo = solve_poisson(resolution=16, degree=2)
        
        # Calculate L2 error
        domain_lo = fem.Cells(geometry=geo_lo)
        err_sq_lo = fem.integrate(l2_error_form, fields={"u": field_lo}, domain=domain_lo)
        l2_error_lo = math.sqrt(err_sq_lo)
        
        print(f"L2 Error (N=16): {l2_error_lo}")
        
        # Run solver with higher resolution
        field_hi, geo_hi = solve_poisson(resolution=32, degree=2)
        
        domain_hi = fem.Cells(geometry=geo_hi)
        err_sq_hi = fem.integrate(l2_error_form, fields={"u": field_hi}, domain=domain_hi)
        l2_error_hi = math.sqrt(err_sq_hi)
        
        print(f"L2 Error (N=32): {l2_error_hi}")
        
        # Check if error decreased significantly (expect roughly 1/4 or better for 2nd order)
        self.assertLess(l2_error_hi, l2_error_lo * 0.5)
        self.assertLess(l2_error_hi, 0.01) # Absolute check

import math

if __name__ == "__main__":
    unittest.main()

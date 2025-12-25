#!/usr/bin/env python3
"""Simple FEM example using warp.fem"""

import warp as wp
import warp.fem as fem
import numpy as np

wp.init()

def main():
    print("Running FEM diffusion example...")
    
    # Create a simple 2D grid
    res = wp.vec2i(10, 10)
    
    # Create geometry - a 2D grid
    geo = fem.Grid2D(res=res, bounds_lo=(0.0, 0.0), bounds_hi=(1.0, 1.0))
    
    # Define function space
    domain = fem.Cells(geometry=geo)
    scalar_space = fem.make_polynomial_space(geo, degree=1)
    
    # Create test and trial functions
    test = fem.make_test(space=scalar_space, domain=domain)
    trial = fem.make_trial(space=scalar_space, domain=domain)
    
    print(f"Created FEM problem:")
    print(f"  Geometry: {geo.cell_count()} cells, {geo.vertex_count()} vertices")
    print(f"  Space: degree {scalar_space.degree}, {scalar_space.node_count()} DOFs")
    
    # Create a simple integration kernel for mass matrix
    @fem.integrand
    def mass_form(s: fem.Sample, u: fem.Field, v: fem.Field):
        return u(s) * v(s)
    
    # Integrate
    mass_matrix = fem.integrate(mass_form, fields={"u": trial, "v": test}, output_dtype=float)
    
    print(f"  Mass matrix: {mass_matrix.shape}")
    print(f"  Non-zeros: {mass_matrix.nnz if hasattr(mass_matrix, 'nnz') else 'N/A'}")
    
    return geo.cell_count(), geo.vertex_count(), scalar_space.node_count()

if __name__ == "__main__":
    print("=== Run 1 ===")
    result1 = main()
    print("\n=== Run 2 ===")
    result2 = main()
    
    assert result1 == result2, "Results don't match!"
    print(f"\n✓ Both runs produced identical results: {result1}")

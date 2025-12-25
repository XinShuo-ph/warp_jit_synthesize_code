"""Simple FEM diffusion example to test warp.fem module."""

import warp as wp
import warp.fem as fem

wp.init()

# Create a simple 2D grid geometry
geo = fem.Grid2D(res=wp.vec2i(10, 10))

# Create function space (linear elements)
scalar_space = fem.make_polynomial_space(geo, degree=1)

print(f"Geometry: {type(geo).__name__}")
print(f"Function space: {type(scalar_space).__name__}")
print(f"Number of cells: {geo.cell_count()}")

# Define a simple integrand
@fem.integrand
def linear_form(s: fem.Sample, v: fem.Field):
    return v(s)

# Create field for testing
test_field = scalar_space.make_field()

print("\nFEM example initialized successfully!")
print("This demonstrates warp.fem module basics for M3 preparation.")

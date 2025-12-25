from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from typing import Literal

import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem


@fem.integrand
def rhs_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    x = fem.position(domain, s)
    u = wp.sin(pi * x[0]) * wp.sin(pi * x[1])
    f = 2.0 * (pi * pi) * u  # -Δu = f
    return f * v(s)


@fem.integrand
def poisson_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def dirichlet_projector_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return u(s) * v(s)


@fem.integrand
def dirichlet_value_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    # u_exact = sin(pi x) sin(pi y) is 0 on boundary of [0,1]^2
    return 0.0 * v(s)


@fem.integrand
def l2_error_form(s: fem.Sample, domain: fem.Domain, u_h: fem.Field):
    x = fem.position(domain, s)
    u = wp.sin(pi * x[0]) * wp.sin(pi * x[1])
    e = u_h(s) - u
    return e * e


@fem.integrand
def exact_solution(domain: fem.Domain, s: fem.Sample):
    x = domain(s)
    return wp.sin(pi * x[0]) * wp.sin(pi * x[1])


@dataclass(frozen=True)
class PoissonResult:
    field: fem.DiscreteField
    l2_error: float
    max_nodal_error: float
    resolution: int
    degree: int
    device: str


def solve_poisson_unit_square(
    *,
    resolution: int = 32,
    degree: int = 1,
    mesh: Literal["grid", "tri", "quad"] = "grid",
    device: str = "cpu",
    tol: float = 1.0e-12,
    quiet: bool = True,
) -> PoissonResult:
    """Solve -Δu = f on [0,1]^2 with Dirichlet u=0 using Warp FEM.

    Analytic solution: u(x,y) = sin(pi x) sin(pi y).
    """
    wp.set_module_options({"enable_backward": False})

    # Reduce Warp's compilation/logging noise for tests and batch runs.
    try:
        import warp._src.config as _wp_config  # noqa: PLC0415

        _wp_config.quiet = quiet
        _wp_config.verbose = False
    except Exception:
        pass

    wp.init()

    with wp.ScopedDevice(device):
        if mesh == "tri":
            positions, tri_vidx = fem_example_utils.gen_trimesh(res=wp.vec2i(resolution))
            geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions, build_bvh=False)
        elif mesh == "quad":
            positions, quad_vidx = fem_example_utils.gen_quadmesh(res=wp.vec2i(resolution))
            geo = fem.Quadmesh2D(quad_vertex_indices=quad_vidx, positions=positions, build_bvh=False)
        else:
            geo = fem.Grid2D(res=wp.vec2i(resolution))

        domain = fem.Cells(geometry=geo)
        space = fem.make_polynomial_space(geo, degree=degree)

        test = fem.make_test(space=space, domain=domain)
        trial = fem.make_trial(space=space, domain=domain)

        rhs = fem.integrate(rhs_form, fields={"v": test}, output_dtype=float)
        matrix = fem.integrate(poisson_form, fields={"u": trial, "v": test}, output_dtype=float)

        # Hard Dirichlet boundary conditions u=0 on all boundary sides
        boundary = fem.BoundarySides(geo)
        bd_test = fem.make_test(space=space, domain=boundary)
        bd_trial = fem.make_trial(space=space, domain=boundary)

        bd_matrix = fem.integrate(
            dirichlet_projector_form, fields={"u": bd_trial, "v": bd_test}, assembly="nodal", output_dtype=float
        )
        bd_rhs = fem.integrate(dirichlet_value_form, fields={"v": bd_test}, assembly="nodal", output_dtype=float)
        fem.project_linear_system(matrix, rhs, bd_matrix, bd_rhs)

        x = wp.zeros_like(rhs)
        fem_example_utils.bsr_cg(matrix, b=rhs, x=x, quiet=True, tol=tol)

        field = space.make_field()
        field.dof_values = x

        l2_err_sq = fem.integrate(l2_error_form, fields={"u_h": field}, domain=domain)
        l2_error = float(sqrt(float(l2_err_sq)))

        # Nodal max error (safe for degree-1 spaces; used as a simple check)
        dof_np = field.dof_values.numpy()
        exact = space.make_field()
        fem.interpolate(exact_solution, dest=exact)
        exact_np = exact.dof_values.numpy()
        max_nodal_error = float(abs(dof_np - exact_np).max())

        return PoissonResult(
            field=field,
            l2_error=l2_error,
            max_nodal_error=max_nodal_error,
            resolution=resolution,
            degree=degree,
            device=str(wp.get_device()),
        )


def main() -> None:
    res = solve_poisson_unit_square(resolution=32, degree=1, device="cpu", quiet=False)
    print(
        f"poisson: resolution={res.resolution} degree={res.degree} "
        f"l2_error={res.l2_error:.3e} max_nodal_error={res.max_nodal_error:.3e} device={res.device}"
    )


if __name__ == "__main__":
    main()


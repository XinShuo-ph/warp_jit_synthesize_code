from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem


@wp.func
def u_exact(pos: wp.vec2) -> float:
    return wp.sin(wp.pi * pos[0]) * wp.sin(wp.pi * pos[1])


@wp.func
def f_forcing(pos: wp.vec2) -> float:
    # For u=sin(pi x) sin(pi y):  -Δu = 2*pi^2*sin(pi x) sin(pi y)
    return 2.0 * wp.pi * wp.pi * u_exact(pos)


@fem.integrand
def stiffness_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def rhs_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    pos = domain(s)
    return f_forcing(pos) * v(s)


@fem.integrand
def boundary_projector_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    # Nodal boundary "mass" used to build a Dirichlet projector.
    return u(s) * v(s)


@fem.integrand
def boundary_value_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    pos = domain(s)
    return u_exact(pos) * v(s)


@dataclass(frozen=True)
class PoissonResult:
    field: fem.DiscreteField
    l2_error: float
    linf_error: float


def solve_poisson(
    *,
    resolution: int = 32,
    degree: int = 2,
    device: str = "cpu",
    quiet: bool = True,
) -> PoissonResult:
    wp.set_module_options({"enable_backward": False})

    with wp.ScopedDevice(device):
        geo = fem.Grid2D(res=wp.vec2i(resolution))
        domain = fem.Cells(geometry=geo)

        space = fem.make_polynomial_space(geo, degree=degree)
        field = space.make_field()

        test = fem.make_test(space=space, domain=domain)
        trial = fem.make_trial(space=space, domain=domain)

        matrix = fem.integrate(stiffness_form, fields={"u": trial, "v": test}, output_dtype=float)
        rhs = fem.integrate(rhs_form, fields={"v": test}, output_dtype=float)

        # Full Dirichlet boundary from analytic solution (here: homogeneous).
        boundary = fem.BoundarySides(geo)
        bd_test = fem.make_test(space=space, domain=boundary)
        bd_trial = fem.make_trial(space=space, domain=boundary)
        bd_matrix = fem.integrate(
            boundary_projector_form,
            fields={"u": bd_trial, "v": bd_test},
            assembly="nodal",
            output_dtype=float,
        )
        bd_value = fem.integrate(
            boundary_value_form,
            fields={"v": bd_test},
            assembly="nodal",
            output_dtype=float,
        )

        fem.project_linear_system(matrix, rhs, bd_matrix, bd_value)

        x = wp.zeros_like(rhs)
        fem_example_utils.bsr_cg(matrix, b=rhs, x=x, quiet=quiet, tol=1.0e-10)
        field.dof_values = x

        # Evaluate error at nodes.
        node_pos = field.space.node_positions().numpy()
        u_num = field.dof_values.numpy()
        u_ref = np.sin(np.pi * node_pos[:, 0]) * np.sin(np.pi * node_pos[:, 1])
        diff = u_num - u_ref
        l2 = float(np.sqrt(np.mean(diff * diff)))
        linf = float(np.max(np.abs(diff)))

        return PoissonResult(field=field, l2_error=l2, linf_error=linf)


if __name__ == "__main__":
    wp.init()
    r = solve_poisson(resolution=32, degree=2, device="cpu", quiet=True)
    print(f"L2 error: {r.l2_error:.3e}  Linf error: {r.linf_error:.3e}")


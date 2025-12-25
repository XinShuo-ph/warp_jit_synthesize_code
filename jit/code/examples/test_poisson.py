from __future__ import annotations

from jit.code.examples.poisson_solver import solve_poisson_unit_square


def test_error_decreases_with_refinement() -> None:
    r1 = solve_poisson_unit_square(resolution=16, degree=1, mesh="grid", device="cpu", tol=1.0e-12, quiet=True)
    r2 = solve_poisson_unit_square(resolution=32, degree=1, mesh="grid", device="cpu", tol=1.0e-12, quiet=True)

    # Expect improvement with refinement; keep slack since CG tolerance and basis choice can vary.
    assert r2.l2_error < r1.l2_error * 0.75, (r1.l2_error, r2.l2_error)
    assert r2.max_nodal_error < r1.max_nodal_error * 0.75, (r1.max_nodal_error, r2.max_nodal_error)


def main() -> None:
    # Minimal runner so we can satisfy the "run twice" protocol without a test framework.
    test_error_decreases_with_refinement()


if __name__ == "__main__":
    main()


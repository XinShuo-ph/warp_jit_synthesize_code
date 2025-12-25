from __future__ import annotations

import pytest
import warp as wp

from jit.code.examples.poisson_solver import solve_poisson


@pytest.fixture(scope="session", autouse=True)
def _init_warp():
    wp.init()


def test_poisson_error_decreases_with_resolution():
    r_coarse = solve_poisson(resolution=8, degree=2, device="cpu", quiet=True)
    r_fine = solve_poisson(resolution=16, degree=2, device="cpu", quiet=True)

    assert r_fine.l2_error < r_coarse.l2_error
    assert r_fine.linf_error < r_coarse.linf_error


def test_poisson_reasonable_accuracy():
    r = solve_poisson(resolution=16, degree=2, device="cpu", quiet=True)

    assert r.l2_error < 5.0e-5
    assert r.linf_error < 2.0e-4


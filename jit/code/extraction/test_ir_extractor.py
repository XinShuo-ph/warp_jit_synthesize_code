from __future__ import annotations

import hashlib

import pytest
import warp as wp

from jit.code.extraction import fixture_kernels as fk
from jit.code.extraction.ir_extractor import extract_ir


@pytest.fixture(scope="session", autouse=True)
def _init_warp():
    wp.init()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.mark.parametrize(
    "kernel",
    [
        fk.add_constant,
        fk.conditional_scale,
        fk.struct_math,
        fk.atomic_accumulate,
        fk.trig_mix,
    ],
)
def test_extract_ir_contains_symbol_and_is_stable(kernel: wp.Kernel):
    ir1 = extract_ir(kernel, device="cpu")
    ir2 = extract_ir(kernel, device="cpu")

    mangled = kernel.get_mangled_name()
    assert mangled in ir1
    assert _sha256(ir1) == _sha256(ir2)


"""Offline Warp codegen helpers (no device required).

This module generates CPU/CUDA source via Warp's internal codegen APIs without
loading or launching kernels. This allows producing CUDA `.cu` sources even on
machines without an NVIDIA GPU/driver.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from code.common.device import normalize_warp_target


@dataclass(frozen=True)
class CodegenResult:
    target: str  # "cpu" | "cuda"
    module_id: str
    source: str


def codegen_module_source(module: wp.context.Module, *, target: str, enable_backward: bool = True) -> CodegenResult:
    """Generate full module source for the given target via internal codegen."""
    import warp._src.context as ctx

    tgt = normalize_warp_target(target)

    # Create options dict
    options = module.options.copy() if module.options else {}
    options.setdefault("block_dim", 256)
    options.setdefault("enable_backward", enable_backward)
    options.setdefault("mode", "release")

    hasher = ctx.ModuleHasher(module)
    builder = ctx.ModuleBuilder(module, options, hasher)
    source = builder.codegen(tgt)

    # Stable identifier for metadata; does not require compilation.
    module_id = module.get_module_identifier()

    return CodegenResult(target=tgt, module_id=module_id, source=source)


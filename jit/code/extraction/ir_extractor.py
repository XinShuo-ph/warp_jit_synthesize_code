from __future__ import annotations

from typing import Any

import warp as wp


def extract_ir(kernel: Any, device: str = "cpu") -> str:
    """Extract a deterministic JIT 'IR' artifact for a Warp kernel.

    Current implementation returns the generated CPU/CUDA source emitted by Warp's
    `ModuleBuilder.codegen(...)` for the kernel's owning module. This is the primary
    intermediate representation used by Warp's JIT compilation pipeline.
    """

    if device not in ("cpu", "cuda"):
        raise ValueError(f"Unsupported device '{device}', expected 'cpu' or 'cuda'")

    if device == "cuda" and not wp.is_cuda_available():
        raise RuntimeError("CUDA requested but Warp CUDA is not available in this environment")

    wp.init()

    kernel_mod = getattr(kernel, "__module__", None)
    if kernel_mod is None and hasattr(kernel, "func"):
        kernel_mod = getattr(kernel.func, "__module__", None)
    if kernel_mod is None:
        raise TypeError("Could not determine kernel module (__module__) for IR extraction")

    module = wp.get_module(kernel_mod)

    # Import from private API: stable enough for internal tooling, but may change across Warp versions.
    import warp._src.context as wpc  # noqa: PLC0415

    builder_options = dict(module.options)
    if device == "cpu":
        builder_options["output_arch"] = None
    else:
        builder_options["output_arch"] = module.get_compile_arch(wp.get_device())

    builder = wpc.ModuleBuilder(
        module,
        builder_options,
        hasher=module.hashers.get(module.options["block_dim"], None),
    )

    return builder.codegen(device)


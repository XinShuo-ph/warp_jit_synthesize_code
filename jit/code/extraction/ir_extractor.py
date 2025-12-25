from __future__ import annotations

from typing import Final

import warp as wp


SUPPORTED_DEVICES: Final[set[str]] = {"cpu"}


def extract_ir(kernel: wp.Kernel, device: str = "cpu") -> str:
    """Extract a stable, string-serializable IR-like representation for a Warp kernel.

    Current implementation returns Warp's generated C++ source for the kernel's owning module.
    """
    if device not in SUPPORTED_DEVICES:
        raise NotImplementedError(f"Only {sorted(SUPPORTED_DEVICES)} supported, got {device!r}")

    # The codegen happens at the module level; the returned source should include this kernel's mangled symbol.
    module = kernel.module

    # Mirror Warp's CPU build path: output_arch=None (CPU) and use module options.
    builder_options = dict(module.options)
    builder_options["output_arch"] = None

    # Internal API, but it gives direct access to the generated source string.
    from warp._src.context import ModuleBuilder  # noqa: PLC0415

    builder = ModuleBuilder(module, builder_options, hasher=module.hashers.get(module.options["block_dim"], None))
    source = builder.codegen("cpu")

    if not source:
        raise RuntimeError(f"Empty IR extracted for kernel {kernel.key!r} on device {device!r}")

    return source


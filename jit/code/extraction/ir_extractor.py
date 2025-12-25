from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import warp as wp


@dataclass(frozen=True)
class IRExtractionResult:
    device: str
    codegen_device: str
    kernel_key: str
    mangled_name: str
    module_name: str
    module_hash: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_ir(kernel: Any, device: str = "cpu") -> IRExtractionResult:
    """Extract Warp's generated code (JIT intermediate) for a kernel.

    Note: Warp's JIT pipeline generates C++ (CPU) / CUDA C++ (GPU) source code
    via `warp._src.context.ModuleBuilder.codegen()`. This is the "IR" we return.
    """
    wp.init()

    # Internal imports are required to reach ModuleBuilder/runtime device resolution.
    import warp._src.context as ctx  # noqa: PLC0415

    if not hasattr(kernel, "module") or not hasattr(kernel, "get_mangled_name"):
        raise TypeError("extract_ir() expects a Warp kernel object (from @wp.kernel).")

    dev = ctx.runtime.get_device(device)
    codegen_device = "cuda" if dev.is_cuda else "cpu"

    module = kernel.module

    output_arch = None if not dev.is_cuda else module.get_compile_arch(dev)
    builder_options = {**module.options, "output_arch": output_arch}

    builder = ctx.ModuleBuilder(
        module,
        builder_options,
        hasher=module.hashers.get(module.options["block_dim"], None),
    )

    source = builder.codegen(codegen_device)
    module_hash = module.get_module_hash().hex()

    return IRExtractionResult(
        device=str(dev),
        codegen_device=codegen_device,
        kernel_key=getattr(kernel, "key", getattr(kernel.func, "__name__", "unknown")),
        mangled_name=kernel.get_mangled_name(),
        module_name=getattr(module, "name", "unknown"),
        module_hash=module_hash,
        source=source,
    )


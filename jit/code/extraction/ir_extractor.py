"""IR Extractor: Extract Python source and generated C++/CUDA code from Warp kernels."""

from __future__ import annotations

import re
from typing import Any

import warp as wp
import warp._src.context


def extract_ir(kernel, device: str = "cpu", include_backward: bool = True) -> dict[str, Any]:
    """
    Extract generated code from a Warp kernel.

    Notes:
    - In this project, we treat Warp's generated source (C++ for CPU, CUDA C++ for CUDA)
      as an "IR-like" intermediate artifact emitted by the JIT/codegen pipeline.
    - This function does **not** require running the kernel; it uses Warp's ModuleBuilder
      to generate source text.
    """
    if device not in {"cpu", "cuda"}:
        raise ValueError('device must be "cpu" or "cuda"')

    module = kernel.module
    hasher = warp._src.context.ModuleHasher(module)

    options = module.options.copy() if module.options else {}
    options.setdefault("block_dim", 256)
    options.setdefault("enable_backward", include_backward)
    options.setdefault("mode", "release")

    builder = warp._src.context.ModuleBuilder(module, options, hasher)
    code = builder.codegen(device)

    python_source = kernel.adj.source
    kernel_name = kernel.key
    mangled_name = kernel.get_mangled_name()

    forward_code = _extract_function(code, f"{mangled_name}_{device}_kernel_forward")

    backward_code = None
    if include_backward:
        backward_code = _extract_function(code, f"{mangled_name}_{device}_kernel_backward")

    metadata = {
        "kernel_name": kernel_name,
        "mangled_name": mangled_name,
        "device": device,
        "arg_names": list(kernel.adj.arg_types.keys()),
        "arg_types": {k: str(v) for k, v in kernel.adj.arg_types.items()},
        "has_backward": backward_code is not None,
    }

    return {
        "python_source": python_source,
        "code": code,
        "kernel_name": kernel_name,
        "forward_code": forward_code,
        "backward_code": backward_code,
        "metadata": metadata,
    }


def _extract_function(code: str, func_name: str) -> str | None:
    """Extract a single function from the generated code."""
    pattern = rf"void {re.escape(func_name)}\s*\([^)]*\)\s*\{{"
    match = re.search(pattern, code)
    if not match:
        return None

    start = match.start()

    brace_count = 0
    in_function = False
    end = start

    for i, ch in enumerate(code[start:], start):
        if ch == "{":
            brace_count += 1
            in_function = True
        elif ch == "}":
            brace_count -= 1
            if in_function and brace_count == 0:
                end = i + 1
                break

    return code[start:end]


def extract_python_codegen_pair(kernel, device: str = "cpu") -> tuple[str, str]:
    """Return (python_source, forward_generated_code) for a single kernel."""
    result = extract_ir(kernel, device=device, include_backward=False)
    return result["python_source"], result["forward_code"]


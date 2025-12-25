from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class IRArtifact:
    """A resolved on-disk IR artifact produced by Warp's JIT."""

    kind: str  # "cpp" | "cu" | "ptx" | "cubin" | "meta"
    path: str
    module_id: str
    module_dir: str
    device: str


def _get_warp_module_for_kernel(kernel) -> "object":
    import warp as wp
    from warp._src import context as wp_context

    # Warp kernels are grouped by their Python module.
    module_name = getattr(getattr(kernel, "func", None), "__module__", None)
    if not module_name:
        raise TypeError(f"Unsupported kernel object (missing kernel.func.__module__): {type(kernel)}")

    # Ensure Warp is initialized and cache dir is configured.
    wp.init()

    return wp_context.get_module(module_name)


def _device_to_string(device: object | str | None) -> str:
    if device is None:
        return "cpu"
    if isinstance(device, str):
        return device
    # Warp Device objects print nicely as aliases ("cpu", "cuda:0", ...)
    return str(device)


def _candidate_paths(
    module_id: str,
    module_dir: str,
    prefer: Sequence[str],
    output_arch: Optional[int] = None,
) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for kind in prefer:
        if kind == "cpp":
            candidates.append((kind, os.path.join(module_dir, f"{module_id}.cpp")))
        elif kind == "cu":
            candidates.append((kind, os.path.join(module_dir, f"{module_id}.cu")))
        elif kind == "ptx":
            # Warp's CUDA output name includes architecture suffix.
            if output_arch is not None:
                candidates.append((kind, os.path.join(module_dir, f"{module_id}.sm{output_arch}.ptx")))
        elif kind == "cubin":
            if output_arch is not None:
                candidates.append((kind, os.path.join(module_dir, f"{module_id}.sm{output_arch}.cubin")))
        elif kind == "meta":
            candidates.append((kind, os.path.join(module_dir, f"{module_id}.meta")))
        else:
            raise ValueError(f"Unknown IR kind '{kind}'")
    return candidates


def extract_ir_artifact(
    kernel,
    device: str | None = "cpu",
    prefer: Sequence[str] = ("cpp", "cu", "ptx"),
) -> IRArtifact:
    """Resolve a kernel's cached IR artifact and return its path + metadata.

    Notes:
    - This function **forces a module load** on the requested device, which triggers
      codegen/compile on a cache miss.
    - On CPU-only environments, the default and most reliable artifact is the generated
      C++ source: `<module_id>.cpp` in Warp's kernel cache directory.
    """

    import warp as wp

    wp.init()
    dev = wp.get_device(device) if device is not None else wp.get_device("cpu")

    module = _get_warp_module_for_kernel(kernel)

    # Ensure the module is built/loaded for the device so cache artifacts exist.
    module.load(dev)

    module_id = module.get_module_identifier()
    cache_dir = wp.config.kernel_cache_dir
    if not cache_dir:
        raise RuntimeError("warp.config.kernel_cache_dir is not set (Warp not initialized?)")

    module_dir = os.path.join(cache_dir, module_id)

    output_arch = None
    try:
        output_arch = module.get_compile_arch(dev)
    except Exception:
        # CPU path will return None; keep as None.
        output_arch = None

    candidates = _candidate_paths(module_id, module_dir, prefer=prefer, output_arch=output_arch)
    tried = [p for _k, p in candidates]

    for kind, path in candidates:
        if os.path.exists(path):
            return IRArtifact(kind=kind, path=path, module_id=module_id, module_dir=module_dir, device=_device_to_string(dev))

    raise FileNotFoundError(
        "No IR artifact found for kernel. "
        f"Module '{module_id}' cache dir '{module_dir}'. Tried: {', '.join(tried)}"
    )


def extract_ir(
    kernel,
    device: str | None = "cpu",
    prefer: Sequence[str] = ("cpp", "cu", "ptx"),
) -> str:
    """Extract a kernel's IR text from Warp's kernel cache.

    Returns the contents of the first matching artifact in `prefer`.
    """

    artifact = extract_ir_artifact(kernel=kernel, device=device, prefer=prefer)

    # Only text artifacts are supported here.
    if artifact.kind in ("cubin",):
        raise ValueError(f"Binary artifact '{artifact.kind}' is not readable as text: {artifact.path}")

    with open(artifact.path, "r", encoding="utf-8", errors="replace") as f:
        data = f.read()

    if not data.strip():
        raise RuntimeError(f"Extracted IR file is empty: {artifact.path}")

    return data


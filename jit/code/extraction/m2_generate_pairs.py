from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import warp as wp

from jit.code.extraction.ir_extractor import extract_ir


@wp.kernel(module="unique")
def k_add(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = a[i] + b[i]


@wp.kernel(module="unique")
def k_saxpy(a: float, x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = a * x[i] + y[i]


@wp.kernel(module="unique")
def k_clamp(x: wp.array(dtype=wp.float32), lo: float, hi: float, out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    v = x[i]
    out[i] = wp.clamp(v, lo, hi)


@wp.kernel(module="unique")
def k_where(x: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = wp.where(x[i] > 0.0, x[i], -x[i])


@wp.kernel(module="unique")
def k_sin_cos(x: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = wp.sin(x[i]) + wp.cos(x[i])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    wp.init()

    out_path = Path(__file__).resolve().parents[2] / "data" / "samples" / "m2_pairs.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kernels = [k_add, k_saxpy, k_clamp, k_where, k_sin_cos]

    records: list[dict] = []
    for k in kernels:
        py_src = inspect.getsource(k.func)
        ir = extract_ir(k, device="cpu")
        records.append(
            {
                "name": k.key,
                "python": py_src,
                "ir": ir.source,
                "meta": {
                    "warp_version": getattr(wp, "__version__", "unknown"),
                    "device": ir.device,
                    "codegen_device": ir.codegen_device,
                    "mangled_name": ir.mangled_name,
                    "module_name": ir.module_name,
                    "module_hash": ir.module_hash,
                },
            }
        )

    # Deterministic order + formatting for stable reruns.
    records.sort(key=lambda r: r["name"])
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")

    print(f"wrote {len(records)} pairs -> {out_path}")
    print(f"sha256 {out_path.name}: {_sha256(out_path)}")


if __name__ == "__main__":
    main()


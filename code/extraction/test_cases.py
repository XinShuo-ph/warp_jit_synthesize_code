from __future__ import annotations

import hashlib
import importlib
import inspect
import os
import sys

import warp as wp


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main() -> None:
    wp.init()

    # Ensure the package root (`jit/code/`) is importable when running this as a script.
    code_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    if code_root not in sys.path:
        sys.path.insert(0, code_root)

    from extraction.ir_extractor import extract_ir  # noqa: PLC0415

    case_mods = [
        "extraction.cases.case_arith",
        "extraction.cases.case_branch",
        "extraction.cases.case_loop",
        "extraction.cases.case_atomic",
        "extraction.cases.case_vec",
    ]

    pairs: list[tuple[str, str]] = []

    for mod_name in case_mods:
        m = importlib.import_module(mod_name)
        k = m.get_kernel()

        py_src = inspect.getsource(k.func if hasattr(k, "func") else k)

        ir1 = extract_ir(k, device="cpu").cpp_ir
        ir2 = extract_ir(k, device="cpu").cpp_ir
        if ir1 != ir2:
            raise SystemExit(f"{mod_name}: non-deterministic IR (cpu)")

        pairs.append((py_src, ir1))
        print(f"{mod_name}: ir_sha256={_sha256(ir1)[:12]} py_lines={len(py_src.splitlines())}")

    if len(pairs) < 5:
        raise SystemExit("expected >= 5 pairs")

    print(f"ok: pairs={len(pairs)}")


if __name__ == "__main__":
    main()


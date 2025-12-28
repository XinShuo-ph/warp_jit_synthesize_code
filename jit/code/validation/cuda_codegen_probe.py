"""
CUDA codegen probe (GPU not required).

This script attempts to ask Warp to generate CUDA-targeted source via ModuleBuilder.codegen("cuda").
It should succeed on machines with CUDA toolchain components available (e.g. NVRTC), even if no GPU exists.
"""

import sys

import warp as wp

# Local import path: use workspace-relative structure
sys.path.insert(0, "/workspace/jit/code/extraction")

from ir_extractor import extract_ir  # noqa: E402


@wp.kernel
def _probe_kernel(a: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = a[tid] * 2.0 + 1.0


def main() -> int:
    wp.init()

    try:
        result = extract_ir(_probe_kernel, device="cuda", include_backward=False)
    except Exception as e:
        print("CUDA codegen probe FAILED.")
        print("Reason:", str(e))
        print(
            "\nIf you expected this to work, ensure CUDA toolchain components are installed "
            "(Warp typically needs NVRTC / CUDA toolkit libraries available)."
        )
        return 1

    code = result.get("cpp_code") or ""
    forward = result.get("forward_code") or ""

    print("CUDA codegen probe PASSED.")
    print(f"Generated module source length: {len(code)}")
    print(f"Generated forward function length: {len(forward)}")
    print("\n--- Forward function (first 40 lines) ---")
    print("\n".join(forward.splitlines()[:40]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


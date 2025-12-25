from __future__ import annotations

import warp as wp

from jit.code.extraction.ir_extractor import extract_ir, extract_ir_artifact


# ----------------------------
# 5+ small kernels (CPU-safe)
# ----------------------------


@wp.kernel
def k_add(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = a[i] + b[i]


@wp.kernel
def k_saxpy(a: wp.array(dtype=wp.float32), x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = a[0] * x[i] + y[i]


@wp.kernel
def k_trig(x: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = wp.sin(x[i]) + wp.cos(x[i]) + wp.sqrt(wp.abs(x[i]) + 1.0)


@wp.kernel
def k_branch(x: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    v = x[i]
    if v > 0.0:
        out[i] = v * 2.0
    else:
        out[i] = -v * 3.0


@wp.kernel
def k_vec_ops(x: wp.array(dtype=wp.vec3f), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    v = x[i]
    out[i] = wp.dot(v, v) + wp.length(v)


@wp.kernel
def k_atomic(accum: wp.array(dtype=wp.int32)):
    i = wp.tid()
    wp.atomic_add(accum, 0, 1)


def _compile_kernel(kernel, device: str = "cpu") -> None:
    # Force compilation by launching with tiny inputs.
    with wp.ScopedDevice(device):
        if kernel is k_atomic:
            accum = wp.zeros(1, dtype=wp.int32)
            wp.launch(kernel, dim=128, inputs=[accum])
            wp.synchronize()
            return

        n = 256
        if kernel is k_vec_ops:
            x = wp.zeros(n, dtype=wp.vec3f)
            out = wp.zeros(n, dtype=wp.float32)
            wp.launch(kernel, dim=n, inputs=[x, out])
            wp.synchronize()
            return

        x = wp.zeros(n, dtype=wp.float32)
        y = wp.zeros(n, dtype=wp.float32)
        out = wp.zeros(n, dtype=wp.float32)
        a = wp.array([2.0], dtype=wp.float32)

        if kernel is k_add:
            wp.launch(kernel, dim=n, inputs=[x, y, out])
        elif kernel is k_saxpy:
            wp.launch(kernel, dim=n, inputs=[a, x, y, out])
        elif kernel is k_trig:
            wp.launch(kernel, dim=n, inputs=[x, out])
        elif kernel is k_branch:
            wp.launch(kernel, dim=n, inputs=[x, out])
        else:
            raise RuntimeError(f"Unhandled kernel in test harness: {kernel}")

        wp.synchronize()


def _assert_ir_non_empty(kernel) -> None:
    ir = extract_ir(kernel, device="cpu", prefer=("cpp",))
    assert isinstance(ir, str)
    assert len(ir.strip()) > 0

    artifact = extract_ir_artifact(kernel, device="cpu", prefer=("cpp", "meta"))
    assert artifact.path.endswith(".cpp")


def main() -> int:
    wp.init()

    kernels = [k_add, k_saxpy, k_trig, k_branch, k_vec_ops, k_atomic]
    for k in kernels:
        _compile_kernel(k, device="cpu")
        _assert_ir_non_empty(k)

    print("ok: extracted non-empty IR for", len(kernels), "kernels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


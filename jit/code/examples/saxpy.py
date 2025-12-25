import numpy as np
import warp as wp


@wp.kernel
def saxpy_kernel(a: float, x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = a * x[i] + y[i]


def main():
    wp.init()

    n = 2048
    rng = np.random.default_rng(1234)
    x_np = rng.standard_normal(n, dtype=np.float32)
    y_np = rng.standard_normal(n, dtype=np.float32)
    a = 2.5

    x = wp.array(x_np, dtype=wp.float32, device="cpu")
    y = wp.array(y_np, dtype=wp.float32, device="cpu")
    out = wp.empty(n, dtype=wp.float32, device="cpu")

    wp.launch(saxpy_kernel, dim=n, inputs=[a, x, y, out], device="cpu")

    out_np = out.numpy()
    expected = a * x_np + y_np
    if not np.allclose(out_np, expected, rtol=1e-6, atol=1e-6):
        raise SystemExit("saxpy.py: mismatch")

    print(f"saxpy.py ok: checksum={float(out_np.sum()):.6f}")


if __name__ == "__main__":
    main()


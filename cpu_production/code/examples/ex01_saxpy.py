import numpy as np

import warp as wp


@wp.kernel
def saxpy(a: float, x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = a * x[i] + y[i]


def main() -> None:
    wp.init()

    n = 8192
    device = "cpu"
    a = 2.5

    x_np = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    y_np = np.linspace(3.0, 4.0, n, dtype=np.float32)

    x = wp.array(x_np, dtype=wp.float32, device=device)
    y = wp.array(y_np, dtype=wp.float32, device=device)
    out = wp.zeros(n, dtype=wp.float32, device=device)

    wp.launch(saxpy, dim=n, inputs=[a, x, y, out], device=device)
    wp.synchronize_device(device)

    out_np = out.numpy()
    expected = a * x_np + y_np
    max_err = float(np.max(np.abs(out_np - expected)))
    checksum = float(np.sum(out_np))
    print(f"ex01_saxpy: n={n} device={device} max_err={max_err:.3e} checksum={checksum:.6f}")

    if not np.allclose(out_np, expected, rtol=0.0, atol=1e-6):
        raise SystemExit("ex01_saxpy failed: output mismatch")


if __name__ == "__main__":
    main()


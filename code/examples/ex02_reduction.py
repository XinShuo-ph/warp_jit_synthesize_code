import numpy as np

import warp as wp


@wp.kernel
def square(x: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = x[i] * x[i]


def main() -> None:
    wp.init()

    n = 16384
    device = "cpu"

    x_np = np.linspace(-2.0, 2.0, n, dtype=np.float32)
    x = wp.array(x_np, dtype=wp.float32, device=device)
    squares = wp.zeros(n, dtype=wp.float32, device=device)

    wp.launch(square, dim=n, inputs=[x, squares], device=device)
    wp.synchronize_device(device)

    squares_np = squares.numpy()
    sum_sq = float(np.sum(squares_np))
    expected = float(np.sum(x_np * x_np))
    abs_err = float(abs(sum_sq - expected))
    print(f"ex02_reduction: n={n} device={device} abs_err={abs_err:.6e} sum_sq={sum_sq:.6f}")

    if not np.isfinite(sum_sq) or abs_err > 1e-3:
        raise SystemExit("ex02_reduction failed: unexpected reduction error")


if __name__ == "__main__":
    main()


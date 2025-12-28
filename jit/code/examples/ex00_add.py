import numpy as np

import warp as wp


@wp.kernel
def add(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = a[i] + b[i]


def main() -> None:
    wp.init()

    n = 4096
    device = "cpu"

    a_np = np.arange(n, dtype=np.float32)
    b_np = (np.arange(n, dtype=np.float32) * 0.5).astype(np.float32)

    a = wp.array(a_np, dtype=wp.float32, device=device)
    b = wp.array(b_np, dtype=wp.float32, device=device)
    out = wp.zeros(n, dtype=wp.float32, device=device)

    wp.launch(add, dim=n, inputs=[a, b, out], device=device)
    wp.synchronize_device(device)

    out_np = out.numpy()
    expected = a_np + b_np
    max_err = float(np.max(np.abs(out_np - expected)))
    checksum = float(np.sum(out_np))
    print(f"ex00_add: n={n} device={device} max_err={max_err:.3e} checksum={checksum:.6f}")

    if not np.allclose(out_np, expected, rtol=0.0, atol=1e-6):
        raise SystemExit("ex00_add failed: output mismatch")


if __name__ == "__main__":
    main()


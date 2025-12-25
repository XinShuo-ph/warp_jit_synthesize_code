import numpy as np
import warp as wp


@wp.kernel
def add_kernel(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = a[i] + b[i]


def main():
    wp.init()

    n = 1024
    a_np = np.arange(n, dtype=np.float32)
    b_np = (2.0 * np.arange(n, dtype=np.float32))

    a = wp.array(a_np, dtype=wp.float32, device="cpu")
    b = wp.array(b_np, dtype=wp.float32, device="cpu")
    out = wp.empty(n, dtype=wp.float32, device="cpu")

    wp.launch(add_kernel, dim=n, inputs=[a, b, out], device="cpu")

    out_np = out.numpy()
    expected = a_np + b_np
    if not np.allclose(out_np, expected):
        raise SystemExit("add.py: mismatch")

    print(f"add.py ok: out[-1]={float(out_np[-1]):.1f}")


if __name__ == "__main__":
    main()


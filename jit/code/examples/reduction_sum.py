import numpy as np
import warp as wp


@wp.kernel
def sum_kernel(x: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    wp.atomic_add(out, 0, x[i])


def main():
    wp.init()

    n = 4096
    rng = np.random.default_rng(4321)
    x_np = rng.standard_normal(n, dtype=np.float32)

    x = wp.array(x_np, dtype=wp.float32, device="cpu")
    out = wp.zeros(1, dtype=wp.float32, device="cpu")

    wp.launch(sum_kernel, dim=n, inputs=[x, out], device="cpu")

    out_np = out.numpy()
    expected = np.sum(x_np, dtype=np.float32)
    if not np.allclose(out_np[0], expected, rtol=1e-5, atol=1e-5):
        raise SystemExit(f"reduction_sum.py: mismatch got={out_np[0]} expected={expected}")

    print(f"reduction_sum.py ok: sum={float(out_np[0]):.6f}")


if __name__ == "__main__":
    main()


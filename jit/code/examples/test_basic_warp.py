"""Basic warp kernels to verify compilation and understand IR extraction."""
import warp as wp

wp.init()


@wp.kernel
def add_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]


@wp.kernel
def saxpy_kernel(
    x: wp.array(dtype=float), y: wp.array(dtype=float), alpha: float
):
    tid = wp.tid()
    y[tid] = alpha * x[tid] + y[tid]


@wp.kernel
def dot_product_kernel(
    a: wp.array(dtype=float), b: wp.array(dtype=float), result: wp.array(dtype=float)
):
    tid = wp.tid()
    wp.atomic_add(result, 0, a[tid] * b[tid])


def test_add():
    n = 1024
    a = wp.array([float(i) for i in range(n)], dtype=float)
    b = wp.array([float(i * 2) for i in range(n)], dtype=float)
    c = wp.zeros(n, dtype=float)

    wp.launch(kernel=add_kernel, dim=n, inputs=[a, b, c])
    wp.synchronize()

    c_np = c.numpy()
    assert c_np[0] == 0.0
    assert c_np[1] == 3.0
    assert c_np[10] == 30.0
    print("test_add: PASSED")


def test_saxpy():
    n = 1024
    x = wp.array([float(i) for i in range(n)], dtype=float)
    y = wp.array([float(1) for _ in range(n)], dtype=float)
    alpha = 2.0

    wp.launch(kernel=saxpy_kernel, dim=n, inputs=[x, y, alpha])
    wp.synchronize()

    y_np = y.numpy()
    assert y_np[0] == 1.0  # 2*0 + 1
    assert y_np[1] == 3.0  # 2*1 + 1
    assert y_np[10] == 21.0  # 2*10 + 1
    print("test_saxpy: PASSED")


def test_dot():
    n = 100
    a = wp.array([float(1) for _ in range(n)], dtype=float)
    b = wp.array([float(2) for _ in range(n)], dtype=float)
    result = wp.zeros(1, dtype=float)

    wp.launch(kernel=dot_product_kernel, dim=n, inputs=[a, b, result])
    wp.synchronize()

    r = result.numpy()[0]
    assert r == 200.0  # 100 * 1 * 2
    print("test_dot: PASSED")


if __name__ == "__main__":
    test_add()
    test_saxpy()
    test_dot()
    print("\nAll basic tests passed!")

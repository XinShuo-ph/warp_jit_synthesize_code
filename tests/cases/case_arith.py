import warp as wp


@wp.kernel
def k_arith(x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = (x[i] * 2.0 + y[i]) / 3.0


def get_kernel():
    return k_arith


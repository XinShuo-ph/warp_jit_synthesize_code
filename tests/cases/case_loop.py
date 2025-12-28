import warp as wp


@wp.kernel
def k_loop(x: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    acc = 0.0
    for j in range(4):
        acc = acc + x[i] * float(j + 1)
    out[i] = acc


def get_kernel():
    return k_loop


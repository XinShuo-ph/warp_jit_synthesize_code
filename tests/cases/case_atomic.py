import warp as wp


@wp.kernel
def k_atomic_sum(x: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    wp.atomic_add(out, 0, x[i])


def get_kernel():
    return k_atomic_sum


import warp as wp


@wp.kernel
def k_branch(x: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    v = x[i]
    if v > 0.0:
        out[i] = v
    else:
        out[i] = -v


def get_kernel():
    return k_branch


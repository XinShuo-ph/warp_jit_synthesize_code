import warp as wp


@wp.kernel
def k_vec(x: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    v = x[i]
    out[i] = wp.dot(v, v)


def get_kernel():
    return k_vec


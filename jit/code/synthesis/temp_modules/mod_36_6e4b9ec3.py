import warp as wp


@wp.kernel
def kernel_36(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    v = a[tid]
    out[tid] = v - b[tid]

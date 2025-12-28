@wp.kernel
def vec_syjfja(a: wp.array(dtype=wp.vec2), b: wp.array(dtype=wp.vec2), out: wp.array(dtype=wp.vec2)):
    tid = wp.tid()
    out[tid] = wp.normalize(a[tid])

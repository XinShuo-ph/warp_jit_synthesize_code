@wp.kernel
def vec_okuuez(a: wp.array(dtype=wp.vec4), b: wp.array(dtype=wp.vec4), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = wp.length(a[tid])

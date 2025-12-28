@wp.kernel
def mat_akfmrx(m: wp.array(dtype=wp.mat22), v: wp.array(dtype=wp.vec2), out: wp.array(dtype=wp.vec2)):
    tid = wp.tid()
    out[tid] = m[tid] * v[tid]

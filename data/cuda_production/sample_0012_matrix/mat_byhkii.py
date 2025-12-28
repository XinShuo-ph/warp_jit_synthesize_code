@wp.kernel
def mat_byhkii(m: wp.array(dtype=wp.mat33), out: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    out[tid] = wp.transpose(m[tid])

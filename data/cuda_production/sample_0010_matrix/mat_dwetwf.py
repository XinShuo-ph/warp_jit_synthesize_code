@wp.kernel
def mat_dwetwf(a: wp.array(dtype=wp.mat22), b: wp.array(dtype=wp.mat22), out: wp.array(dtype=wp.mat22)):
    tid = wp.tid()
    out[tid] = a[tid] * b[tid]

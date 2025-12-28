@wp.kernel
def mat_ufrugc(a: wp.array(dtype=wp.mat33), b: wp.array(dtype=wp.mat33), out: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    out[tid] = a[tid] * b[tid]

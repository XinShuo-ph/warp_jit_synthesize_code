import warp as wp


@wp.kernel
def kernel_99(in1: wp.array(dtype=int), in2: wp.array(dtype=int), out: wp.array(dtype=int)):
    tid = wp.tid()
    x = in1[tid]
    y = in2[tid]
    out[tid] = x - y

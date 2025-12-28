import warp as wp


@wp.kernel
def kernel_32(in1: wp.array(dtype=float), in2: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    x = in1[tid]
    y = in2[tid]
    out[tid] = x / y

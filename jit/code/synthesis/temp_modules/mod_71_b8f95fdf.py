import warp as wp


@wp.kernel
def kernel_71(data: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    val = data[tid]
    if val < 7.53:
        out[tid] = val * 2.0
    else:
        out[tid] = val

import warp as wp


@wp.kernel
def kernel_7(data: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    val = data[tid]
    if val <= 8.56:
        out[tid] = val * 2.0
    else:
        out[tid] = val

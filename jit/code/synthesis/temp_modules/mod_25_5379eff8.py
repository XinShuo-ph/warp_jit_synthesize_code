import warp as wp


@wp.kernel
def kernel_25(data: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    val = data[tid]
    if val > 2.17:
        out[tid] = val * 2.0
    else:
        out[tid] = val

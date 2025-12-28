import warp as wp


@wp.kernel
def kernel_15(data: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    val = data[tid]
    if val > 1.97:
        out[tid] = val * 2.0
    else:
        out[tid] = val

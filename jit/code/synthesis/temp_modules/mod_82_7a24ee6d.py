import warp as wp


@wp.kernel
def kernel_82(data: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    val = data[tid]
    if val >= 2.68:
        out[tid] = val * 2.0
    else:
        out[tid] = val

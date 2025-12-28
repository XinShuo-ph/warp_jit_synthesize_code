import warp as wp


@wp.kernel
def kernel_53(data: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    val = data[tid]
    if val > 4.07:
        out[tid] = val * 2.0
    else:
        out[tid] = val

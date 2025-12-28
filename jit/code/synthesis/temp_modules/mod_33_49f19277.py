import warp as wp


@wp.kernel
def kernel_33(inp: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    x = inp[tid]
    # Ensure safe args for sqrt/log if needed (using abs)
    v1 = wp.abs(x)
    v2 = wp.cos(x)
    out[tid] = v1 + v2

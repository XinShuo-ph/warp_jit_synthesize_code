import warp as wp


@wp.kernel
def kernel_38(inp: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    x = inp[tid]
    # Ensure safe args for sqrt/log if needed (using abs)
    v1 = wp.sqrt(x)
    v2 = wp.sqrt(x)
    out[tid] = v1 + v2

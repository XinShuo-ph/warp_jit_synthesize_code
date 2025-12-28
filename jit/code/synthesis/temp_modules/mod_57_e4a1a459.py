import warp as wp


@wp.kernel
def kernel_57(n: int, out: wp.array(dtype=float)):
    tid = wp.tid()
    sum = float(0.0)
    for i in range(n):
        sum = sum + float(i) * 0.1
    out[tid] = sum

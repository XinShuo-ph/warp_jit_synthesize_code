import warp as wp


@wp.kernel
def kernel_19(width: int, height: int, out: wp.array(dtype=float)):
    tid = wp.tid()
    sum = float(0.0)
    for i in range(width):
        for j in range(height):
            sum = sum + float(i) * float(j)
    out[tid] = sum

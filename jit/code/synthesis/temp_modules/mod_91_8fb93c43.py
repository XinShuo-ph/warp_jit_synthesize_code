import warp as wp


@wp.kernel
def kernel_91(data: wp.array(dtype=int), counter: wp.array(dtype=int)):
    tid = wp.tid()
    val = data[tid]
    if val > 0:
        wp.atomic_add(counter, 0, val)

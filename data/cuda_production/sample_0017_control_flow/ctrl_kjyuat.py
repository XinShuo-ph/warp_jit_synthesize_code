@wp.kernel
def ctrl_kjyuat(a: wp.array(dtype=float), threshold: float, out: wp.array(dtype=float)):
    tid = wp.tid()
    if a[tid] > threshold:
        out[tid] = 1.0
    else:
        out[tid] = 0.0

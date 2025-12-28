@wp.kernel
def scalar_tlxqzy(x: wp.array(dtype=float), out: wp.array(dtype=float), scale: float, offset: float):
    tid = wp.tid()
    out[tid] = (x[tid] - scale) + offset

@wp.kernel
def combined_ldrywc(a: wp.array(dtype=float), b: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    acc = float(0.0)
    for i in range(3):
        if a[tid] * float(i) > 1.28:
            acc = acc + wp.cos(b[tid])
        else:
            acc = acc + b[tid]
    out[tid] = acc

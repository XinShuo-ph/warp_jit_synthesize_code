@wp.kernel
def combined_afksoi(a: wp.array(dtype=float), b: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    acc = float(0.0)
    for i in range(5):
        if a[tid] * float(i) > 0.84:
            acc = acc + wp.sin(b[tid])
        else:
            acc = acc + b[tid]
    out[tid] = acc

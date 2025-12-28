@wp.kernel
def nested_hqlsqc(data: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    total = float(0.0)
    for i in range(4):
        for j in range(2):
            total = total + data[tid] * float(i * j + 1)
    out[tid] = total

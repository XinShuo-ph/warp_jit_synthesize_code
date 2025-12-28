@wp.kernel
def nested_bxkbgg(data: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    total = float(0.0)
    for i in range(3):
        for j in range(3):
            total = total + data[tid] * float(i * j + 1)
    out[tid] = total

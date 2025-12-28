@wp.kernel
def atom_jrimtx(values: wp.array(dtype=float), result: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_add(result, 0, values[tid])

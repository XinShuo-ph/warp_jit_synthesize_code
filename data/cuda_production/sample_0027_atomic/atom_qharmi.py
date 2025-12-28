@wp.kernel
def atom_qharmi(values: wp.array(dtype=float), result: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_max(result, 0, values[tid])

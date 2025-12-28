@wp.kernel
def atom_sxoaqm(values: wp.array(dtype=float), result: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_min(result, 0, values[tid])

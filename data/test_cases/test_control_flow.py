@wp.kernel
def test_control_flow(data: wp.array(dtype=float),
                      threshold: float,
                      output: wp.array(dtype=float)):
    """Conditional branching."""
    i = wp.tid()

    val = data[i]

    if val < 0.0:
        output[i] = -val
    elif val < threshold:
        output[i] = val * 2.0
    else:
        output[i] = threshold + (val - threshold) * 0.5

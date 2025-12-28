import warp as wp
@wp.kernel
def test_functions(inputs: wp.array(dtype=float),
                   weights: wp.array(dtype=float),
                   bias: float,
                   outputs: wp.array(dtype=float)):
    """Kernel using helper function."""
    i = wp.tid()

    # Weighted sum
    z = inputs[i] * weights[i] + bias

    # Apply activation
    outputs[i] = sigmoid(z)

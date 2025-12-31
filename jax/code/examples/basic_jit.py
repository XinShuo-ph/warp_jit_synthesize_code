"""Basic JAX JIT examples demonstrating compilation and IR extraction."""
import jax
import jax.numpy as jnp


@jax.jit
def vector_add(a, b):
    """Simple vector addition."""
    return a + b


@jax.jit
def matmul(a, b):
    """Matrix multiplication."""
    return jnp.dot(a, b)


@jax.jit
def simple_nn_layer(x, w, b):
    """Single neural network layer with ReLU activation."""
    return jax.nn.relu(jnp.dot(x, w) + b)


@jax.jit
def softmax_cross_entropy(logits, labels):
    """Softmax cross entropy loss."""
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.sum(labels * log_probs, axis=-1).mean()


@jax.jit
def convolution_1d(x, kernel):
    """1D convolution using lax.conv_general_dilated."""
    # x: (batch, length, channels)
    # kernel: (kernel_size, in_channels, out_channels)
    return jax.lax.conv_general_dilated(
        x, kernel,
        window_strides=(1,),
        padding='SAME',
        dimension_numbers=('NWC', 'WIO', 'NWC')
    )


def run_examples():
    """Run all examples and verify they work."""
    # Vector add
    a = jnp.array([1.0, 2.0, 3.0])
    b = jnp.array([4.0, 5.0, 6.0])
    print("vector_add:", vector_add(a, b))

    # Matrix multiplication
    m1 = jnp.ones((3, 4))
    m2 = jnp.ones((4, 5))
    print("matmul shape:", matmul(m1, m2).shape)

    # NN layer
    x = jnp.ones((2, 8))
    w = jnp.ones((8, 16))
    b = jnp.zeros((16,))
    print("nn_layer shape:", simple_nn_layer(x, w, b).shape)

    # Softmax cross entropy
    logits = jnp.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    labels = jnp.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    print("cross_entropy:", softmax_cross_entropy(logits, labels))

    # 1D convolution
    x_conv = jnp.ones((1, 10, 3))  # batch=1, length=10, channels=3
    kernel = jnp.ones((3, 3, 8))   # kernel_size=3, in=3, out=8
    print("conv1d shape:", convolution_1d(x_conv, kernel).shape)

    print("\nAll examples ran successfully!")


if __name__ == "__main__":
    run_examples()

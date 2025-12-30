"""Simple neural network example with JAX."""
import jax
import jax.numpy as jnp


def init_mlp_params(layer_sizes, key):
    """Initialize MLP parameters with random weights."""
    params = []
    for i in range(len(layer_sizes) - 1):
        key, subkey = jax.random.split(key)
        in_size, out_size = layer_sizes[i], layer_sizes[i + 1]
        # Xavier initialization
        scale = jnp.sqrt(2.0 / (in_size + out_size))
        W = jax.random.normal(subkey, (in_size, out_size)) * scale
        b = jnp.zeros(out_size)
        params.append((W, b))
    return params


def relu(x):
    """ReLU activation."""
    return jnp.maximum(0, x)


def mlp_forward(params, x):
    """Forward pass through MLP."""
    for W, b in params[:-1]:
        x = relu(x @ W + b)
    # Last layer (no activation for regression, or softmax for classification)
    W, b = params[-1]
    return x @ W + b


def mse_loss(params, x, y):
    """Mean squared error loss."""
    pred = mlp_forward(params, x)
    return jnp.mean((pred - y) ** 2)


def cross_entropy_loss(params, x, y_onehot):
    """Cross-entropy loss for classification."""
    logits = mlp_forward(params, x)
    log_probs = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    return -jnp.mean(jnp.sum(y_onehot * log_probs, axis=-1))


def accuracy(params, x, y_labels):
    """Compute classification accuracy."""
    logits = mlp_forward(params, x)
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == y_labels)


@jax.jit
def train_step(params, x, y, learning_rate=0.01):
    """Single training step with gradient descent."""
    loss, grads = jax.value_and_grad(mse_loss)(params, x, y)
    # Update parameters
    new_params = [
        (W - learning_rate * dW, b - learning_rate * db)
        for (W, b), (dW, db) in zip(params, grads)
    ]
    return new_params, loss


def train_loop(params, x_train, y_train, epochs=100, learning_rate=0.01):
    """Training loop."""
    losses = []
    for epoch in range(epochs):
        params, loss = train_step(params, x_train, y_train, learning_rate)
        losses.append(float(loss))
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: loss = {loss:.6f}")
    return params, losses


def main():
    key = jax.random.PRNGKey(42)
    
    print("=" * 60)
    print("Neural Network Training Example")
    print("=" * 60)
    
    # Generate synthetic data: y = 2x1 - x2 + 0.5x3 + noise
    n_samples = 100
    key, data_key = jax.random.split(key)
    X = jax.random.normal(data_key, (n_samples, 3))
    y_true = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2]
    key, noise_key = jax.random.split(key)
    y = y_true + 0.1 * jax.random.normal(noise_key, (n_samples,))
    y = y.reshape(-1, 1)  # Shape: (n_samples, 1)
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Initialize MLP: 3 -> 16 -> 8 -> 1
    layer_sizes = [3, 16, 8, 1]
    key, init_key = jax.random.split(key)
    params = init_mlp_params(layer_sizes, init_key)
    
    print(f"MLP architecture: {layer_sizes}")
    print(f"Parameters: {sum(W.size + b.size for W, b in params)} total")
    
    # Initial loss
    initial_loss = mse_loss(params, X, y)
    print(f"Initial loss: {initial_loss:.6f}")
    
    # Train
    print("\nTraining...")
    params, losses = train_loop(params, X, y, epochs=100, learning_rate=0.05)
    
    # Final evaluation
    final_loss = mse_loss(params, X, y)
    predictions = mlp_forward(params, X)
    r2_score = 1 - jnp.sum((y - predictions)**2) / jnp.sum((y - jnp.mean(y))**2)
    
    print(f"\nFinal loss: {final_loss:.6f}")
    print(f"R² score: {r2_score:.4f}")
    print(f"Loss improved: {initial_loss:.4f} -> {final_loss:.4f}")
    
    return params, losses, final_loss


if __name__ == "__main__":
    params1, losses1, loss1 = main()
    print("\n" + "=" * 60)
    print("Run 2:")
    print("=" * 60)
    params2, losses2, loss2 = main()
    
    # Verify consistency
    assert jnp.isclose(loss1, loss2, rtol=1e-4), f"Loss mismatch: {loss1} vs {loss2}"
    print("\n✓ Consistency check passed!")

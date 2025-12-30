import jax
import jax.numpy as jnp

def simple_kernel(x, y):
    """A simple element-wise operation."""
    return x * y + jnp.sin(x)

def extract_ir(func, *args):
    """Extracts HLO IR from a JAX function."""
    print(f"--- Extracting IR for {func.__name__} ---")
    
    # 1. Lower the function to HLO
    lowered = jax.jit(func).lower(*args)
    
    # 2. Get HLO text
    hlo_text = lowered.as_text()
    print("HLO Text:")
    print(hlo_text)
    
    # 3. Get StableHLO (if available in newer JAX versions)
    # Note: .as_text() usually returns HLO. 
    # To get stablehlo, we might need other APIs or it's the default in newer versions.
    # Let's check what compile().as_text() gives.
    
    compiled = lowered.compile()
    print("\nCompiled executable text (ASM/PTX usually, but depends on backend):")
    # compiled.as_text() might give assembly on CPU/GPU
    try:
        print(compiled.as_text()[:500] + "...") # Truncate
    except:
        print("Could not get compiled text.")

    return hlo_text

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (10,))
    y = jax.random.normal(key, (10,))
    
    ir = extract_ir(simple_kernel, x, y)
    
    # Save to file
    with open("jit/data/simple_kernel.hlo", "w") as f:
        f.write(ir)

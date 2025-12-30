import random
import jax.numpy as jnp

UNARY_OPS = ['jnp.sin', 'jnp.cos', 'jnp.exp', 'jnp.tanh', 'jnp.abs', 'jnp.square']
BINARY_OPS = ['jnp.add', 'jnp.subtract', 'jnp.multiply', 'jnp.maximum', 'jnp.minimum']
# Reductions change shape, requiring more complex shape inference. 
# For now, we stick to element-wise or shape-preserving ops to keep it simple, 
# or specific reductions if we handle them.

class JAXGenerator:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        
    def generate_fn(self, num_ops=5):
        """Generates a random JAX function source code."""
        
        args = ['x', 'y']
        lines = []
        vars_available = list(args)
        
        # Header
        code = "import jax\nimport jax.numpy as jnp\n\n"
        code += "def generated_fn(x, y):\n"
        
        for i in range(num_ops):
            op_type = self.rng.choice(['unary', 'binary'])
            new_var = f"v{i}"
            
            if op_type == 'unary':
                op = self.rng.choice(UNARY_OPS)
                arg = self.rng.choice(vars_available)
                lines.append(f"    {new_var} = {op}({arg})")
            else:
                op = self.rng.choice(BINARY_OPS)
                arg1 = self.rng.choice(vars_available)
                arg2 = self.rng.choice(vars_available)
                lines.append(f"    {new_var} = {op}({arg1}, {arg2})")
            
            vars_available.append(new_var)
            
        # Return the last variable
        lines.append(f"    return {vars_available[-1]}")
        
        return code + "\n".join(lines)

if __name__ == "__main__":
    gen = JAXGenerator(42)
    print(gen.generate_fn())

import random

OPS = ['jnp.add', 'jnp.subtract', 'jnp.multiply', 'jnp.maximum', 'jnp.minimum']
UNARY_OPS = ['jnp.sin', 'jnp.cos', 'jnp.exp', 'jnp.abs', 'jnp.tanh']

def generate_random_function(seed=None):
    if seed:
        random.seed(seed)
        
    # Generate a function with random number of arguments and operations
    num_args = random.randint(1, 3)
    args = [f'x{i}' for i in range(num_args)]
    
    lines = []
    lines.append("import jax")
    lines.append("import jax.numpy as jnp")
    lines.append("")
    lines.append(f"def generated_fn({', '.join(args)}):")
    
    # Simple SSA-like generation
    # We maintain a pool of available variables
    vars = list(args)
    num_stmts = random.randint(3, 10)
    
    for i in range(num_stmts):
        # Decide between binary or unary
        # Bias towards using recent variables to create depth
        
        if len(vars) >= 2 and random.random() > 0.4:
            op = random.choice(OPS)
            v1 = random.choice(vars)
            v2 = random.choice(vars)
            new_var = f"v{i}"
            lines.append(f"    {new_var} = {op}({v1}, {v2})")
            vars.append(new_var)
        else:
            op = random.choice(UNARY_OPS)
            v1 = random.choice(vars)
            new_var = f"v{i}"
            lines.append(f"    {new_var} = {op}({v1})")
            vars.append(new_var)
            
    # Return the last variable
    lines.append(f"    return {vars[-1]}")
    
    return "\n".join(lines)

if __name__ == "__main__":
    print(generate_random_function(42))

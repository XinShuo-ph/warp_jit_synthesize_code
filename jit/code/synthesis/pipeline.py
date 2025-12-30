import jax
import jax.numpy as jnp
import os
import sys
import uuid

# Add path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from jit.code.synthesis.generator import JAXGenerator
from jit.code.extraction.ir_extractor import get_ir

OUTPUT_DIR = "jit/data/samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_sample(idx, generator):
    # 1. Generate Code
    code = generator.generate_fn(num_ops=random.randint(3, 10))
    
    # 2. Compile/Exec
    # We need a clean namespace
    local_scope = {}
    try:
        # Pass local_scope as globals so imports are available to the function
        exec(code, local_scope)
        func = local_scope['generated_fn']
    except Exception as e:
        print(f"Error parsing generated code: {e}")
        return False

    # 3. Create Inputs (same shape for element-wise ops)
    # Random shape
    shape = (random.randint(1, 10), random.randint(1, 10))
    key = jax.random.PRNGKey(idx)
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, shape)
    y = jax.random.normal(k2, shape)
    
    # 4. Extract IR
    try:
        ir = get_ir(func, x, y)
    except Exception as e:
        print(f"Error extracting IR: {e}")
        return False
        
    # 5. Save
    sample_id = uuid.uuid4().hex[:8]
    
    # Save Python
    with open(f"{OUTPUT_DIR}/{sample_id}.py", "w") as f:
        f.write(code)
        
    # Save IR
    with open(f"{OUTPUT_DIR}/{sample_id}.hlo", "w") as f:
        f.write(ir)
        
    return True

import random

if __name__ == "__main__":
    gen = JAXGenerator(seed=123)
    success_count = 0
    target = 100
    
    print(f"Generating {target} samples...")
    
    # Try more times than target to account for failures
    for i in range(target * 2):
        if success_count >= target:
            break
        if generate_sample(i, gen):
            success_count += 1
            print(f"Generated sample {success_count}/{target}")
            
    print(f"Done. Generated {success_count} samples in {OUTPUT_DIR}")

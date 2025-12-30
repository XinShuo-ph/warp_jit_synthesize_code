import jax
import jax.numpy as jnp
import numpy as np
import inspect
import sys
import os

# Fix path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..', 'extraction'))

from generator import generate_random_function
from ir_extractor import extract_ir

def synthesize_pair(seed=None):
    code = generate_random_function(seed)
    
    # Execute code to get function
    local_scope = {}
    try:
        exec(code, local_scope)
    except Exception as e:
        print(f"Exec error: {e}")
        return None
        
    fn = local_scope.get('generated_fn')
    if not fn:
        return None
    
    # Inspect args
    try:
        sig = inspect.signature(fn)
        num_args = len(sig.parameters)
    except ValueError:
        return None
    
    # Generate random inputs
    # Use standard shape (10, 10) for now, floats
    shape = (10, 10)
    # Use jnp.array
    args = [jnp.array(np.random.randn(*shape).astype(np.float32)) for _ in range(num_args)]
    
    # Extract IR
    res = extract_ir(fn, *args)
    
    if not res['success']:
        print(f"Extraction error: {res['error']}")
        return None
        
    return {
        "code": code,
        "jaxpr": res['jaxpr'],
        "hlo": res['hlo']
    }

if __name__ == "__main__":
    res = synthesize_pair(42)
    if res:
        print("Successfully synthesized pair!")
        print(f"Code length: {len(res['code'])}")
        print(f"HLO length: {len(res['hlo'])}")
    else:
        print("Failed to synthesize pair.")

import jax
import jax.numpy as jnp
from poisson_solver import solve_poisson
import sys
import os

# Add workspace root to sys.path
sys.path.append("/workspace")
from jit_jax.code.extraction.ir_extractor import extract_ir

def main():
    N = 16
    u_init = jnp.zeros((N, N))
    f = jnp.ones((N, N))
    
    print("Extracting IR for full solve_poisson...")
    # We trace solve_poisson with n_iter=10
    # Note: n_iter is an integer, so if we jit it, it should be static.
    # But extract_ir traces with arguments.
    # solve_poisson expects (f, u_init, n_iter)
    
    # We need to wrap it to make n_iter static or just pass it if we want to trace it unrolled (bad).
    # Ideally, n_iter should be static for loop generation, or dynamic for while_loop.
    # Our implementation uses scan with length=n_iter. length in scan must be static integer.
    
    # So we define a wrapper that fixes n_iter
    def solve_fixed(f, u):
        return solve_poisson(f, u, n_iter=10)
        
    res = extract_ir(solve_fixed, f, u_init)
    
    if res['success']:
        print("Success!")
        print(f"Jaxpr length: {len(res['jaxpr'])}")
        print(f"HLO length: {len(res['hlo'])}")
        
        # Check if 'scan' or 'while' is in the HLO/Jaxpr to ensure it's not fully unrolled
        if "scan" in res['jaxpr']:
            print("Found 'scan' in Jaxpr -> Loop preserved.")
        else:
            print("Warning: 'scan' not found in Jaxpr.")
            
    else:
        print(f"Failed: {res['error']}")

if __name__ == "__main__":
    main()

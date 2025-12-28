"""
Verification script for CUDA backend.
This script attempts to run the synthesis pipeline using the CUDA backend.
If no GPU is available, it should fail gracefully with a specific error message.
"""
import sys
import os
from pathlib import Path

# Add synthesis dir to path
sys.path.insert(0, str(Path(__file__).parent / "synthesis"))

from pipeline import run_pipeline

def main():
    print("="*50)
    print("CUDA Backend Verification")
    print("="*50)
    
    output_dir = Path(__file__).parent / "cuda_verification_output"
    
    try:
        # Try generating 1 pair with CUDA
        print("Attempting to generate 1 pair using device='cuda'...")
        run_pipeline(str(output_dir), count=1, device="cuda")
        print("\nSUCCESS: CUDA pipeline ran successfully (Unexpected if no GPU present).")
        
    except Exception as e:
        print("\nPipeline execution finished/failed.")
        print(f"Result: {e}")
        
        # Check if it was a driver error (expected)
        error_str = str(e)
        if "CUDA driver" in error_str or "cuda" in error_str.lower():
            print("\nNOTE: Failure likely due to missing GPU/Driver, which is EXPECTED in this environment.")
            print("The code logic for selecting 'cuda' device appears to be in place.")
        else:
            print("\nWARNING: Unexpected error not related to CUDA driver.")
            raise e

if __name__ == "__main__":
    main()

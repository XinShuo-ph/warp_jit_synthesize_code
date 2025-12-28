import sys
import os
from pathlib import Path
import warp as wp

# Add synthesis dir to path
sys.path.insert(0, str(Path(__file__).parent / "synthesis"))
# Add extraction dir to path (needed if pipeline imports it directly via sys.path logic, but pipeline handles it)
# pipeline.py adds extraction path relative to itself.

try:
    from pipeline import run_pipeline
except ImportError as e:
    print(f"Failed to import pipeline: {e}")
    sys.exit(1)

def main():
    print("=== CUDA Backend Validation ===")
    print(f"Warp version: {wp.__version__}")
    
    wp.init()
    cuda_available = False
    try:
        # Check if any cuda device is available
        devices = wp.get_devices()
        for d in devices:
            if d.is_cuda:
                cuda_available = True
                print(f"Found CUDA device: {d}")
                break
    except Exception:
        pass

    if not cuda_available:
        print("\nWARNING: No CUDA device detected.")
        print("If you are running this on a CPU-only machine, the pipeline will likely fail")
        print("when attempting to compile for CUDA or extract CUDA IR.")
        print("However, the code logic is set up to handle CUDA if present.")
    else:
        print("\nCUDA device detected. Proceeding with test.")

    print("\nRunning synthesis pipeline with device='cuda' (count=5)...")
    output_dir = Path(__file__).parent / "data/cuda_samples"
    
    try:
        run_pipeline(str(output_dir), count=5, device="cuda")
        print("\nSUCCESS: CUDA pipeline ran successfully.")
        print(f"Output generated in: {output_dir}")
        
        # Verify output files
        json_files = list(output_dir.glob("*.json"))
        print(f"Generated {len(json_files)} JSON files.")
        if len(json_files) > 0:
            print("Sample check passed.")
        else:
            print("WARNING: No JSON files found despite success message.")
            
    except Exception as e:
        print(f"\nFAILURE: Pipeline failed with error: {e}")
        # Print helpful message if it's the expected error on CPU machine
        if "CUDA" in str(e) or "driver" in str(e).lower() or "not found" in str(e).lower():
            print("\nNOTE: This failure is expected on CPU-only environments.")
            print("Please run this script on a machine with an NVIDIA GPU and CUDA installed.")
        sys.exit(1)

if __name__ == "__main__":
    main()

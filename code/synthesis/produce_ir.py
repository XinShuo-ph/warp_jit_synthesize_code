"""
Production script for generating CUDA IR pairs without a GPU.
"""
import sys
import argparse
import json
import re
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from batch_generator import generate_batch

def validate_cuda_ir(output_dir: Path) -> dict:
    """
    Validate that generated JSONs contain CUDA-specific keywords.
    """
    print(f"Validating CUDA IR in {output_dir}...")
    
    cuda_keywords = [
        r"blockDim",
        r"threadIdx",
        r"gridDim",
        r"blockIdx",
        r"__global__",
        r"__device__",
        r"wp::launch_bounds_t"
    ]
    
    results = {
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "failures": []
    }
    
    for json_file in output_dir.glob("pair_*.json"):
        results["total"] += 1
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            code = data.get("cpp_forward", "")
            
            # Check for at least one CUDA keyword
            # Note: Warp generates C++ that calls CUDA builtins, so we look for those signatures
            # typically: blockDim.x, threadIdx.x, etc.
            
            is_valid = False
            for keyword in cuda_keywords:
                if re.search(keyword, code):
                    is_valid = True
                    break
            
            if is_valid:
                results["valid"] += 1
            else:
                results["invalid"] += 1
                results["failures"].append(str(json_file.name))
                
        except Exception as e:
            results["invalid"] += 1
            results["failures"].append(f"{json_file.name} ({str(e)})")
            
    print(f"Validation complete: {results['valid']}/{results['total']} valid.")
    return results

def main():
    parser = argparse.ArgumentParser(description="Produce CUDA IR pairs")
    parser.add_argument("-n", "--count", type=int, default=100, help="Number of pairs")
    parser.add_argument("-o", "--output", default="data/cuda_v1", help="Output directory")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    
    # Run generation
    # Force device="cuda"
    print(f"Starting production run for {args.count} pairs...")
    stats = generate_batch(
        n=args.count, 
        output_dir=output_path, 
        seed=args.seed,
        chunk_size=min(500, args.count),
        device="cuda"
    )
    
    # Run validation
    val_stats = validate_cuda_ir(output_path)
    
    # Update stats file
    stats_file = output_path / "generation_stats.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            full_stats = json.load(f)
        
        full_stats["validation"] = val_stats
        
        with open(stats_file, 'w') as f:
            json.dump(full_stats, f, indent=2)

if __name__ == "__main__":
    main()

"""
CUDA Production Pipeline.
Generates standalone CUDA C++ code from Python kernels.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "synthesis"))
sys.path.insert(0, str(Path(__file__).parent))

from generator import generate_kernel, GENERATORS
from code_generator import python_to_cuda
from compile_cuda import compile_cuda_to_ptx, check_nvcc_available


def generate_cuda_production_samples(n_per_category=5, output_dir=None):
    """
    Generate standalone CUDA code samples from Python kernels.
    
    Args:
        n_per_category: Number of samples per kernel category
        output_dir: Output directory for generated code
    
    Returns:
        List of generated samples
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data" / "cuda_production"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    categories = list(GENERATORS.keys())
    samples = []
    stats = {cat: 0 for cat in categories}
    
    print("=" * 70)
    print("CUDA Production Code Generation")
    print("=" * 70)
    print(f"Generating {n_per_category} samples per category")
    print(f"Categories: {len(categories)}")
    print(f"Output: {output_path}")
    print()
    
    has_nvcc = check_nvcc_available()
    if has_nvcc:
        print("✓ nvcc available - will generate PTX assembly")
    else:
        print("✗ nvcc not available - skipping PTX generation")
    print()
    
    sample_idx = 0
    
    for cat in categories:
        print(f"Generating {cat}...")
        
        for i in range(n_per_category):
            try:
                # Generate Python kernel
                spec = generate_kernel(cat, seed=hash(f"{cat}_{i}") % 10000)
                python_source = spec.source
                
                # Convert to CUDA
                result = python_to_cuda(python_source)
                cuda_code = result['cuda_code']
                makefile = result['makefile']
                kernel_name = result['kernel_name']
                
                # Save files
                sample_dir = output_path / f"sample_{sample_idx:04d}_{cat}"
                sample_dir.mkdir(exist_ok=True)
                
                # Save Python source
                py_file = sample_dir / f"{kernel_name}.py"
                with open(py_file, 'w') as f:
                    f.write(python_source)
                
                # Save CUDA code
                cu_file = sample_dir / f"{kernel_name}.cu"
                with open(cu_file, 'w') as f:
                    f.write(cuda_code)
                
                # Save Makefile
                makefile_file = sample_dir / "Makefile"
                with open(makefile_file, 'w') as f:
                    f.write(makefile)
                
                # Try to compile to PTX
                ptx_success = False
                ptx_file = None
                if has_nvcc:
                    ptx_file = sample_dir / f"{kernel_name}.ptx"
                    success, ptx_content, error = compile_cuda_to_ptx(
                        cu_file, 
                        ptx_file,
                        arch="sm_50"
                    )
                    if success:
                        ptx_success = True
                    else:
                        print(f"  ⚠ PTX compilation failed: {error[:100]}")
                
                # Save metadata
                metadata = {
                    'sample_id': sample_idx,
                    'category': cat,
                    'kernel_name': kernel_name,
                    'python_file': str(py_file.name),
                    'cuda_file': str(cu_file.name),
                    'makefile': str(makefile_file.name),
                    'ptx_file': str(ptx_file.name) if ptx_file else None,
                    'ptx_generated': ptx_success,
                    'description': spec.description,
                }
                
                metadata_file = sample_dir / "metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                samples.append(metadata)
                stats[cat] += 1
                sample_idx += 1
                
                print(f"  ✓ Sample {i+1}/{n_per_category} - {kernel_name}")
                
            except Exception as e:
                print(f"  ✗ Sample {i+1} failed: {e}")
        
        print()
    
    # Generate summary
    print("=" * 70)
    print("Generation Summary")
    print("=" * 70)
    for cat, count in sorted(stats.items()):
        print(f"  {cat:20s}: {count:3d} samples")
    print(f"  {'TOTAL':20s}: {len(samples):3d} samples")
    
    # Save global summary
    summary = {
        'total_samples': len(samples),
        'categories': stats,
        'has_ptx': has_nvcc,
        'output_directory': str(output_path),
        'samples': samples
    }
    
    summary_file = output_path / "production_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved to {summary_file}")
    
    return samples


def verify_cuda_samples(samples_dir):
    """Verify generated CUDA samples."""
    samples_path = Path(samples_dir)
    
    if not samples_path.exists():
        print(f"✗ Directory not found: {samples_dir}")
        return
    
    sample_dirs = sorted([d for d in samples_path.iterdir() if d.is_dir()])
    
    print("=" * 70)
    print(f"Verifying {len(sample_dirs)} CUDA samples")
    print("=" * 70)
    
    valid = 0
    invalid = 0
    
    for sample_dir in sample_dirs:
        # Check for required files
        metadata_file = sample_dir / "metadata.json"
        
        if not metadata_file.exists():
            print(f"✗ {sample_dir.name}: missing metadata.json")
            invalid += 1
            continue
        
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        py_file = sample_dir / metadata['python_file']
        cu_file = sample_dir / metadata['cuda_file']
        
        if py_file.exists() and cu_file.exists():
            valid += 1
        else:
            print(f"✗ {sample_dir.name}: missing files")
            invalid += 1
    
    print(f"\n✓ Valid: {valid}/{len(sample_dirs)}")
    if invalid > 0:
        print(f"✗ Invalid: {invalid}/{len(sample_dirs)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate CUDA production code")
    parser.add_argument("-n", "--samples-per-category", type=int, default=5,
                        help="Number of samples per category")
    parser.add_argument("-o", "--output", default=None,
                        help="Output directory")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing samples")
    parser.add_argument("--verify-dir", default="/workspace/data/cuda_production",
                        help="Directory to verify")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_cuda_samples(args.verify_dir)
    else:
        generate_cuda_production_samples(args.samples_per_category, args.output)

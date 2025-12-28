"""
Generate CUDA samples with backward pass support.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))
sys.path.insert(0, str(Path(__file__).parent))

import warp as wp
from ir_extractor import extract_ir
from generator import generate_kernel
import json
import tempfile
import importlib.util


def compile_kernel_from_source(source: str, kernel_name: str):
    """Compile kernel from source string."""
    module_source = f'''import warp as wp

{source}
'''
    
    # Create temp file
    import hashlib
    source_hash = hashlib.md5(source.encode()).hexdigest()[:8]
    temp_dir = Path(tempfile.gettempdir()) / "warp_cuda_backward"
    temp_dir.mkdir(exist_ok=True)
    
    module_name = f"cuda_backward_{kernel_name}_{source_hash}"
    temp_file = temp_dir / f"{module_name}.py"
    
    with open(temp_file, 'w') as f:
        f.write(module_source)
    
    # Import
    spec = importlib.util.spec_from_file_location(module_name, temp_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        del sys.modules[module_name]
        raise RuntimeError(f"Failed to load: {e}")
    
    kernel = getattr(module, kernel_name, None)
    if kernel is None:
        del sys.modules[module_name]
        raise RuntimeError(f"Kernel {kernel_name} not found")
    
    return kernel


def generate_cuda_sample_with_backward(category: str, seed: int = None):
    """Generate a CUDA sample with forward and backward passes."""
    
    # Generate kernel spec
    spec = generate_kernel(category, seed)
    
    # Compile kernel
    kernel = compile_kernel_from_source(spec.source, spec.name)
    
    # Extract CPU IR (for comparison)
    cpu_result = extract_ir(kernel, device="cpu", include_backward=True)
    
    # Extract CUDA IR with backward
    cuda_result = extract_ir(kernel, device="cuda", include_backward=True)
    
    return {
        "python_source": spec.source,
        "cpu_forward": cpu_result["forward_code"],
        "cpu_backward": cpu_result["backward_code"],
        "cuda_forward": cuda_result["forward_code"],
        "cuda_backward": cuda_result["backward_code"],
        "metadata": {
            "kernel_name": spec.name,
            "category": spec.category,
            "description": spec.description,
            "has_backward": True,
            **spec.metadata
        }
    }


def generate_backward_dataset(samples_per_category=3):
    """Generate dataset with backward passes."""
    
    wp.init()
    
    categories = [
        "arithmetic",
        "vector", 
        "matrix",
        "control_flow",
        "math",
    ]
    
    output_dir = Path(__file__).parent.parent.parent / "data" / "cuda_backward_samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("CUDA Backward Pass Dataset Generation")
    print("=" * 70)
    print(f"Generating {samples_per_category} samples per category")
    print(f"Categories: {len(categories)}")
    print()
    
    all_samples = []
    stats = {}
    
    for cat in categories:
        print(f"Generating {cat} with backward pass...")
        category_samples = []
        
        for i in range(samples_per_category):
            try:
                sample = generate_cuda_sample_with_backward(
                    cat, 
                    seed=hash(f"{cat}_{i}") % 10000
                )
                category_samples.append(sample)
                print(f"  ✓ Sample {i+1}/{samples_per_category}")
            except Exception as e:
                print(f"  ✗ Sample {i+1} failed: {e}")
        
        stats[cat] = len(category_samples)
        all_samples.extend(category_samples)
        print()
    
    # Save samples
    print("Saving samples...")
    for i, sample in enumerate(all_samples):
        cat = sample["metadata"]["category"]
        filename = f"backward_{cat}_{i:04d}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(sample, f, indent=2)
    
    print(f"✓ Saved {len(all_samples)} samples to {output_dir}")
    
    # Statistics
    print("\n" + "=" * 70)
    print("Statistics")
    print("=" * 70)
    for cat, count in sorted(stats.items()):
        print(f"  {cat:20s}: {count:3d} samples")
    print(f"  {'TOTAL':20s}: {len(all_samples):3d} samples")
    
    # Summary
    summary = {
        "total_samples": len(all_samples),
        "includes_backward": True,
        "devices": ["cpu", "cuda"],
        "categories": stats,
        "output_directory": str(output_dir)
    }
    
    with open(output_dir / "backward_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    return all_samples


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--samples-per-category", type=int, default=3)
    args = parser.parse_args()
    
    generate_backward_dataset(args.samples_per_category)

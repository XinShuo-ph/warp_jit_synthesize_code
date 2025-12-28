"""Test CUDA IR extraction for all kernel types."""
import sys
import json
sys.path.append('/workspace/cuda/code/extraction')
sys.path.append('/workspace/cuda/code/synthesis')

import warp as wp
from cuda_ir_extractor import extract_cuda_ir
from generator import generate_kernel, GENERATORS

wp.init()

print("=" * 70)
print("Testing CUDA IR Extraction for All Kernel Types")
print("=" * 70)

results = []

for i, category in enumerate(GENERATORS.keys(), 1):
    print(f"\n{i}. Testing {category.upper()} kernel...")
    
    try:
        # Generate kernel spec
        spec = generate_kernel(category, seed=1000+i)
        
        # Write to temp file and import
        import tempfile
        from pathlib import Path
        import importlib.util
        
        temp_dir = Path(tempfile.gettempdir()) / "cuda_test"
        temp_dir.mkdir(exist_ok=True)
        
        module_name = f"test_{category}_{i}"
        temp_file = temp_dir / f"{module_name}.py"
        
        module_source = f'''import warp as wp

{spec.source}
'''
        
        with open(temp_file, 'w') as f:
            f.write(module_source)
        
        # Import module
        spec_loader = importlib.util.spec_from_file_location(module_name, temp_file)
        module = importlib.util.module_from_spec(spec_loader)
        sys.modules[module_name] = module
        spec_loader.loader.exec_module(module)
        
        # Get kernel
        kernel = getattr(module, spec.name)
        
        # Extract CUDA IR
        ir = extract_cuda_ir(kernel, device="cuda", include_backward=False)
        
        # Verify CUDA-specific patterns
        forward = ir['forward_code']
        checks = {
            'has_forward': forward is not None,
            'is_cuda_kernel': 'cuda_kernel_forward' in forward if forward else False,
            'has_extern_c': 'extern' in forward if forward else False,
            'has_global': '__global__' in forward if forward else False,
            'has_grid_loop': 'blockDim' in forward and 'blockIdx' in forward if forward else False,
            'has_shared_mem': 'tile_shared_storage' in forward if forward else False,
        }
        
        # Print results
        print(f"   Kernel: {spec.name}")
        print(f"   ✓ Extraction successful")
        print(f"   - Forward code: {len(forward) if forward else 0} chars")
        print(f"   - CUDA kernel: {'✓' if checks['is_cuda_kernel'] else '✗'}")
        print(f"   - extern C: {'✓' if checks['has_extern_c'] else '✗'}")
        print(f"   - __global__: {'✓' if checks['has_global'] else '✗'}")
        print(f"   - Grid-stride loop: {'✓' if checks['has_grid_loop'] else '✗'}")
        print(f"   - Shared memory: {'✓' if checks['has_shared_mem'] else '✗'}")
        
        # Save sample
        sample = {
            "python_source": ir["python_source"],
            "cuda_forward": ir["forward_code"],
            "metadata": ir["metadata"]
        }
        
        output_file = f"/workspace/cuda/data/samples/cuda_{category}_{i:02d}.json"
        with open(output_file, 'w') as f:
            json.dump(sample, f, indent=2)
        
        results.append({
            "category": category,
            "success": True,
            "checks": checks
        })
        
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        results.append({
            "category": category,
            "success": False,
            "error": str(e)
        })

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
successful = sum(1 for r in results if r.get('success', False))
print(f"Successful: {successful}/{len(results)}")

for r in results:
    if r.get('success'):
        checks = r['checks']
        status = "✓" if all(checks.values()) else "⚠"
        print(f"  {status} {r['category']}: {sum(checks.values())}/{len(checks)} checks passed")
    else:
        print(f"  ✗ {r['category']}: {r.get('error', 'Unknown error')}")

print("\nSaved samples to: /workspace/cuda/data/samples/")

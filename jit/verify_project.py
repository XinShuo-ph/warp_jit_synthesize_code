#!/usr/bin/env python3
"""
Quick verification script for JAX JIT Code Synthesis project
"""
import sys
import os
import json

def verify_project():
    """Verify all project components"""
    print("=" * 70)
    print("JAX JIT Code Synthesis - Project Verification")
    print("=" * 70)
    
    checks = []
    
    # Check 1: Dataset exists
    dataset_path = "data/m5_dataset_final.json"
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        checks.append(("Dataset exists", True, f"{len(data)} pairs"))
    else:
        checks.append(("Dataset exists", False, "Not found"))
    
    # Check 2: IR Extractor
    try:
        sys.path.append('code/extraction')
        from ir_extractor import IRExtractor
        import jax.numpy as jnp
        e = IRExtractor()
        ir = e.extract(lambda x: x+1, jnp.array([1.0]))
        checks.append(("IR Extractor", True, f"{len(ir)} IR types"))
    except Exception as ex:
        checks.append(("IR Extractor", False, str(ex)[:30]))
    
    # Check 3: Kernel Generator
    try:
        sys.path.append('code/synthesis')
        from generator import KernelGenerator
        g = KernelGenerator(42)
        k = g.generate_arithmetic()
        checks.append(("Kernel Generator", True, k['operation']))
    except Exception as ex:
        checks.append(("Kernel Generator", False, str(ex)[:30]))
    
    # Check 4: Poisson Solver
    try:
        sys.path.append('code/examples')
        from poisson_solver import PoissonSolver1D, forcing_sin_1d
        s = PoissonSolver1D(21)
        u = s.solve(forcing_sin_1d)
        checks.append(("Poisson Solver", True, f"shape {u.shape}"))
    except Exception as ex:
        checks.append(("Poisson Solver", False, str(ex)[:30]))
    
    # Check 5: Documentation
    docs = ["instructions_jax.md", "PROJECT_SUMMARY.md", "README.md", 
            "notes/jax_basics.md", "notes/ir_format.md", "notes/data_stats.md"]
    doc_exists = sum(1 for d in docs if os.path.exists(d))
    checks.append(("Documentation", doc_exists == len(docs), f"{doc_exists}/{len(docs)} files"))
    
    # Check 6: Task files
    tasks = [f"tasks/m{i}_tasks.md" for i in range(1, 6)]
    task_exists = sum(1 for t in tasks if os.path.exists(t))
    checks.append(("Task Files", task_exists == 5, f"{task_exists}/5 files"))
    
    # Print results
    print()
    all_pass = True
    for name, passed, detail in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {name:20s}: {detail}")
        if not passed:
            all_pass = False
    
    print("\n" + "=" * 70)
    if all_pass:
        print("✅ ALL CHECKS PASSED - Project is complete and working!")
    else:
        print("⚠️  Some checks failed - see details above")
    print("=" * 70)
    
    return all_pass


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__)))
    verify_project()

#!/usr/bin/env python3
"""Comprehensive test of merged codebase capabilities."""

import sys
import os

print("=" * 70)
print("COMPREHENSIVE MERGE VALIDATION TEST")
print("=" * 70)

# Test 1: Generator has 10 types
print("\n[1/5] Testing generator has 10 kernel types...")
sys.path.insert(0, 'code/synthesis')
from generator import KernelGenerator

gen = KernelGenerator()
methods = [m for m in dir(gen) if m.startswith('gen_')]
print(f"   ✓ Found {len(methods)} generator methods")
expected = ['gen_arithmetic', 'gen_conditional', 'gen_loop', 'gen_math_func', 
            'gen_vector', 'gen_atomic', 'gen_nested_loop', 'gen_multi_conditional',
            'gen_combined', 'gen_with_scalar_param']
for exp in expected:
    assert exp in methods, f"Missing generator: {exp}"
print(f"   ✓ All expected generators present: {', '.join(sorted(methods))}")

# Test 2: Pipeline can generate samples
print("\n[2/5] Testing pipeline generation...")
import subprocess
result = subprocess.run([
    sys.executable, 'code/synthesis/pipeline.py',
    '--count', '5', '--output', '/tmp/comprehensive_test', '--seed', '999'
], capture_output=True, text=True)
assert result.returncode == 0, f"Pipeline failed: {result.stderr}"
assert 'Generated 5 valid pairs' in result.stdout, "Pipeline didn't generate expected output"
print("   ✓ Pipeline generated 5 samples successfully")

# Test 3: Samples have forward and backward IR
print("\n[3/5] Testing output format...")
import json
import glob
samples = glob.glob('/tmp/comprehensive_test/*.json')
assert len(samples) >= 5, f"Expected 5 samples, found {len(samples)}"
sample = json.load(open(samples[0]))
required_keys = ['id', 'kernel_name', 'kernel_type', 'python_source', 
                 'cpp_ir_forward', 'cpp_ir_backward', 'metadata']
for key in required_keys:
    assert key in sample, f"Missing key: {key}"
    assert sample[key], f"Empty value for key: {key}"
print(f"   ✓ Sample has all required fields")
print(f"   ✓ Sample type: {sample['kernel_type']}")
print(f"   ✓ Forward IR length: {len(sample['cpp_ir_forward'])} chars")
print(f"   ✓ Backward IR length: {len(sample['cpp_ir_backward'])} chars")

# Test 4: Examples run successfully
print("\n[4/5] Testing examples...")
result = subprocess.run([sys.executable, 'code/examples/01_simple_kernel.py'],
                       capture_output=True, text=True, timeout=30)
assert result.returncode == 0, f"Example failed: {result.stderr}"
assert 'PASSED' in result.stdout or 'Match: True' in result.stdout, "Example validation failed"
print("   ✓ 01_simple_kernel.py ran successfully")

result = subprocess.run([sys.executable, 'code/examples/ex00_add.py'],
                       capture_output=True, text=True, timeout=30)
assert result.returncode == 0, f"Example failed: {result.stderr}"
print("   ✓ ex00_add.py ran successfully")

# Test 5: Utility files exist
print("\n[5/5] Testing utility files...")
utils = [
    'code/extraction/validate_extraction.py',
    'code/extraction/debug_extraction.py',
    'code/extraction/debug_loop.py',
    'code/synthesis/validate_dataset.py',
    'code/synthesis/analyze_dataset.py',
    'tests/cases/case_arith.py',
    'tests/cases/case_loop.py',
    'tests/fixture_kernels.py',
]
for util in utils:
    assert os.path.exists(util), f"Missing utility: {util}"
print(f"   ✓ All {len(utils)} utility files present")

print("\n" + "=" * 70)
print("ALL TESTS PASSED ✅")
print("=" * 70)
print("\nMerged codebase is production-ready!")
print("\nCapabilities:")
print("  • 10 kernel type generators")
print("  • Forward + backward IR extraction")
print("  • Comprehensive validation tools")
print("  • Debug utilities")
print("  • Example progression (beginner → advanced)")
print("  • Categorized test suite")

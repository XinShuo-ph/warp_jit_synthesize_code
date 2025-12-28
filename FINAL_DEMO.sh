#!/bin/bash
# Final demonstration of merged codebase capabilities

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║              FINAL DEMONSTRATION - MERGED CODEBASE                       ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

# Demo 1: Generate all 10 types
echo "━━━ Demo 1: Generate samples from all 10 kernel types ━━━"
python3 code/synthesis/pipeline.py -n 30 -o data/demo_all_types --seed 88888
echo ""

# Demo 2: Batch generation
echo "━━━ Demo 2: Batch generation (50 samples) ━━━"
python3 code/synthesis/batch_generator.py -n 50 -o data/demo_batch --seed 99999
echo ""

# Demo 3: Specific categories (new types from 9177)
echo "━━━ Demo 3: Generate nested_loop samples ━━━"
python3 code/synthesis/pipeline.py -n 3 -o data/demo_nested -c nested_loop --seed 11111
echo ""

echo "━━━ Demo 4: Generate combined samples ━━━"
python3 code/synthesis/pipeline.py -n 3 -o data/demo_combined -c combined --seed 22222
echo ""

# Demo 5: Verify sample quality
echo "━━━ Demo 5: Sample quality verification ━━━"
python3 << 'PYEOF'
import json
from pathlib import Path

sample_file = Path("data/demo_nested/synth_0000.json")
if sample_file.exists():
    data = json.load(open(sample_file))
    print(f"✓ Python source: {len(data['python_source'])} chars")
    print(f"✓ C++ IR code: {len(data['cpp_forward'])} chars")
    print(f"✓ Category: {data['metadata']['category']}")
    print(f"✓ Description: {data['metadata']['description']}")
    print("\nSample Python source (first 5 lines):")
    print('\n'.join(data['python_source'].split('\n')[:5]))
PYEOF
echo ""

# Demo 6: Count total samples
echo "━━━ Demo 6: Total samples generated ━━━"
python3 << 'PYEOF'
from pathlib import Path

total = 0
for d in Path("data").iterdir():
    if d.is_dir():
        json_files = [f for f in d.glob("*.json") if "stats" not in f.name]
        if json_files:
            print(f"  {d.name:25s}: {len(json_files):3d} samples")
            total += len(json_files)
print(f"\n  {'TOTAL':25s}: {total:3d} samples")
PYEOF
echo ""

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                      DEMONSTRATION COMPLETE                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"

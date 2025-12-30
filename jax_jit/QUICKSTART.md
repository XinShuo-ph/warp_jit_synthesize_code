# Quick Start Guide

## Installation (30 seconds)

```bash
# Install JAX
pip install jax jaxlib

# Verify installation
python3 -c "import jax; print(f'JAX {jax.__version__} installed')"
```

## Generate Your First Dataset (2 minutes)

```bash
cd /workspace/jax_jit

# Generate 100 pairs
python3 code/synthesis/batch_generator.py --count 100

# Expected output:
# ================================================================================
# BATCH GENERATION: Target 100 pairs
# ================================================================================
# ...
# Total pairs: 100
# Rate: ~136 pairs/sec
```

## Inspect the Data (1 minute)

```bash
# View statistics
python3 code/synthesis/validate_dataset.py

# Check a sample file
cat data/samples/*.json | head -30
```

## Basic Usage Examples

### Example 1: Extract IR from Your Function

```python
import jax.numpy as jnp
import sys
sys.path.insert(0, '/workspace/jax_jit/code/extraction')
from ir_extractor import extract_ir

# Define your function
def my_function(x, y):
    return jnp.sqrt(x ** 2 + y ** 2)  # Euclidean distance

# Extract IR
x = jnp.array([3.0, 4.0, 5.0])
y = jnp.array([4.0, 3.0, 12.0])

pair = extract_ir(my_function, x, y)

print("Python Source:")
print(pair.python_source)
print("\nStableHLO IR:")
print(pair.stablehlo_ir)
print(f"\nCost: {pair.cost_analysis['flops']} FLOPs")
```

### Example 2: Generate Specific Category

```python
import sys
sys.path.insert(0, '/workspace/jax_jit/code/synthesis')
from pipeline import SynthesisPipeline

pipeline = SynthesisPipeline(output_dir="./my_samples")

# Generate matrix operations
for i in range(10):
    pair = pipeline.generate_single(category='matrix', save=True)
    print(f"Generated: {pair['function_name']}")
```

### Example 3: Batch Generation

```python
import sys
sys.path.insert(0, '/workspace/jax_jit/code/synthesis')
from batch_generator import BatchGenerator

generator = BatchGenerator(output_dir="./my_dataset", seed=42)

# Generate balanced dataset
stats = generator.generate_balanced_dataset(target_count=1000)

print(f"Generated {stats['total_pairs']} pairs in {stats['total_time']:.1f}s")
print(f"Categories: {stats['categories']}")
```

### Example 4: Load and Use Generated Data

```python
import json
from pathlib import Path

# Load all pairs
data_dir = Path("/workspace/jax_jit/data/samples")
pairs = []

for json_file in data_dir.glob("*.json"):
    with open(json_file) as f:
        pairs.append(json.load(f))

# Filter by category
matrix_pairs = [p for p in pairs if p['category'] == 'matrix']
print(f"Found {len(matrix_pairs)} matrix operations")

# Analyze complexity
high_complexity = [p for p in pairs if len(p['stablehlo_ir'].splitlines()) > 15]
print(f"Found {len(high_complexity)} complex functions")

# Extract features for ML
for pair in pairs[:5]:
    features = {
        'name': pair['function_name'],
        'python_lines': len(pair['python_source'].splitlines()),
        'ir_lines': len(pair['stablehlo_ir'].splitlines()),
        'flops': pair['cost_analysis'].get('flops', 0),
    }
    print(features)
```

## Command-Line Usage

### Generate Custom Dataset

```bash
# Generate 1000 pairs
python3 code/synthesis/batch_generator.py --count 1000 --batch-size 100

# Balanced across categories
python3 code/synthesis/batch_generator.py --count 700 --balanced

# With custom seed for reproducibility
python3 code/synthesis/batch_generator.py --count 500 --seed 12345

# Custom output directory
python3 code/synthesis/batch_generator.py --count 500 --output-dir ./custom_data

# Quiet mode (no progress output)
python3 code/synthesis/batch_generator.py --count 1000 --quiet
```

### Validate Dataset

```bash
# Full validation and analysis
python3 code/synthesis/validate_dataset.py

# Check specific directory
cd /workspace/jax_jit && python3 -c "
import sys
sys.path.insert(0, 'code/synthesis')
from validate_dataset import validate_dataset, analyze_dataset
validate_dataset('./custom_data')
analyze_dataset('./custom_data')
"
```

## Typical Workflows

### Workflow 1: Research Dataset

```bash
# 1. Generate diverse dataset
python3 code/synthesis/batch_generator.py --count 5000 --balanced

# 2. Validate quality
python3 code/synthesis/validate_dataset.py

# 3. Split for train/val/test
python3 -c "
import json, random
from pathlib import Path

files = list(Path('data/samples').glob('*.json'))
random.shuffle(files)

train_size = int(0.8 * len(files))
val_size = int(0.1 * len(files))

train_files = files[:train_size]
val_files = files[train_size:train_size+val_size]
test_files = files[train_size+val_size:]

print(f'Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}')
"
```

### Workflow 2: Incremental Generation

```bash
# Generate in chunks
for i in {1..10}; do
  echo "Batch $i"
  python3 code/synthesis/batch_generator.py --count 1000 --seed $i
  sleep 1
done

# Validate all
python3 code/synthesis/validate_dataset.py
```

### Workflow 3: Category-Specific Study

```python
# Generate 1000 samples per category
import sys
sys.path.insert(0, '/workspace/jax_jit/code/synthesis')
from batch_generator import BatchGenerator

categories = ['arithmetic', 'conditional', 'reduction', 'matrix', 
             'elementwise', 'broadcasting', 'composite']

for category in categories:
    print(f"\nGenerating {category}...")
    gen = BatchGenerator(output_dir=f"./data/{category}")
    
    count = 0
    while count < 1000:
        try:
            pair = gen.pipeline.generate_single(category=category, save=True)
            if pair:
                count += 1
                if count % 100 == 0:
                    print(f"  {count}/1000")
        except Exception as e:
            continue
    
    print(f"Completed {category}: {count} pairs")
```

## Performance Tuning

### Maximize Generation Speed

```python
# Use larger batch sizes
python3 code/synthesis/batch_generator.py --count 10000 --batch-size 500

# Disable verbose output
python3 code/synthesis/batch_generator.py --count 10000 --quiet
```

### Reduce Memory Usage

```python
# Generate in streaming fashion
import sys
sys.path.insert(0, '/workspace/jax_jit/code/synthesis')
from pipeline import SynthesisPipeline

pipeline = SynthesisPipeline()

for i in range(10000):
    pair = pipeline.generate_single(save=True)
    if i % 1000 == 0:
        print(f"Progress: {i}/10000")
    # Pair is saved to disk, not kept in memory
```

## Troubleshooting

### Issue: Generation Fails

```bash
# Check JAX installation
python3 -c "import jax; print(jax.devices())"

# Test basic IR extraction
python3 code/extraction/test_ir_extractor.py
```

### Issue: Slow Generation

```bash
# Check JAX is using XLA
python3 -c "
import jax
print(f'JAX backend: {jax.default_backend()}')
print(f'Devices: {jax.devices()}')
"

# Use batch mode
python3 code/synthesis/batch_generator.py --count 100 --batch-size 100
```

### Issue: Invalid Pairs

```bash
# Run validation to identify issues
python3 code/synthesis/validate_dataset.py

# Check specific file
python3 -c "
import json, sys
sys.path.insert(0, 'code/synthesis')
from pipeline import SynthesisPipeline

pipeline = SynthesisPipeline()

with open('data/samples/FILENAME.json') as f:
    pair = json.load(f)

is_valid = pipeline.validate_pair(pair)
print(f'Valid: {is_valid}')
"
```

## Next Steps

1. **Explore Examples**: Check `code/examples/explore_jax_ir.py`
2. **Read Technical Notes**: See `notes/technical_notes.md`
3. **Scale Up**: Generate 10k+ pairs for training
4. **Extend**: Add custom function categories
5. **Integrate**: Use data with your LLM training pipeline

## Support

- Review `README.md` for architecture details
- Check `notes/technical_notes.md` for implementation details
- Run tests: `python3 code/extraction/test_ir_extractor.py`

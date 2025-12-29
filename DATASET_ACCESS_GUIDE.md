# Dataset Access and Usage Guide

## ✅ Successfully Pushed to GitHub

The production datasets have been successfully pushed to the remote repository at:
**https://github.com/XinShuo-ph/warp_jit_synthesize_code**

**Branch**: `cursor/dataset-and-report-generation-891a`

---

## 📦 What Was Pushed

### Compressed Archives (Git LFS)
- **`production/cpu_dataset.tar.gz`** - 22 MB
  - Contains 69,000 CPU code samples (200.82 MB uncompressed)
  - Python→C++ IR pairs
  
- **`production/cuda_dataset.tar.gz`** - 20 MB  
  - Contains 60,000 CUDA code samples (201.44 MB uncompressed)
  - Python→CUDA IR pairs

### Documentation
- **`report/REPORT.md`** - Comprehensive technical report (8,120 words)
- **`README_PRODUCTION.md`** - Project summary
- **`COMPLETION_SUMMARY.md`** - Final results
- **`INDEX.md`** - Navigation guide
- **`production/cpu_analysis.md`** - CPU methodology
- **`production/cuda_analysis.md`** - CUDA methodology
- **`GIT_LFS_ISSUE.md`** - Explanation of compression strategy

### Scripts
- **`production/scripts/`** - All generation scripts
  - `cpu_production.py` - CPU dataset generator
  - `cuda_production.py` - CUDA dataset generator
  - `cpu_generator.py` - 11 kernel type generators
  - `cpu_ir_extractor.py` - IR extraction utilities
  - `cpu_batch_generator.py` - Batch processing

---

## 🚀 How to Use the Datasets

### Step 1: Clone the Repository

```bash
git clone https://github.com/XinShuo-ph/warp_jit_synthesize_code
cd warp_jit_synthesize_code
git checkout cursor/dataset-and-report-generation-891a
```

### Step 2: Install Git LFS (if not already installed)

```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# macOS
brew install git-lfs

# Windows
# Download from: https://git-lfs.github.com/

# Initialize Git LFS
git lfs install
```

### Step 3: Pull LFS Files

```bash
git lfs pull
```

This will download the compressed archives (`cpu_dataset.tar.gz` and `cuda_dataset.tar.gz`).

### Step 4: Extract the Datasets

```bash
cd production

# Extract CPU dataset
tar -xzf cpu_dataset.tar.gz
# Creates cpu_code/ directory with 69,000 JSON files

# Extract CUDA dataset  
tar -xzf cuda_dataset.tar.gz
# Creates cuda_code/ directory with 60,000 JSON files
```

### Step 5: Use the Data

```python
import json
from pathlib import Path

# Load a CPU sample
with open('production/cpu_code/pair_000000.json') as f:
    cpu_sample = json.load(f)
    print("Python source:")
    print(cpu_sample['python_source'])
    print("\nC++ IR:")
    print(cpu_sample['cpp_forward'])
    print("\nMetadata:")
    print(cpu_sample['metadata'])

# Load a CUDA sample
with open('production/cuda_code/pair_000000.json') as f:
    cuda_sample = json.load(f)
    print("CUDA IR includes grid-stride loops:")
    print('blockDim' in cuda_sample['cpp_forward'])  # True
```

---

## 📊 Dataset Statistics

### CPU Dataset
- **Compressed size**: 22 MB
- **Uncompressed size**: 200.82 MB (actual file content)
- **Disk usage**: 349 MB (with filesystem overhead)
- **Total samples**: 69,000
- **Average sample size**: 2.98 KB
- **Format**: JSON (Python→C++ pairs)
- **Device**: CPU backend

### CUDA Dataset
- **Compressed size**: 20 MB
- **Uncompressed size**: 201.44 MB (actual file content)
- **Disk usage**: 315 MB (with filesystem overhead)
- **Total samples**: 60,000
- **Average sample size**: 3.44 KB
- **Format**: JSON (Python→CUDA pairs)
- **Device**: CUDA backend

### Combined
- **Total compressed**: 42 MB
- **Total uncompressed**: 402 MB
- **Total samples**: 129,000
- **Kernel categories**: 11 types (uniform distribution)

---

## 📄 Sample Data Format

Each JSON file contains:

```json
{
  "python_source": "@wp.kernel\ndef kernel_name(...):\n    ...",
  "cpp_forward": "void kernel_name_..._kernel_forward(...) { ... }",
  "metadata": {
    "kernel_name": "kernel_name",
    "category": "arithmetic|vector|matrix|control_flow|math|...",
    "device": "cpu|cuda",
    "description": "...",
    "seed": 12345,
    ...
  }
}
```

---

## 🔧 Regenerating Datasets

If you want to regenerate the datasets with different parameters:

```bash
cd production/scripts

# Install dependencies
pip install warp-lang

# Generate CPU dataset (200MB target)
python3 cpu_production.py

# Generate CUDA dataset (200MB target)  
python3 cuda_production.py
```

---

## 📚 Documentation

### Main Report
Read the comprehensive technical report:
```bash
cat report/REPORT.md | less
```

Or view on GitHub:
https://github.com/XinShuo-ph/warp_jit_synthesize_code/blob/cursor/dataset-and-report-generation-891a/report/REPORT.md

### Quick Summaries
- **Project Summary**: `README_PRODUCTION.md`
- **Completion Status**: `COMPLETION_SUMMARY.md`
- **Navigation**: `INDEX.md`

---

## 🎯 Use Cases

### 1. LLM Training
Fine-tune code generation models on Python→IR translation:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your model
model = AutoModelForCausalLM.from_pretrained("codegen-model")

# Prepare training data from JSON files
training_pairs = []
for file in Path("production/cpu_code").glob("pair_*.json"):
    with open(file) as f:
        sample = json.load(f)
        training_pairs.append({
            "input": sample['python_source'],
            "output": sample['cpp_forward']
        })

# Fine-tune...
```

### 2. Compiler Research
Analyze IR patterns and optimization opportunities:

```python
# Count operation types
import re
from collections import Counter

ops = Counter()
for file in Path("production/cpu_code").glob("pair_*.json"):
    with open(file) as f:
        ir = json.load(f)['cpp_forward']
        ops.update(re.findall(r'wp::(\w+)\(', ir))

print("Most common operations:")
for op, count in ops.most_common(10):
    print(f"  {op}: {count}")
```

### 3. Benchmark Creation
Use as test cases for code generation systems:

```python
# Create evaluation benchmark
benchmark = []
for cat in ['arithmetic', 'vector', 'matrix', 'control_flow']:
    files = list(Path(f"production/cpu_code").glob(f"pair_*.json"))
    # Sample 100 from each category based on metadata
    # ...
```

---

## ⚠️ Important Notes

### Git LFS Required
The compressed archives are stored using Git LFS. You **must** have Git LFS installed and run `git lfs pull` to download them. Otherwise, you'll only get pointer files (~100 bytes each).

### Extraction Required
The archives must be extracted before use. The individual JSON files are not tracked in git (excluded via `.gitignore`) to avoid the 129,000-file rate limit issue.

### Disk Space
After extraction, you'll need:
- CPU: ~349 MB disk space
- CUDA: ~315 MB disk space
- Total: ~664 MB (plus 42 MB for archives)

### Compression Strategy
We used compressed archives instead of individual files because:
- GitHub LFS rate limit: 129,000 individual files triggered rate limits
- Solution: 2 archives upload successfully
- Compression ratio: ~5:1 (402 MB → 42 MB)

---

## 🔗 Links

- **Repository**: https://github.com/XinShuo-ph/warp_jit_synthesize_code
- **Branch**: cursor/dataset-and-report-generation-891a
- **NVIDIA Warp**: https://github.com/NVIDIA/warp
- **Git LFS**: https://git-lfs.github.com/

---

## 📞 Questions?

For questions about:
- **Dataset usage**: See `report/REPORT.md`
- **Generation process**: See `production/cpu_analysis.md` and `production/cuda_analysis.md`
- **Git LFS issues**: See `GIT_LFS_ISSUE.md`
- **Project overview**: See `README_PRODUCTION.md`

---

**Status**: ✅ Datasets successfully pushed to GitHub  
**Date**: December 29, 2025  
**Branch**: cursor/dataset-and-report-generation-891a  
**Total Size**: 42 MB compressed (402 MB uncompressed)

# Remote Repository Access Guide

## ✅ All Data Successfully Pushed

All production code and datasets (408 MB) have been successfully pushed to the remote GitHub repository.

---

## 📍 Repository Location

**Repository:** `github.com/XinShuo-ph/warp_jit_synthesize_code`  
**Branch:** `cursor/dataset-and-report-generation-0622`

### Direct Links

- **Branch Home:** https://github.com/XinShuo-ph/warp_jit_synthesize_code/tree/cursor/dataset-and-report-generation-0622
- **CPU Dataset:** https://github.com/XinShuo-ph/warp_jit_synthesize_code/tree/cursor/dataset-and-report-generation-0622/datasets/cpu_code
- **CUDA Dataset:** https://github.com/XinShuo-ph/warp_jit_synthesize_code/tree/cursor/dataset-and-report-generation-0622/datasets/cuda_code
- **Technical Report:** https://github.com/XinShuo-ph/warp_jit_synthesize_code/blob/cursor/dataset-and-report-generation-0622/report/chief_scientist_report.md
- **Project Summary:** https://github.com/XinShuo-ph/warp_jit_synthesize_code/blob/cursor/dataset-and-report-generation-0622/PROJECT_SUMMARY.md

---

## 📦 What's on Remote

### Datasets (247 files, 408 MB)
- `datasets/cpu_code/` - 144 JSON files (200.25 MB)
  - batch_01/ through batch_08/
  - 71,505 CPU Python→IR training pairs
  
- `datasets/cuda_code/` - 101 JSON files (207.85 MB)
  - batch_01/ through batch_05/
  - 50,005 CUDA Python→IR training pairs

- `datasets/statistics/`
  - cpu_stats.txt - CPU dataset statistics
  - cuda_stats.txt - CUDA dataset statistics

### Production Code
- `production_code/cpu_pipeline/`
  - simple_batch_gen.py - CPU batch generator
  - generator.py - CPU kernel generator
  - ir_extractor.py - IR extraction utilities
  - pipeline.py - Pipeline utilities
  
- `production_code/cuda_pipeline/`
  - cuda_batch_gen.py - CUDA batch generator
  - cuda_generator.py - CUDA kernel generator

### Documentation
- `report/chief_scientist_report.md` - 20-page technical report
- `PROJECT_SUMMARY.md` - Executive summary
- `instructions_dataset_production.md` - Project instructions
- `PRODUCTION_STATE.md` - Final state tracking

---

## 💾 Cloning the Repository

To clone the entire repository with all data:

```bash
git clone https://github.com/XinShuo-ph/warp_jit_synthesize_code.git
cd warp_jit_synthesize_code
git checkout cursor/dataset-and-report-generation-0622
```

To clone just this branch:

```bash
git clone -b cursor/dataset-and-report-generation-0622 \
  https://github.com/XinShuo-ph/warp_jit_synthesize_code.git
```

---

## 📥 Downloading Specific Datasets

### Option 1: Clone and Navigate
```bash
git clone -b cursor/dataset-and-report-generation-0622 \
  https://github.com/XinShuo-ph/warp_jit_synthesize_code.git
cd warp_jit_synthesize_code/datasets
```

### Option 2: Sparse Checkout (Datasets Only)
```bash
git clone --no-checkout https://github.com/XinShuo-ph/warp_jit_synthesize_code.git
cd warp_jit_synthesize_code
git sparse-checkout init --cone
git sparse-checkout set datasets
git checkout cursor/dataset-and-report-generation-0622
```

### Option 3: Download Individual Files via GitHub API
```bash
# Example: Download a specific batch
curl -L -o batch_0000.json \
  https://raw.githubusercontent.com/XinShuo-ph/warp_jit_synthesize_code/cursor/dataset-and-report-generation-0622/datasets/cpu_code/batch_01/batch_0000.json
```

---

## 🔍 Verifying Download Integrity

After cloning, verify the data:

```bash
# Count dataset files
find datasets -name "*.json" | wc -l
# Expected: 245 files

# Check total size
du -sh datasets/
# Expected: ~408 MB (201M cpu_code + 209M cuda_code)

# Verify a JSON file
python3 -c "import json; data=json.load(open('datasets/cpu_code/batch_01/batch_0000.json')); print(f'Pairs: {len(data)}'); print(f'Keys: {list(data[0].keys())}')"
# Expected: Pairs: 500, Keys: ['python_source', 'ir_code', 'kernel_name']
```

---

## 🚀 Using the Datasets

### Load a Single Batch
```python
import json

# Load CPU batch
with open('datasets/cpu_code/batch_01/batch_0000.json', 'r') as f:
    cpu_pairs = json.load(f)

print(f"Loaded {len(cpu_pairs)} CPU training pairs")
print(f"Sample pair keys: {list(cpu_pairs[0].keys())}")
print(f"Python source: {cpu_pairs[0]['python_source'][:100]}...")
print(f"IR code: {cpu_pairs[0]['ir_code'][:100]}...")
```

### Load All Data
```python
import json
import glob

def load_all_pairs(dataset_path):
    """Load all training pairs from a dataset directory."""
    all_pairs = []
    json_files = glob.glob(f"{dataset_path}/**/*.json", recursive=True)
    
    for json_file in sorted(json_files):
        with open(json_file, 'r') as f:
            pairs = json.load(f)
            all_pairs.extend(pairs)
    
    return all_pairs

# Load all CPU pairs
cpu_pairs = load_all_pairs('datasets/cpu_code')
print(f"Total CPU pairs: {len(cpu_pairs)}")  # 71,505

# Load all CUDA pairs
cuda_pairs = load_all_pairs('datasets/cuda_code')
print(f"Total CUDA pairs: {len(cuda_pairs)}")  # 50,005
```

---

## 📊 Dataset Structure

Each JSON file contains an array of training pairs:

```json
[
  {
    "python_source": "@wp.kernel\ndef kernel_name(...):\n    ...",
    "ir_code": "void kernel_name_hash(...){\n    ...\n}",
    "kernel_name": "kernel_name",
    "backend": "cpu"  // or "cuda"
  },
  ...
]
```

---

## ⚠️ Git LFS Not Required

**Good news:** Git LFS is **NOT needed** for this repository!

- GitHub's file size limit: 100 MB
- Our largest files: ~2.2 MB each
- Total dataset: 408 MB across 245 files
- Average file size: 1.66 MB

All files are well under the 100 MB limit, so standard Git handles them perfectly.

---

## 🔄 Syncing with Remote

If you make local changes and want to verify sync:

```bash
# Fetch latest from remote
git fetch origin cursor/dataset-and-report-generation-0622

# Check if in sync
git diff HEAD origin/cursor/dataset-and-report-generation-0622

# Pull latest changes (if any)
git pull origin cursor/dataset-and-report-generation-0622
```

---

## 📧 Questions or Issues?

If you encounter any issues accessing the data:

1. **Check branch name:** Make sure you're on `cursor/dataset-and-report-generation-0622`
2. **Verify remote:** `git remote -v` should show the correct repository
3. **Check file count:** Should see 247 dataset files total
4. **Verify size:** Total dataset size should be ~408 MB

---

## ✅ Verification Checklist

- [x] Repository accessible at github.com/XinShuo-ph/warp_jit_synthesize_code
- [x] Branch `cursor/dataset-and-report-generation-0622` exists
- [x] 247 dataset files present on remote
- [x] Production code pushed
- [x] Technical report available
- [x] All documentation pushed
- [x] Local and remote in sync (SHA: 69d36e8a)

**Status: All data successfully pushed and accessible! 🎉**

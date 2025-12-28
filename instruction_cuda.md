# CUDA Backend Development

## Objective
Adapt the JIT code synthesis pipeline to use CUDA backend for GPU-accelerated kernel generation and IR extraction. Since no GPU device is available during development, provide concise code and test commands for later validation on actual GPU hardware.

---

## File Structure (create as needed)

```
jit/
├── instruction_cuda.md      # This file (read-only reference)
├── CUDA_STATE.md            # CRITICAL: Current progress tracker
├── code/
│   ├── extraction/          # IR extraction (CPU + CUDA)
│   │   ├── ir_extractor.py
│   │   └── cuda_extractor.py
│   ├── synthesis/           # Data synthesis pipeline
│   │   ├── generator.py
│   │   ├── pipeline.py
│   │   └── batch_generator.py
│   └── examples/            # Example kernels
├── data/                    # Generated samples (≤100 for git)
├── tests/                   # CUDA test suite for GPU validation
│   ├── test_cuda_kernels.py
│   └── run_gpu_tests.sh
└── notes/                   # Technical findings
    └── cuda_notes.md
```

---

## State Management Protocol

### On Session Start
1. Read `CUDA_STATE.md` (create if missing)
2. Read the current milestone's task list
3. Resume from documented next action

### On Session End (or ~20k tokens remaining)
1. Update `CUDA_STATE.md` with:
   - Current milestone and task
   - Exact next action (be specific)
   - Any blockers or failed attempts
   - Key findings
2. Commit working code (no broken states)
3. Push to remote
4. Stop—do not start new tasks

### CUDA_STATE.md Template
```markdown
# CUDA Development State
- **Milestone**: M1/M2
- **Task**: [task number and name]
- **Status**: in_progress | blocked | ready_for_next

## Next Action
[Exactly what to do next. Be specific enough that a new agent can execute immediately.]

## Blockers (if any)
[What's preventing progress, what was tried]

## GPU Test Commands
[Commands to run on actual GPU hardware]

## Session Log
- [date/session]: [brief summary of what was accomplished]
```

---

## Milestones

### M1: Establish CPU Baseline
**Goal**: Reproduce and validate the production CPU code as a working base

**Tasks**:
1. Study the `cursor/agent-work-merge-process-*` branches
2. Identify the best merge branch with complete pipeline
3. Copy the best code to this branch's `jit/` directory
4. Validate pipeline works with CPU backend:
   ```bash
   python jit/code/synthesis/pipeline.py --count 5
   python jit/code/extraction/ir_extractor.py
   ```
5. Document baseline in `CUDA_STATE.md`

**Done when**: CPU pipeline generates valid Python→IR pairs

### M2: CUDA Backend Adaptation
**Goal**: Adapt all components to support CUDA device and generate GPU kernel IR

**Iteration Order** (focus on one at a time):

#### Phase A: Kernel Types (Forward Pass)
For each kernel type, add CUDA support:
1. **arithmetic** - Basic arithmetic operations
2. **math** - Unary math functions (sin, cos, exp, etc.)
3. **loop** - For loop constructs
4. **conditional** - If/else branching
5. **vector** - wp.vec3 operations
6. **matrix** - wp.mat33 operations
7. **combined** - Multi-pattern kernels

Per-kernel workflow:
```python
# 1. Add device parameter to kernel generation
kernel = generate_kernel(kernel_type, device="cuda")

# 2. Extract CUDA IR
ir_code = extract_ir(kernel, device="cuda")

# 3. Create test case
def test_kernel_cuda():
    # Will be run on GPU later
    ...
```

#### Phase B: Backward Pass
Adapt gradient computation kernels for CUDA:
1. Identify kernels with backward pass
2. Add CUDA support for adjoint code generation
3. Create validation tests

#### Phase C: Batch Generation Pipeline
Update batch generator for CUDA:
1. Add `--device cuda` parameter
2. Support mixed CPU/CUDA generation
3. Parallel batch processing support

#### Phase D: Validation Tools
Create validation utilities:
1. `validate_cuda_ir.py` - Verify CUDA IR structure
2. `compare_cpu_cuda.py` - Compare CPU vs CUDA output
3. Data format validation

#### Phase E: GPU Test Suite
Create comprehensive test suite for GPU validation:
```
tests/
├── test_cuda_kernels.py      # All kernel types
├── test_cuda_backward.py     # Gradient computation
├── test_cuda_pipeline.py     # End-to-end pipeline
└── run_gpu_tests.sh          # Script for GPU execution
```

**Done when**: All components have CUDA support and test suite is ready

---

## Iteration Template

For each CUDA adaptation task:

### Step 1: Identify Changes
```python
# Document what needs to change
# - File: [filename]
# - Function: [function name]
# - Current behavior: [CPU only]
# - Target behavior: [CPU + CUDA]
```

### Step 2: Implement Changes
```python
# Add device parameter
def function(input, device="cpu"):
    if device == "cuda":
        # CUDA-specific code
    else:
        # CPU code
```

### Step 3: Create Test (for later GPU run)
```python
def test_function_cuda():
    """Run on GPU to validate CUDA support."""
    result = function(input, device="cuda")
    assert validate(result)
```

### Step 4: Document
Update `CUDA_STATE.md` with:
- What was changed
- Test commands for GPU validation
- Expected results

---

## Key Warp CUDA Concepts

### Device Selection
```python
import warp as wp

# Initialize for CUDA
wp.init()

# Check available devices
print(wp.get_devices())  # ['cpu', 'cuda:0', ...]

# Set default device
with wp.ScopedDevice("cuda:0"):
    kernel.launch(...)
```

### CUDA-specific Kernel Compilation
```python
@wp.kernel
def my_kernel(a: wp.array(dtype=float)):
    tid = wp.tid()
    a[tid] = a[tid] * 2.0

# Compile for CUDA
module = wp.Module.load(__name__)
module.codegen(device="cuda")
```

### Accessing CUDA IR
```python
# Get generated CUDA code
kernel = module.kernels["my_kernel"]
cuda_source = kernel.module.cuda_source  # .cu file content
```

---

## Git Commands Reference

```bash
# Check available merge branches
git branch -a | grep agent-work-merge

# View files in a merge branch
git ls-tree -r --name-only origin/cursor/agent-work-merge-process-{SUFFIX}

# Copy file from merge branch
git show origin/cursor/agent-work-merge-process-{SUFFIX}:path/to/file > local/path/file

# Copy directory from merge branch
git checkout origin/cursor/agent-work-merge-process-{SUFFIX} -- jit/

# Commit and push
git add -A
git commit -m "cuda: [description]"
git push origin HEAD
```

---

## Token Budget Guidelines

| Phase | Budget | Activities |
|-------|--------|------------|
| M1 Orientation | ~10k | Study branches, identify best base |
| M1 Setup | ~15k | Copy code, validate CPU baseline |
| M2 Per-Kernel | ~8k | Add CUDA support, create test |
| M2 Pipeline | ~15k | Batch generator updates |
| M2 Tests | ~20k | Create comprehensive test suite |

**Expected iterations**: ~20 (7 kernel types × 2 passes + pipeline + validation + tests)

---

## GPU Test Script Template

Create `tests/run_gpu_tests.sh`:
```bash
#!/bin/bash
# Run on a machine with NVIDIA GPU

# Check CUDA availability
python -c "import warp as wp; wp.init(); print(wp.get_devices())"

# Run kernel tests
python -m pytest tests/test_cuda_kernels.py -v

# Run pipeline test
python jit/code/synthesis/pipeline.py --device cuda --count 10

# Validate generated data
python jit/code/extraction/validate_cuda_ir.py
```

---

## Anti-Patterns (Avoid)

- ❌ Attempting to run CUDA code without GPU (it will fail)
- ❌ Generating large datasets during development
- ❌ Modifying CPU code in ways that break it
- ❌ Skipping CPU baseline validation
- ❌ Creating CUDA code without corresponding tests
- ❌ Starting new kernel before completing current one
- ❌ Committing broken code

---

## Success Criteria

M1 Complete when:
- Working CPU baseline copied from best merge branch
- Pipeline generates Python→IR pairs
- All kernel types work with CPU backend

M2 Complete when:
- All 7 kernel types have CUDA support code
- Backward pass has CUDA support
- Batch generator supports `--device cuda`
- Validation tools created
- Test suite ready for GPU execution
- `tests/run_gpu_tests.sh` script prepared
- All changes documented in `CUDA_STATE.md`

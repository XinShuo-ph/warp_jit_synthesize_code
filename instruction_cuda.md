# CUDA Backend Development

## Objective
Adapt the current production CPU code to support CUDA GPU backend and generate CUDA IR training data.

**Key Insight**: CUDA IR code generation does NOT require an actual GPU. Warp's code generator can produce CUDA C++ code on any machine - only execution requires a GPU. This means we can generate large-scale CUDA training datasets without GPU access.

---

## File Structure (create as needed)

```
jit/
├── instruction_cuda.md       # This file (read-only reference)
├── CUDA_STATE.md             # CRITICAL: Current progress, next action, blockers
├── tasks/
│   ├── cuda_tasks.md         # Task breakdown for CUDA adaptation
│   └── ...
├── code/
│   ├── extraction/           # IR extraction (CPU + CUDA)
│   │   ├── ir_extractor.py   # Updated with device parameter
│   │   └── cuda_extractor.py # CUDA-specific extraction utilities
│   ├── synthesis/            # Data synthesis pipeline
│   │   ├── generator.py      # Kernel generators (all types)
│   │   ├── pipeline.py       # Generation pipeline
│   │   └── batch_generator.py
│   └── examples/             # Example kernels
├── data/
│   ├── cpu/                  # CPU-generated samples
│   └── cuda/                 # CUDA-generated samples (for user testing)
├── tests/
│   ├── test_cuda.py          # CUDA test suite for user
│   └── run_cuda_tests.sh     # Script to run all CUDA tests
└── notes/
    └── cuda_analysis.md      # CPU vs CUDA differences
```

---

## State Management Protocol

### On Session Start
1. Read `CUDA_STATE.md` first
2. Read `tasks/cuda_tasks.md` for current task list
3. Resume from the documented next action

### On Session End (or ~20k tokens remaining)
1. Update `CUDA_STATE.md` with:
   - Current phase and task
   - Exact next action (be specific: file, function, iteration)
   - Any blockers or failed attempts
   - Key findings that affect next steps
2. Commit working code (no broken states)
3. Push to remote
4. Stop—do not start new tasks

### CUDA_STATE.md Template
```markdown
# CUDA Development State
- **Phase**: P1/P2/P3/P4
- **Current Task**: [task name]
- **Kernel Type**: [current kernel being adapted]
- **Status**: in_progress | blocked | ready_for_next

## Next Action
[Exactly what to do next. Be specific enough that a new agent can execute immediately.]

## Blockers (if any)
[What's preventing progress, what was tried]

## Completed Kernels
- [ ] arithmetic
- [ ] math
- [ ] loop
- [ ] conditional
- [ ] vector
- [ ] matrix
- [ ] combined
- [ ] reduction
- [ ] atomic
- [ ] memory

## Session Log
- [date/session]: [brief summary of what was accomplished]
```

---

## Phases

### P1: Base Code Selection
**Goal**: Select the best branch and set up working base code

**Tasks**:
1. Review `branch_progresses.md` for branch rankings
2. Pick the best production-ready branch (likely `12c4` from Tier 1)
3. Copy code from selected branch to working directory
4. Verify CPU code works: `python code/synthesis/pipeline.py --count 5`
5. Create `CUDA_STATE.md` and `tasks/cuda_tasks.md`

**Done when**: CPU pipeline runs successfully, state files created

### P2: CUDA Analysis
**Goal**: Understand CPU vs CUDA differences in Warp IR

**Tasks**:
1. Study `ir_extractor.py` for device parameter support
2. Analyze Warp source: how does `device="cuda"` change IR output?
3. Document findings in `notes/cuda_analysis.md`:
   - File extension differences (.cpp vs .cu)
   - Memory model differences
   - Kernel launch syntax
   - Type mapping differences
4. Create comparison table of CPU vs CUDA patterns

**Done when**: `notes/cuda_analysis.md` has concrete technical findings

### P3: Kernel Type Adaptation
**Goal**: Adapt each kernel type for CUDA, iteratively

Process each kernel type one at a time:

#### Kernel Types to Adapt (10 types)
1. **arithmetic** - basic math ops (+, -, *, /)
2. **math** - unary functions (sin, cos, exp, sqrt)
3. **loop** - for loops with wp.range
4. **conditional** - if/else branches
5. **vector** - wp.vec3 operations
6. **matrix** - wp.mat33 operations
7. **combined** - multi-pattern kernels
8. **reduction** - parallel reductions
9. **atomic** - atomic operations (wp.atomic_add, etc.)
10. **memory** - shared memory, memory access patterns

#### Per-Kernel Workflow (~1 iteration per kernel)
1. **Locate**: Find the kernel generator function
2. **Analyze**: Check if it's CUDA-compatible
3. **Adapt**: Add device parameter, handle CUDA-specific patterns
4. **Test Script**: Create test case for user to run on GPU
5. **Document**: Add to completed list in `CUDA_STATE.md`

**Done when**: All 10 kernel types have CUDA-compatible generators

### P4: Pipeline & Validation
**Goal**: End-to-end CUDA pipeline and test suite

**Tasks**:
1. Update `pipeline.py` to support `--device cuda` flag
2. Update `batch_generator.py` for CUDA batch generation
3. Create `tests/test_cuda.py` with comprehensive test suite
4. Create `tests/run_cuda_tests.sh` for user convenience
5. Generate sample CUDA test commands in `CUDA_STATE.md`

**Done when**: Complete test suite ready for user to run on GPU

### P5: CUDA Data Production
**Goal**: Generate large-scale CUDA IR training dataset (no GPU required)

**Key Insight**: CUDA IR generation is purely a code generation step - Warp compiles Python kernels to CUDA C++ code without needing a GPU. Only kernel *execution* requires a GPU.

**Tasks**:
1. Create dedicated CUDA production pipeline (`code/synthesis/cuda_pipeline.py`)
2. Generate 10,000+ CUDA Python→IR pairs
3. Validate generated data quality (check for `_cuda_kernel_forward` in output)
4. Create data statistics report
5. Compare CPU vs CUDA dataset characteristics

**Production Script**:
```bash
# Generate CUDA training data (works without GPU!)
python3 code/synthesis/pipeline.py --device cuda -n 10000 -o data/cuda_training

# Or use batch generator for better performance
python3 code/synthesis/batch_generator.py --device cuda -n 10000 -o data/cuda_large
```

**Validation Checks**:
```python
# Verify CUDA IR is properly generated
import json
from pathlib import Path

data_dir = Path("data/cuda_training")
for f in data_dir.glob("*.json"):
    data = json.load(open(f))
    assert "_cuda_kernel_forward" in data["cpp_forward"]
    assert data["metadata"]["device"] == "cuda"
```

**Done when**: 
- 10,000+ CUDA Python→IR pairs generated
- All pairs validated (contain proper CUDA IR)
- Data statistics documented

---

## Iteration Protocol

For each kernel type in P3:

### Step 1: Check Current State
```python
# Verify kernel generator exists and works on CPU
from code.synthesis.generator import GENERATORS
print(GENERATORS.keys())  # Should show kernel types
```

### Step 2: Add CUDA Support
```python
# Example pattern for CUDA adaptation
def generate_arithmetic_kernel(device="cpu"):
    @wp.kernel(device=device)
    def kernel(...):
        ...
    return kernel
```

### Step 3: Create Test Script
```python
# tests/test_{kernel_type}_cuda.py
import warp as wp
from code.synthesis.generator import generate_{kernel_type}_kernel

def test_on_cuda():
    wp.init()
    if not wp.is_cuda_available():
        print("CUDA not available, skipping")
        return
    
    kernel = generate_{kernel_type}_kernel(device="cuda")
    # ... test code
    print("PASS: {kernel_type} on CUDA")

if __name__ == "__main__":
    test_on_cuda()
```

### Step 4: Update State
```bash
git add -A
git commit -m "cuda: adapt {kernel_type} kernel for CUDA backend"
git push origin HEAD
```

---

## Test Commands for User

After completing development, provide these commands for user testing:

```bash
# 1. Check CUDA availability
python -c "import warp as wp; wp.init(); print('CUDA:', wp.is_cuda_available())"

# 2. Run single kernel test
python tests/test_arithmetic_cuda.py

# 3. Run full CUDA test suite
bash tests/run_cuda_tests.sh

# 4. Generate CUDA IR samples
python code/synthesis/pipeline.py --device cuda --count 10 --output data/cuda/

# 5. Compare CPU vs CUDA output
diff data/cpu/sample_0.json data/cuda/sample_0.json
```

---

## Validation Protocol

Before marking any kernel complete:
1. Code compiles without syntax errors
2. CPU mode still works (no regression)
3. Test script is self-contained and runnable
4. Clear output messages for pass/fail/skip

---

## Token Budget Guidelines

| Phase | Budget | Activities |
|-------|--------|------------|
| P1: Base Selection | ~20k | Review branches, set up base |
| P2: CUDA Analysis | ~30k | Study Warp, document differences |
| P3: Per Kernel | ~15k each | Adapt + test script (~150k total) |
| P4: Pipeline | ~30k | Integration, final test suite |
| P5: Data Production | ~50k | Generate 10k+ CUDA pairs, validate |

**Estimated total**: 280-330k tokens (4-6 sessions)

If blocked for >20k tokens on same issue:
1. Document the blocker in CUDA_STATE.md
2. Move to next kernel type
3. Mark blocker for later resolution

---

## Key Resources

- Warp documentation: https://nvidia.github.io/warp/
- CUDA device management: `wp.set_device("cuda:0")`
- Device-specific compilation: `@wp.kernel` with `device` parameter
- Key Warp source files:
  - `warp/codegen.py` (IR generation)
  - `warp/context.py` (device management)
  - `warp/build.py` (CUDA compilation)

---

## Anti-Patterns (Avoid)

- ❌ Trying to *execute* CUDA kernels without GPU (will fail)
- ❌ Breaking CPU functionality while adding CUDA
- ❌ Skipping test script creation
- ❌ Starting new kernel before completing current one
- ❌ Over-engineering device abstraction
- ❌ Leaving code in broken state at session end

**Note**: CUDA IR *generation* works without a GPU - only execution requires one.

---

## Success Criteria

Development is complete when:
1. All kernel types have CUDA-compatible generators
2. `pipeline.py` supports `--device cuda` flag
3. Comprehensive test suite in `tests/`
4. Clear user documentation for GPU testing
5. No regression in CPU functionality
6. `CUDA_STATE.md` shows all kernels completed
7. **P5**: 10,000+ CUDA Python→IR pairs generated and validated

# CUDA Backend Development

## Objective
Adapt the existing CPU-based JIT code synthesis pipeline to generate CUDA/GPU kernels. Since no GPU device is available in the agent environment, produce concise, testable code that can be validated manually on a GPU device.

---

## File Structure

```
cuda/
├── instruction_cuda.md      # This file (read-only reference)
├── CUDA_STATE.md            # Current progress tracker
├── code/                    # CUDA-adapted implementation
│   ├── extraction/          # GPU IR extraction utilities
│   ├── synthesis/           # GPU kernel generation
│   └── examples/            # Test kernels for GPU
├── data/                    # Sample GPU IR pairs (≤100 for git)
├── tests/                   # Test suite for manual GPU validation
└── notes/                   # GPU-specific findings
```

---

## State Management Protocol

### On Session Start
1. Read `CUDA_STATE.md` first
2. Resume from documented next action

### On Session End (or ~20k tokens remaining)
1. Update `CUDA_STATE.md` with:
   - Current milestone and task
   - Exact next action
   - Any blockers or failed attempts
   - Key findings that affect next steps
2. Commit working code (no broken states)
3. Stop—do not start new tasks

### CUDA_STATE.md Template
```markdown
# CUDA Development State
- **Milestone**: M1/M2/M3/M4
- **Task**: [task number and name]
- **Status**: in_progress | blocked | ready_for_next

## Next Action
[Exactly what to do next. Be specific enough that a new agent can execute immediately.]

## Blockers (if any)
[What's preventing progress, what was tried]

## Session Log
- [date/session]: [brief summary of what was accomplished]
```

---

## Milestones

### M1: Baseline - CPU Pipeline Analysis
**Goal**: Select and understand the best CPU-based branch as foundation

**Tasks**:
1. Analyze branch_progresses.md and identify best candidate (likely 12c4, 9177, or 8631)
2. Extract core files from selected branch to study:
   - `code/extraction/ir_extractor.py`
   - `code/synthesis/generator.py`
   - `code/synthesis/pipeline.py`
   - `code/synthesis/batch_generator.py`
3. Run CPU pipeline to generate 5-10 sample pairs (device="cpu")
4. Document current architecture in `notes/cpu_baseline.md` (max 50 lines)

**Deliverables**:
- Working CPU baseline in `code/` (copied from best branch)
- 5-10 CPU IR sample pairs in `data/cpu_samples/`
- `notes/cpu_baseline.md`: Architecture overview

**Done when**: CPU pipeline runs successfully and generates valid output

---

### M2: GPU IR Extraction
**Goal**: Adapt IR extraction to capture CUDA kernel code

**Tasks**:
1. Study warp's CUDA code generation:
   - Read `warp/codegen.py` for CUDA codegen paths
   - Understand difference between `.cpp` and `.cu` output
   - Identify how to access CUDA kernel source
2. Modify `ir_extractor.py` to support `device="cuda"`:
   - Add device parameter
   - Extract CUDA kernel code instead of CPU
   - Handle CUDA-specific metadata (thread blocks, shared memory, etc.)
3. Create test cases for each kernel type with GPU backend
4. Document findings in `notes/gpu_ir_format.md` (max 40 lines)

**Deliverables**:
- `code/extraction/ir_extractor.py`: GPU-aware version
- `code/extraction/test_gpu_extraction.py`: Test suite (runs on GPU)
- `notes/gpu_ir_format.md`: CUDA IR structure documentation

**Done when**: IR extractor can extract CUDA code (verified by code inspection, not execution)

**Kernel Types to Test**:
1. Arithmetic (basic ops)
2. Math functions (unary)
3. Loops (for/while)
4. Conditionals (if/else)
5. Vector operations (wp.vec3)
6. Matrix operations (wp.mat33)
7. Combined patterns

---

### M3: GPU Kernel Generation - Forward Pass
**Goal**: Adapt synthesis pipeline for CUDA kernel generation (forward pass only)

**Tasks**:
For each kernel type (7 types × forward pass):
1. Update `generator.py` to include GPU-specific patterns:
   - Thread indexing: `wp.tid()`, `wp.grid_dim()`
   - Shared memory: `wp.shared_*`
   - Atomic operations: `wp.atomic_add()`, etc.
   - Synchronization: `wp.syncthreads()`
2. Test generator creates valid CUDA-compatible Python code
3. Verify IR extraction produces valid CUDA output
4. Save 1-2 samples per kernel type

**Deliverables**:
- `code/synthesis/generator.py`: GPU kernel generation
- `code/synthesis/pipeline.py`: Updated for device="cuda"
- `data/gpu_samples/`: 10-15 forward pass samples
- `tests/test_forward_kernels.py`: Validation script

**Done when**: Generator creates 7 kernel types, each producing CUDA IR

**Iteration Plan** (~10 iterations):
- Iteration 1: Arithmetic kernels
- Iteration 2: Math function kernels
- Iteration 3: Loop kernels
- Iteration 4: Conditional kernels
- Iteration 5: Vector operation kernels
- Iteration 6: Matrix operation kernels
- Iteration 7: Combined pattern kernels
- Iteration 8-10: Fix issues, refine

---

### M4: GPU Kernel Generation - Backward Pass
**Goal**: Add gradient/backward pass for CUDA kernels

**Tasks**:
For each kernel type (7 types × backward pass):
1. Add gradient computation patterns to generators
2. Ensure backward kernels use CUDA features correctly
3. Test forward-backward pair generation
4. Validate gradient flow patterns in CUDA IR

**Deliverables**:
- `code/synthesis/generator.py`: Backward pass generation
- `data/gpu_samples/`: 10-15 backward pass samples
- `tests/test_backward_kernels.py`: Validation script

**Done when**: Generator creates forward+backward pairs for all 7 kernel types

**Iteration Plan** (~10 iterations):
- Iteration 1: Arithmetic backward
- Iteration 2: Math function backward
- Iteration 3: Loop backward
- Iteration 4: Conditional backward
- Iteration 5: Vector backward
- Iteration 6: Matrix backward
- Iteration 7: Combined backward
- Iteration 8-10: Fix issues, refine

---

### M5: Batch Generation & Validation Suite
**Goal**: Create production pipeline and comprehensive test suite

**Tasks**:
1. Update `batch_generator.py` for GPU:
   - Support device="cuda" parameter
   - Handle batch generation efficiently
   - Add error handling for CUDA-specific issues
2. Create validation suite:
   - `tests/validate_gpu_kernels.py`: Checks CUDA syntax
   - `tests/test_gpu_batch.py`: Batch generation tests
   - `tests/run_on_gpu.py`: Execution tests (for manual GPU testing)
3. Generate 50-100 diverse GPU IR samples
4. Create README with manual testing instructions

**Deliverables**:
- `code/synthesis/batch_generator.py`: GPU batch generation
- `tests/`: Complete validation suite
- `data/gpu_samples/`: 50-100 diverse samples
- `README.md`: Setup and testing instructions
- `notes/manual_test_guide.md`: Step-by-step GPU testing

**Done when**: 
- Batch generator produces valid CUDA Python→IR pairs
- Test suite ready for manual GPU validation
- README has clear testing instructions

---

## Validation Protocol

Since no GPU is available:
1. **Code Review**: Manually verify CUDA syntax correctness
2. **Static Analysis**: Check IR contains CUDA-specific patterns
3. **Comparison**: Compare CPU vs GPU IR structure
4. **Documentation**: Provide test commands for GPU execution

Before marking any milestone complete:
1. Code runs without Python errors
2. Generated IR contains expected CUDA patterns
3. Test scripts are ready (even if can't execute)
4. Manual testing instructions are clear

---

## Token Budget Guidelines

| Phase | Budget | Activities |
|-------|--------|------------|
| Orientation | ~5k | Read STATE, branch analysis |
| M1 | ~40k | Baseline setup, CPU testing |
| M2 | ~50k | GPU IR extraction (7 kernel types) |
| M3 | ~80k | Forward pass (10 iterations) |
| M4 | ~80k | Backward pass (10 iterations) |
| M5 | ~40k | Batch pipeline, validation suite |

Total estimate: ~300k tokens (multiple sessions expected)

If blocked for >20k tokens on same issue:
1. Document the blocker in CUDA_STATE.md
2. Move to next kernel type or milestone
3. Mark blocker for later resolution

---

## Key Resources

- Warp CUDA docs: https://nvidia.github.io/warp/
- Warp repo: https://github.com/NVIDIA/warp.git
- Key files to study:
  - `warp/codegen.py` (CUDA code generation)
  - `warp/context.py` (device management)
  - `warp/native/cuda/*.h` (CUDA runtime)
  - CPU branches: `origin/cursor/following-instructions-md-12c4`

---

## GPU-Specific Patterns to Include

### Thread Management
```python
@wp.kernel
def example(data: wp.array(dtype=float)):
    i = wp.tid()  # Thread index
    # ... operation on data[i]
```

### Shared Memory
```python
@wp.kernel
def example(input: wp.array(dtype=float), output: wp.array(dtype=float)):
    shared_data = wp.shared_array(shape=(256,), dtype=float)
    # ... use shared memory
```

### Atomic Operations
```python
@wp.kernel
def example(data: wp.array(dtype=float), result: wp.array(dtype=float)):
    wp.atomic_add(result, 0, data[wp.tid()])
```

### Grid/Block Configuration
```python
wp.launch(
    kernel=my_kernel,
    dim=1024,  # Number of threads
    inputs=[array],
    device="cuda"
)
```

---

## Anti-Patterns (Avoid These)

- ❌ Trying to execute CUDA code without GPU
- ❌ Generating large datasets (keep ≤100 samples)
- ❌ Over-commenting or writing lengthy docs
- ❌ Starting new kernel type with <20k tokens remaining
- ❌ Leaving code in broken state at session end
- ❌ Mixing CPU and GPU code in same samples

---

## Success Criteria

Project complete when:
1. ✅ CPU baseline established from best branch
2. ✅ IR extractor supports device="cuda"
3. ✅ Generator creates 7 kernel types with CUDA patterns
4. ✅ Forward pass works for all kernel types
5. ✅ Backward pass works for all kernel types
6. ✅ Batch generator produces GPU IR pairs
7. ✅ Test suite ready for manual GPU validation
8. ✅ 50-100 sample GPU IR pairs generated
9. ✅ README with clear testing instructions
10. ✅ All code is Python-correct (even if can't GPU-test)

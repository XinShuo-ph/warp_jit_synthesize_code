# CUDA Backend Development

## Objective
Adapt the current production CPU code to support CUDA backend for GPU acceleration. Since no GPU device is available in this environment, provide clean, testable code and validation commands for testing on GPU devices.

---

## File Structure

```
cuda/
├── instructions_cuda.md     # This file (read-only)
├── CUDA_STATE.md            # Current progress tracker
├── tasks/                   # Task lists for each milestone
│   ├── m1_tasks.md
│   ├── m2_tasks.md
│   └── ...
├── code/                    # CUDA-adapted implementation
│   ├── examples/            # Test examples for both CPU and CUDA
│   ├── extraction/          # IR extraction with device parameter
│   └── synthesis/           # Pipeline supporting both backends
├── tests/                   # Validation test suite
└── notes/                   # Implementation findings
```

---

## State Management Protocol

### On Session Start
1. Read `CUDA_STATE.md` first
2. Read the current milestone's task file (e.g., `tasks/m1_tasks.md`)
3. Resume from the documented next action

### On Session End (or ~20k tokens remaining)
1. Update `CUDA_STATE.md` with:
   - Current milestone and task
   - Exact next action (be specific: file, function, line if applicable)
   - Any blockers or failed attempts
   - Key findings that affect next steps
2. Commit working code (no broken states)
3. Stop—do not start new tasks

### CUDA_STATE.md Template
```markdown
# CUDA Development State
- **Milestone**: M1/M2/M3/M4/M5
- **Task**: [task number and name]
- **Status**: in_progress | blocked | ready_for_next

## Next Action
[Exactly what to do next. Be specific enough that a new agent can execute immediately.]

## Blockers (if any)
[What's preventing progress, what was tried]

## Key Findings
- [finding 1]
- [finding 2]

## Session Log
- [date/session]: [brief summary of what was accomplished]
```

---

## Milestones

### M1: Base Branch Selection & Analysis
**Goal**: Select best CPU branch as base and understand current implementation
**Deliverables**:
- Selected base branch from Tier 1 (12c4, 9177, or 8631)
- `notes/base_analysis.md`: Current architecture, key files (max 50 lines)
- Working copy of base code in `code/` directory
- CPU pipeline verified working

### M2: Device Parameter Infrastructure
**Goal**: Add device parameter support throughout codebase
**Deliverables**:
- Updated `code/extraction/ir_extractor.py` with `device` parameter
- Modified `code/synthesis/generator.py` to accept device parameter
- Updated `code/synthesis/pipeline.py` with device flag
- CPU tests still pass with `device="cpu"`
- `notes/device_param.md`: Implementation notes (max 30 lines)

### M3: Kernel Type Adaptation (Iterative)
**Goal**: Adapt each kernel type for CUDA backend
**Deliverables**:
- For each kernel type (arithmetic, math, loop, conditional, vector, matrix, combined):
  - Update generator to support CUDA patterns
  - Create test example in `code/examples/test_{kernel_type}_cuda.py`
  - Verify IR extraction works (on CPU, simulating CUDA patterns)
- `tasks/m3_tasks.md` with one task per kernel type
- Tests pass for each adapted kernel type

### M4: Forward & Backward Pass Support
**Goal**: Ensure CUDA support for both forward and backward operations
**Deliverables**:
- `code/examples/test_forward_cuda.py`: Forward pass examples
- `code/examples/test_backward_cuda.py`: Backward pass examples with gradients
- Updated pipeline to handle both pass types
- Documentation of CUDA-specific gradient computation patterns

### M5: Validation & Test Suite
**Goal**: Comprehensive test suite for GPU validation
**Deliverables**:
- `tests/test_cuda_extraction.py`: Test IR extraction with device="cuda"
- `tests/test_cuda_pipeline.py`: End-to-end pipeline test
- `tests/test_cuda_kernels.py`: Individual kernel type tests
- `tests/run_all_cuda_tests.sh`: Master test script for GPU execution
- `README_CUDA.md`: Setup instructions and testing guide for GPU users
- All tests include clear success criteria and expected outputs

---

## Task Breakdown Rules

When starting a milestone, create `tasks/mX_tasks.md` with:
```markdown
# Milestone X Tasks

## Task 1: [name]
- [ ] Step 1.1: [specific action]
- [ ] Step 1.2: [specific action]
- **Done when**: [concrete success criterion]

## Task 2: [name]
...
```

Rules:
- Each step should be completable in <5k tokens
- "Done when" must be testable (not subjective)
- Mark completed steps with [x]

---

## Validation Protocol

For all code (since GPU testing happens externally):
1. Verify code runs on CPU without errors
2. Include clear comments explaining CUDA-specific behavior
3. Provide expected output examples in comments or docs
4. Include validation commands that can be copy-pasted on GPU

Test file template:
```python
"""
Test: [description]

To run on GPU device:
    python test_file.py --device cuda

Expected output:
    [describe expected output]

Expected IR patterns:
    [describe what CUDA IR should contain]
"""
```

---

## Kernel Type Iteration Plan (M3)

Process in this order:

1. **Arithmetic kernels** (add, sub, mul, div)
   - Simplest, good starting point
   - Test basic CUDA thread operations

2. **Math kernels** (sin, cos, exp, log)
   - Test CUDA math library functions
   - Validate unary operations

3. **Loop kernels** (for loops, reductions)
   - Test CUDA thread indexing
   - Validate parallel loop patterns

4. **Conditional kernels** (if/else, switch)
   - Test CUDA branch divergence patterns
   - Validate predicate handling

5. **Vector kernels** (wp.vec3 operations)
   - Test CUDA vector types
   - Validate SIMD patterns

6. **Matrix kernels** (wp.mat33 operations)
   - Test CUDA matrix operations
   - Validate memory layout

7. **Combined kernels** (multi-pattern)
   - Test complex CUDA patterns
   - Validate composition of operations

Each iteration:
- Create generator function for CUDA variant
- Create test example
- Verify IR extraction (CPU simulation)
- Document CUDA-specific patterns
- Commit and push

---

## Token Budget Guidelines

| Phase | Budget | Activities |
|-------|--------|------------|
| Orientation | ~5k | Read CUDA_STATE.md, task file, understand context |
| Planning | ~10k | Break down next task, explore relevant code |
| Execution | ~150k | Implement, test, iterate |
| Handoff | ~10k | Update CUDA_STATE.md, clean up, verify state |

If blocked for >20k tokens on same issue:
1. Document the blocker in CUDA_STATE.md
2. Move to next kernel type or task
3. Mark blocker for later resolution

---

## Key Implementation Notes

### Device Parameter Pattern
```python
def extract_ir(kernel_fn, device="cpu"):
    """Extract IR for given device backend.
    
    Args:
        kernel_fn: Warp kernel function
        device: "cpu" or "cuda"
    
    Returns:
        str: Generated IR code (.cpp for cpu, .cu for cuda)
    """
    # Implementation
```

### CUDA-Specific Patterns
- Thread indexing: `wp.tid()` 
- Shared memory: `wp.shared` arrays
- Atomic operations: `wp.atomic_add()`, etc.
- Synchronization: `wp.synchronize()`

### IR Differences to Document
- CPU: C++ code with OpenMP pragmas
- CUDA: CUDA C code with `__global__`, `__device__` qualifiers
- Memory: CPU uses malloc, CUDA uses cudaMalloc/cudaMemcpy
- Launch: CPU is direct call, CUDA uses `<<<grid, block>>>`

---

## Anti-Patterns (Avoid These)

- ❌ Attempting to run CUDA code without GPU (will fail)
- ❌ Breaking CPU functionality while adding CUDA support
- ❌ Skipping CPU validation before marking complete
- ❌ Adding CUDA code without clear documentation
- ❌ Starting new kernel type before previous is complete
- ❌ Writing lengthy analysis documents
- ❌ Committing broken or incomplete code at session end

---

## Success Criteria

Project is complete when:
1. All kernel types support both CPU and CUDA backends
2. Device parameter flows through entire pipeline
3. CPU tests pass for both device="cpu" and CUDA patterns
4. Comprehensive test suite ready for GPU execution
5. Clear documentation for GPU users
6. Example commands for each test case
7. Clean code structure matching original CPU implementation

---

## Quick Reference Commands

```bash
# Setup base from best branch
git show origin/cursor/following-instructions-md-12c4:jit/code/extraction/ir_extractor.py > code/extraction/ir_extractor.py

# Test CPU baseline
python code/extraction/ir_extractor.py --device cpu

# Verify pipeline
python code/synthesis/pipeline.py --count 5 --device cpu

# Run validation tests (CPU mode)
python tests/test_cuda_extraction.py --simulate

# For GPU users (in documentation)
export CUDA_VISIBLE_DEVICES=0
python tests/run_all_cuda_tests.sh
```

---

## Branch Integration Note

This CUDA development is on branch `cursor/cuda-backend-development-db73`. The base CPU code comes from the merged JIT branches (primarily 12c4, 9177, 8631). After CUDA development is complete, this branch can be merged back to provide dual CPU/CUDA support.

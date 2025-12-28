# CUDA Backend Development

## Objective
Adapt the current CPU-based production code to generate CUDA backend IR (`.cu` code) alongside CPU IR (`.cpp` code). Produce a test suite and documentation for GPU device validation.

---

## File Structure

```
jit/
├── instruction_cuda.md      # This file (read-only)
├── CUDA_STATE.md            # Current progress tracker
├── code/
│   ├── extraction/          # IR extraction (CPU + CUDA)
│   │   ├── ir_extractor.py  # Modified to support device param
│   │   └── cuda_extractor.py # CUDA-specific extraction utilities
│   ├── synthesis/           # Kernel generation
│   │   ├── generator.py     # Kernel generators (unchanged)
│   │   ├── pipeline.py      # Modified for multi-device
│   │   └── cuda_pipeline.py # CUDA-specific pipeline
│   └── examples/            # Test kernels
├── data/
│   └── cuda_samples/        # CUDA IR pairs (for testing)
├── tests/
│   └── cuda/                # CUDA validation tests
│       ├── test_extraction.py
│       ├── test_kernels.py
│       └── run_gpu_tests.sh # Script for GPU device testing
└── notes/
    └── cuda_notes.md        # CUDA-specific findings
```

---

## State Management Protocol

### On Session Start
1. Read `CUDA_STATE.md` (create if missing)
2. Resume from documented next action

### On Session End (or ~20k tokens remaining)
1. Update `CUDA_STATE.md` with exact next action
2. Commit working changes
3. Stop—do not start new tasks

### CUDA_STATE.md Template
```markdown
# CUDA State
- **Phase**: P1/P2/P3/P4
- **Iteration**: [current iteration, e.g., "1/20 - arithmetic forward"]
- **Status**: in_progress | blocked | completed

## Next Action
[Specific next step]

## Blockers (if any)
[What's preventing progress, what was tried]

## Session Log
- [session]: [what was done]
```

---

## Phases

### P1: Setup Base Code
**Goal**: Reproduce working CPU codebase in this branch

**Tasks**:
1. Identify best merge branch from `cursor/agent-work-merge-process-*` branches
2. Copy production code to this branch:
   ```bash
   git checkout origin/cursor/agent-work-merge-process-XXXX -- jit/
   ```
3. Install dependencies: `pip install warp-lang`
4. Verify CPU pipeline works:
   ```bash
   python jit/code/synthesis/pipeline.py -n 5 -o /tmp/test_cpu
   ```
5. Document baseline in `CUDA_STATE.md`

**Done when**: CPU pipeline generates valid Python→C++ pairs

---

### P2: CUDA IR Extraction
**Goal**: Modify extraction to support `device="cuda"`

This phase has ~10 iterations, one per kernel type:
1. arithmetic
2. vector
3. matrix
4. control_flow
5. math
6. atomic
7. nested_loop
8. multi_condition
9. combined
10. scalar_param

**Per-Iteration Workflow** (for each kernel type):

#### Step 1: Forward Pass Extraction
1. Generate a single kernel of this type
2. Modify `ir_extractor.py` to accept `device` parameter
3. Extract forward kernel with `device="cuda"`
4. Save to `data/cuda_samples/{kernel_type}_forward.json`
5. Verify `.cu` code generated (not `.cpp`)

#### Step 2: Backward Pass Extraction
1. Enable backward pass: `include_backward=True`
2. Extract backward kernel with `device="cuda"`
3. Save to `data/cuda_samples/{kernel_type}_backward.json`
4. Verify adjoint kernel generated

#### Step 3: Validation
1. Compare CUDA vs CPU IR structure
2. Document differences in `notes/cuda_notes.md`

**Code Template**:
```python
# In ir_extractor.py, modify extract_ir():
def extract_ir(kernel, device: str = "cpu", include_backward: bool = True) -> dict:
    """
    Args:
        device: "cpu" for .cpp output, "cuda" for .cu output
    """
    # ... existing code ...
    cpp_code = builder.codegen(device)  # device param already supported!
    # ...
```

**Done when**: All 10 kernel types extract CUDA IR for forward + backward

---

### P3: CUDA Pipeline Integration
**Goal**: Update pipeline to generate CUDA pairs

**Tasks**:
1. Modify `pipeline.py` to support `--device` argument:
   ```python
   parser.add_argument("-d", "--device", default="cpu", choices=["cpu", "cuda"])
   ```
2. Update `synthesize_pair()` to pass device parameter
3. Update output file naming: `synth_cuda_{i:04d}.json` for CUDA
4. Add `ir_type` field to metadata: `"ir_type": "cuda"` or `"ir_type": "cpu"`
5. Test with small batch:
   ```bash
   python jit/code/synthesis/pipeline.py -n 10 -d cuda -o data/cuda_samples
   ```

**Done when**: Pipeline generates valid CUDA IR pairs

---

### P4: Test Suite for GPU Validation
**Goal**: Create tests that user can run on actual GPU device

**Tasks**:
1. Create `tests/cuda/test_extraction.py`:
   - Test that CUDA IR extraction produces `.cu` code markers
   - Test that CUDA-specific intrinsics appear in output
   - Test forward and backward for each kernel type

2. Create `tests/cuda/test_kernels.py`:
   - Test kernel compilation with `device="cuda"` (requires GPU)
   - Test kernel execution on GPU (requires GPU)
   - Mark as `@pytest.mark.skipif(not wp.is_cuda_available())`

3. Create `tests/cuda/run_gpu_tests.sh`:
   ```bash
   #!/bin/bash
   # Run on machine with GPU
   pip install warp-lang pytest
   python -c "import warp as wp; wp.init(); print('CUDA available:', wp.is_cuda_available())"
   pytest tests/cuda/ -v
   ```

4. Create `tests/cuda/README.md`:
   - How to run tests on GPU machine
   - Expected output
   - Troubleshooting

**Done when**: Test suite ready for GPU device validation

---

### P5: CUDA Production Pipeline
**Goal**: Create a dedicated CUDA batch generation pipeline for large-scale IR dataset production (no GPU required)

**Key Insight**: Warp's code generation system can produce CUDA IR code without an actual GPU device. This phase creates a production-ready pipeline to generate large CUDA IR datasets on any machine.

**Tasks**:

1. Create `code/synthesis/cuda_batch_generator.py`:
   - Dedicated CUDA batch generator with checkpointing
   - Support for all 10 kernel types
   - Configurable forward/backward pass inclusion
   - Progress tracking and resumption
   - Parallel generation for speed

2. Create `code/synthesis/cuda_dataset_stats.py`:
   - Analyze generated CUDA dataset
   - Report distribution by kernel type
   - Validate IR structure (forward/backward present)
   - Check for CUDA-specific markers

3. Generate production dataset:
   ```bash
   # Generate 1000 CUDA IR pairs with backward pass
   python3 jit/code/synthesis/cuda_batch_generator.py \
       --count 1000 \
       --output jit/data/cuda_production \
       --backward \
       --checkpoint
   ```

4. Validate production dataset:
   ```bash
   python3 jit/code/synthesis/cuda_dataset_stats.py jit/data/cuda_production
   ```

5. Create sample dataset for git (100 pairs max):
   - Keep small subset in `data/cuda_samples/`
   - Document how to generate full dataset

**Code Template** for `cuda_batch_generator.py`:
```python
#!/usr/bin/env python3
"""CUDA IR Batch Generator - Produces CUDA IR pairs without GPU."""
import argparse
from pathlib import Path
from generator import generate_kernel, GENERATORS
from pipeline import compile_kernel_from_source, extract_ir_from_kernel

def generate_cuda_batch(count, output_dir, include_backward=True, checkpoint=True):
    """Generate CUDA IR pairs in batches with checkpointing."""
    # Implementation here
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CUDA IR Batch Generator")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--checkpoint", action="store_true")
    # ...
```

**Validation Criteria**:
1. Generator produces valid CUDA IR without GPU
2. All 10 kernel types represented in output
3. Both forward and backward passes included (when --backward)
4. Checkpointing allows resumption after interruption
5. Dataset statistics tool validates output

**Done when**: 
- `cuda_batch_generator.py` can produce 1000+ CUDA IR pairs
- Dataset statistics tool validates all pairs
- Sample dataset (100 pairs) in git
- Documentation for large-scale generation

---

## Kernel Type Reference

| # | Kernel Type | Forward | Backward | Notes |
|---|-------------|---------|----------|-------|
| 1 | arithmetic | [ ] | [ ] | Basic +,-,*,/ |
| 2 | vector | [ ] | [ ] | wp.vec2/3/4 ops |
| 3 | matrix | [ ] | [ ] | wp.mat ops |
| 4 | control_flow | [ ] | [ ] | if/for loops |
| 5 | math | [ ] | [ ] | sin, cos, exp |
| 6 | atomic | [ ] | [ ] | atomic_add/min/max |
| 7 | nested_loop | [ ] | [ ] | Nested for loops |
| 8 | multi_condition | [ ] | [ ] | if/elif/else |
| 9 | combined | [ ] | [ ] | Mixed patterns |
| 10 | scalar_param | [ ] | [ ] | Scalar params |

---

## Key CUDA vs CPU Differences

| Aspect | CPU (.cpp) | CUDA (.cu) |
|--------|------------|------------|
| File extension | `.cpp` | `.cu` |
| Thread ID | Varies by platform | `blockIdx`, `threadIdx` |
| Memory | Standard malloc | `__shared__`, device memory |
| Functions | Standard C++ | `__device__`, `__global__` |
| Atomics | Platform-specific | CUDA atomics |

---

## Validation Protocol

Before marking any iteration complete:
1. Generated IR contains `.cu`-style markers (e.g., `__device__`, `__global__`)
2. Forward function extracted successfully
3. Backward function extracted (if applicable)
4. JSON pair saved to `data/cuda_samples/`
5. No errors or exceptions

---

## Token Budget

| Phase | Budget | Activities |
|-------|--------|------------|
| P1: Setup | ~20k | Copy base code, verify CPU works |
| P2: Per iteration | ~8k | Extract forward + backward, validate |
| P2: Total (~20 iters) | ~160k | All kernel types, both passes |
| P3: Pipeline | ~30k | Modify pipeline, test |
| P4: Tests | ~40k | Create test suite |
| P5: Production | ~50k | Batch generator, stats tool, dataset |

**Estimated total**: 300-350k tokens (4-6 sessions)

---

## Key Commands Reference

```bash
# Check available devices
python -c "import warp as wp; wp.init(); print('CUDA:', wp.is_cuda_available())"

# Generate CPU IR (baseline)
python jit/code/synthesis/pipeline.py -n 5 -d cpu -o /tmp/test_cpu

# Generate CUDA IR
python jit/code/synthesis/pipeline.py -n 5 -d cuda -o /tmp/test_cuda

# Run tests (no GPU required for extraction tests)
pytest tests/cuda/test_extraction.py -v

# Run GPU tests (requires GPU)
pytest tests/cuda/test_kernels.py -v

# Compare CPU vs CUDA output
diff /tmp/test_cpu/synth_0000.json /tmp/test_cuda/synth_0000.json
```

---

## Anti-Patterns (Avoid)

- ❌ Generating large datasets (keep ≤100 samples in git)
- ❌ Testing on GPU when no GPU available (extraction can be tested without GPU)
- ❌ Skipping backward pass extraction
- ❌ Starting new kernel type before completing current one
- ❌ Leaving broken code at session end
- ❌ Major refactoring of working CPU code

---

## Success Criteria

Phase 4 is complete when:
1. All 10 kernel types extract CUDA IR (forward + backward)
2. Pipeline supports `--device cuda` flag
3. Test suite in `tests/cuda/` ready for GPU validation
4. Sample CUDA pairs in `data/cuda_samples/`
5. `notes/cuda_notes.md` documents CPU vs CUDA differences
6. README with instructions for GPU testing

Phase 5 is complete when:
1. `cuda_batch_generator.py` produces CUDA IR without GPU
2. All 10 kernel types represented in generated dataset
3. Forward + backward passes included
4. Checkpointing supports resumption
5. `cuda_dataset_stats.py` validates dataset
6. Sample dataset (≤100 pairs) committed to git
7. Documentation for large-scale generation

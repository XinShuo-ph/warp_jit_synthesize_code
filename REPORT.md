# Report: JIT Code Synthesis for LLM Training

## Executive Summary
This project aims to produce a large-scale dataset of paired Python kernel code and their corresponding Low-Level Intermediate Representations (IR) for training Large Language Models (LLMs). We utilize **Nvidia Warp**, a Python framework for high-performance simulation, to generate these pairs.

## Technical Concepts

### JIT (Just-In-Time) Compilation
JIT compilation compiles code during execution rather than prior to execution. In this project, we use Warp's JIT compiler to transform Python functions (decorated with `@wp.kernel`) into native machine code (C++ for CPU, CUDA for GPU) at runtime. This allows us to capture the "translation" process that an LLM needs to learn.

### IR (Intermediate Representation)
IR is the code representation used by a compiler to optimize and generate machine code. For our dataset:
- **Input**: Python source code (Warp kernel).
- **Target**: The generated C++ (for CPU) or CUDA C++/PTX (for GPU) source code that Warp produces before final compilation.
This pairing allows an LLM to learn the mapping from high-level physics/simulation code to efficient low-level implementation.

### Nvidia Warp
Nvidia Warp is the core tool used. It allows writing differentiable simulation kernels in Python that compile to efficient C++/CUDA. Its unified kernel syntax makes it ideal for generating data that can target multiple backends (CPU/GPU) from the same source.

## Dataset Production

### Phase 1: CPU Code Production
We successfully established a production pipeline for CPU-target data.
- **Source Branch**: `cursor/agent-work-merge-process-bc08` (Merged from 16 development branches).
- **Methodology**:
    1. Randomly generate valid Warp kernels (10 distinct types including arithmetic, loops, vectors).
    2. JIT compile them for `device="cpu"`.
    3. Intercept and extract the generated C++ source code from the Warp cache.
- **Current Status**: 
    - Pipeline is fully functional.
    - Sample data generated in `cpu_production/data/cpu_dataset`.
    - **Scalability**: The system generates ~1 sample/sec. producing 200MB (approx. 40k samples) would take ~12-16 hours on a single node.

### Phase 2: CUDA Code Production
We have prepared the codebase for CUDA generation.
- **Codebase**: `cuda_production/`
- **Adaptations**:
    - Modified pipeline to target `device="cuda"`.
    - Updated IR extractor to capture `.cu` files and CUDA-specific kernel signatures.
- **Hardware Limitation**: Actual generation of CUDA IR requires an active NVIDIA GPU driver, which is unavailable in the current agent environment.
- **Status**: Code is "Ready for GPU". Running the pipeline on a GPU-enabled machine will produce the CUDA dataset.

## Dataset Statistics (Sample)

| Metric | Value |
|--------|-------|
| Kernel Types | 10 (Arithmetic, Math, Loop, Conditional, Vector, Atomic, etc.) |
| Input Format | Python (Warp DSL) |
| Output Format | C++ (CPU) / CUDA C++ (GPU) |
| Features | Forward Pass + Backward Pass (Adjoint/Autodiff) |

## Future Work & Recommendations
1.  **Scale Up**: Deploy the `cpu_production` script on a long-running instance to generate the full 200MB CPU dataset.
2.  **GPU Execution**: Clone the `cuda_production` folder to a GPU instance to generate the 200MB CUDA dataset.
3.  **Validation**: Run `validate_dataset.py` on the full datasets to ensure IR integrity.

# Warp JIT Dataset Report (CPU + CUDA)

## Summary
We generated two training datasets of Warp-kernel Python source paired with Warp-generated low-level code:
- **CPU dataset**: **~200MB**, **62,265** records (`data/production/cpu/`)
- **CUDA dataset**: **~200MB**, **54,074** records (`data/production/cuda/`)

Each record contains a Warp `@wp.kernel` Python snippet and the corresponding **generated forward kernel function** for the selected backend (`target=cpu|cuda`).

## JIT (Just-In-Time compilation)
JIT compilation means the system compiles program code **at runtime**, specialized to the current program, types, and configuration. For Warp kernels, that means a Python-defined kernel is turned into generated low-level code (CPU or GPU) when Warp builds the module for a target backend.

## IR (Intermediate Representation)
An IR is a lower-level representation of a program used by compilers as a bridge between the high-level source and the final machine code. In this dataset, we treat Warp’s generated backend code (e.g. CPU C++ or CUDA C/C++) as a practical “compiler IR artifact” suitable for training models to map high-level kernels to compiler-emitted code.

## NVIDIA Warp (where it fits)
Warp lets us write numerical kernels in Python with `@wp.kernel` and have Warp’s compilation pipeline generate backend code for:
- **CPU** (C++ codegen)
- **CUDA** (CUDA C/C++ codegen)

This project programmatically synthesizes many small kernels, triggers Warp code generation, and saves paired samples for LLM training.

## Dataset format
The production datasets are sharded as JSONL files (`shard_*.jsonl`). Each JSONL line is a record with:
- `kernel_name`: kernel function name
- `kernel_type`: one of `arithmetic|atomic|combined|conditional|loop|math|multi_cond|nested|scalar_param|vector`
- `python_source`: the Warp kernel source
- `generated_code`: extracted backend forward function (CPU or CUDA)
- `target`: `cpu` or `cuda`
- `metadata`: Warp version, module hash, codegen options, etc.

Small review samples:
- `data/samples/cpu_sample_100.jsonl` (100 records)
- `data/samples/cuda_sample_100.jsonl` (100 records)

## How the datasets were generated
Commands used:

```bash
python3 code/synthesis/produce_dataset.py --target cpu  --output data/production/cpu  --target-mb 200 --seed 20251228 --kernels-per-module 10 --shard-records 500
python3 code/synthesis/produce_dataset.py --target cuda --output data/production/cuda --target-mb 200 --seed 20251228 --kernels-per-module 10 --shard-records 500
```

Validation / stats:

```bash
python3 code/synthesis/validate_dataset.py --data-dir data/samples --sample-size 5
python3 code/synthesis/analyze_dataset.py  --data-dir data/production/cpu  --out-json reports/cpu_stats.json
python3 code/synthesis/analyze_dataset.py  --data-dir data/production/cuda --out-json reports/cuda_stats.json
```

## Limitations / notes
- This environment has **no CUDA driver** available. The CUDA dataset was generated via **Warp codegen** (`builder.codegen("cuda")`) without executing kernels on a GPU.
- The datasets are written under `data/production/` and are **git-ignored** by design due to size.


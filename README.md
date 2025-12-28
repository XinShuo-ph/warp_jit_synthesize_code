# Warp JIT Code Synthesis (Merged)

This repo contains a small, reproducible pipeline for generating **Python → Warp-generated C++ IR** training pairs from synthetic Warp kernels.

## Requirements

```bash
python3 -m pip install -r requirements.txt
```

Notes:
- This environment runs **CPU-only** (no CUDA driver), but Warp still generates C++ IR for CPU kernels.

## Generate sample pairs

Generates JSON files into `data/samples/`:

```bash
python3 code/synthesis/pipeline.py --count 30 --seed 42
```

Each JSON pair contains:
- `python_source`
- `cpp_ir_forward` (the extracted `*_cpu_kernel_forward` function)
- `cpp_ir_backward` (may be empty depending on kernel / Warp settings)
- `metadata` (includes `module_id`)

## Validate / analyze the dataset

```bash
python3 code/synthesis/validate_dataset.py
python3 code/synthesis/analyze_dataset.py
```

## IR extraction utilities

```bash
python3 code/extraction/ir_extractor.py
```

## Tests / fixtures

- Categorized deterministic kernels live under `code/extraction/cases/`.
- Determinism runner:

```bash
python3 code/extraction/test_cases.py
```

## Notes

Technical docs are in `notes/` (Warp compilation flow, IR format, dataset stats).


# GPU Test Plan (CUDA)

This repo is designed to run **CPU-first**. CUDA support is enabled via `--device cuda` and CUDA-marked tests.

## Requirements (on your GPU machine)

```bash
python3 -m pip install -U warp-lang numpy pytest
```

## Quick CUDA sanity checks

```bash
python3 -c "import warp as wp; wp.init(); print('cuda_available=', wp.is_cuda_available()); print('devices=', [d.alias for d in wp.get_devices()])"
```

## Run CUDA-marked tests

```bash
python3 -m pytest -q -m cuda
```

## Run synthesis pipeline on CUDA

```bash
python3 code/synthesis/pipeline.py --count 10 --output data/gpu_smoke --device cuda
python3 code/synthesis/validate_dataset.py data/gpu_smoke --sample-size 10 --device cuda
```

## Run batch generator on CUDA (optional)

```bash
python3 code/synthesis/batch_generator.py --count 20 --output data/gpu_batch_smoke --device cuda --kernels-per-module 10
python3 code/synthesis/validate_dataset.py data/gpu_batch_smoke --sample-size 20 --device cuda
```


# Data Generation Stats

## Full Datasets (not committed due to size)
- CPU: 209 MB (21,281 pairs) in data/cpu/
- CUDA: 231 MB (20,001 pairs) in data/cuda/

## Sample Data (committed)
- 100 sample pairs in data/samples/ (50 CPU + 50 CUDA)

## Regenerating Data
```bash
# CPU data
python3 code/synthesis/fast_batch_generator.py -n 25000 -o data/cpu --seed 42

# CUDA data
python3 code/synthesis/fast_batch_generator_cuda.py -n 20000 -o data/cuda --seed 42
```

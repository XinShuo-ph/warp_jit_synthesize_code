# Milestone 3 Tasks: CUDA Synthesis Pipeline

## Task 3.1: Adapt Pipeline for CUDA
- [ ] Copy pipeline.py to cuda_pipeline.py
- [ ] Replace ir_extractor import with cuda_ir_extractor
- [ ] Add device parameter with default "cuda"
- [ ] Update metadata to indicate CUDA
- [ ] Test with 5 sample kernels
- **Done when**: CUDA pipeline generates 5 valid pairs

## Task 3.2: Test Full Category Distribution
- [ ] Run pipeline with all categories
- [ ] Generate 50 samples (mixed categories)
- [ ] Verify distribution is balanced
- [ ] Check all CUDA-specific patterns present
- **Done when**: 50 samples with all 6 categories generated

## Task 3.3: Adapt Batch Generator
- [ ] Copy batch_generator.py to cuda_batch_generator.py
- [ ] Update to use cuda_pipeline
- [ ] Add device parameter
- [ ] Test batch generation
- **Done when**: Batch generator creates 100+ pairs

## Task 3.4: Generate Large CUDA Dataset
- [ ] Use batch generator to create 500 CUDA pairs
- [ ] Save to data/cuda_large/
- [ ] Generate statistics
- [ ] Verify quality (random sampling)
- **Done when**: 500 valid CUDA pairs generated

## Task 3.5: Compare CPU vs CUDA Output
- [ ] Generate same kernels with both CPU and CUDA
- [ ] Document structural differences
- [ ] Create comparison examples
- [ ] Save to notes/cpu_vs_cuda_samples.md
- **Done when**: Comparison documentation complete

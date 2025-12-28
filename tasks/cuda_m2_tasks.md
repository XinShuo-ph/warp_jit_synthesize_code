# Milestone 2 Tasks: CUDA Adaptation

## Task 1: Enable CUDA in Pipeline
- [ ] Step 1.1: Add `--device` argument to `code/synthesis/pipeline.py`.
- [ ] Step 1.2: Pass `device` to `synthesize_pair` and `extract_ir_from_kernel`.
- **Done when**: `python pipeline.py --device cuda` runs without error.

## Task 2: Enable CUDA in Batch Generator
- [ ] Step 2.1: Add `--device` argument to `code/synthesis/batch_generator.py`.
- [ ] Step 2.2: Ensure metadata includes correct device.
- **Done when**: Batch generation produces JSONs with `"device": "cuda"`.

## Task 3: Verify CUDA Generation
- [ ] Step 3.1: Generate a sample with `device="cuda"`.
- [ ] Step 3.2: Inspect the output JSON. Check `ir_code` for CUDA specific content.
- **Done when**: `ir_code` is confirmed to be different from CPU code and contains CUDA specifics.

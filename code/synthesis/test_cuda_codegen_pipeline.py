import warp as wp


def test_pipeline_cuda_codegen_without_gpu(tmp_path):
    """Pipeline should be able to produce CUDA IR without a GPU via offline codegen."""
    wp.init()

    from code.synthesis.pipeline import SynthesisPipeline

    out = tmp_path / "cuda_codegen"
    pipeline = SynthesisPipeline(str(out), seed=123, device="cuda")
    pair = pipeline.generate_pair("arithmetic")
    assert pair is not None

    # Should be CUDA kernel function, not CPU.
    assert "_cuda_kernel_forward" in pair.cpp_ir_forward
    assert pair.metadata.get("device") == "cuda"


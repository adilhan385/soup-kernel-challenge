"""Custom Triton kernel patches for Soup 0.77.0 on Kaggle Tesla T4 (Qwen2.5-1.5B-Instruct)."""


def apply():
    """Attach custom Triton kernels and memory optimizations to Soup and Qwen2 before training."""
    from solution.fused_ce import patch_soup_chunked_ce
    from solution.nf4_mlp import patch_nf4_and_qwen2_mlp
    from solution.rmsnorm import patch_qwen2_rmsnorm
    from solution.rope import patch_qwen2_rope

    patch_qwen2_rmsnorm()
    patch_qwen2_rope()
    patch_nf4_and_qwen2_mlp()
    patch_soup_chunked_ce(default_chunk=32)

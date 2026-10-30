"""Fused chunked online Cross-Entropy + Shannon Entropy + Token Accuracy in Triton."""

import torch
import triton
import triton.language as tl


@triton.jit
def qwen2_online_ce_entropy_fwd_kernel(
    logits_ptr,
    target_ptr,
    lse_ptr,
    entropy_ptr,
    loss_ptr,
    hits_ptr,
    stride_lm,
    vocab_size,
    ignore_index,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    row_ptr = logits_ptr + row * stride_lm

    m_i = -float("inf")
    d_i = 0.0
    w_i = 0.0
    best_val = -float("inf")
    best_idx = 0

    for col_start in range(0, vocab_size, BLOCK_V):
        cols = col_start + tl.arange(0, BLOCK_V)
        mask = cols < vocab_size
        z = tl.load(row_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)

        m_blk = tl.max(z, axis=0)
        arg_blk = tl.argmax(z, axis=0)
        is_better = m_blk > best_val
        best_val = tl.where(is_better, m_blk, best_val)
        best_idx = tl.where(is_better, col_start + arg_blk, best_idx)

        m_new = tl.maximum(m_i, m_blk)
        alpha = tl.exp(m_i - m_new)
        exp_z = tl.where(mask, tl.exp(z - m_new), 0.0)
        z_safe = tl.where(mask, z, 0.0)

        d_i = d_i * alpha + tl.sum(exp_z, axis=0)
        w_i = w_i * alpha + tl.sum(exp_z * z_safe, axis=0)
        m_i = m_new

    lse = m_i + tl.log(d_i)
    entropy = lse - (w_i / d_i)

    tl.store(lse_ptr + row, lse)
    tl.store(entropy_ptr + row, entropy)

    t = tl.load(target_ptr + row)
    valid = (t != ignore_index) & (t >= 0) & (t < vocab_size)
    safe_t = tl.where(valid, t, 0)
    z_t = tl.load(row_ptr + safe_t, mask=valid, other=0.0).to(tl.float32)

    token_loss = tl.where(valid, lse - z_t, 0.0)
    hit = tl.where(valid & (best_idx == safe_t), 1, 0).to(tl.int64)

    tl.store(loss_ptr + row, token_loss)
    tl.store(hits_ptr + row, hit)


@triton.jit
def qwen2_online_ce_bwd_kernel(
    logits_ptr,
    lse_ptr,
    target_ptr,
    inv_denom,
    stride_lm,
    vocab_size,
    ignore_index,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    v_blk = tl.program_id(1)

    cols = v_blk * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = cols < vocab_size
    row_ptr = logits_ptr + row * stride_lm

    t = tl.load(target_ptr + row)
    valid = (t != ignore_index) & (t >= 0) & (t < vocab_size)

    if not valid:
        z_zero = tl.zeros((BLOCK_V,), dtype=tl.float32)
        orig = tl.load(row_ptr + cols, mask=mask, other=0.0)
        tl.store(row_ptr + cols, z_zero.to(orig.dtype), mask=mask)
    else:
        lse = tl.load(lse_ptr + row)
        orig = tl.load(row_ptr + cols, mask=mask, other=0.0)
        z = orig.to(tl.float32)
        prob = tl.exp(z - lse)
        ind = (cols == t).to(tl.float32)
        grad = (prob - ind) * inv_denom
        tl.store(row_ptr + cols, grad.to(orig.dtype), mask=mask)


_LOGITS_SCRATCH = {}


def _get_logits_scratch(device: torch.device, dtype: torch.dtype, rows: int, vocab_size: int) -> torch.Tensor:
    key = (device, dtype, vocab_size)
    buf = _LOGITS_SCRATCH.get(key)
    if buf is None or buf.shape[0] < rows:
        buf = torch.empty((rows, vocab_size), dtype=dtype, device=device)
        _LOGITS_SCRATCH[key] = buf
    return buf[:rows]


def clear_logits_scratch():
    _LOGITS_SCRATCH.clear()


class TritonChunkedHeadLossFn(torch.autograd.Function):
    """Fused chunked LM head + Triton online cross-entropy, entropy, and accuracy."""

    @staticmethod
    def forward(ctx, hidden, head, target, chunk, denominator, ignore_index, with_grad):
        bsz, seq_len, hidden_dim = hidden.shape
        device = hidden.device
        grad = torch.zeros_like(hidden) if with_grad else None
        total = torch.zeros((), dtype=torch.float32, device=device)
        token_entropy = torch.empty((bsz, seq_len), dtype=hidden.dtype, device=device)
        token_loss = torch.empty((bsz, seq_len), dtype=torch.float32, device=device)
        correct = torch.zeros((), dtype=torch.long, device=device)

        if torch.is_tensor(denominator):
            denom_val = float(denominator.detach().float().clamp(min=1.0).item())
        else:
            denom_val = float(max(denominator, 1))
        inv_denom = 1.0 / denom_val

        # Extract linear weight/bias if available for direct Tensor Core GEMM into scratch buffer
        head_mod = getattr(head, "__self__", None)
        head_weight = getattr(head_mod, "weight", None) if head_mod is not None else None
        head_bias = getattr(head_mod, "bias", None) if head_mod is not None else None
        can_direct_gemm = (
            head_weight is not None
            and isinstance(head_weight, torch.Tensor)
            and not head_weight.requires_grad
            and hidden.is_cuda
        )

        if can_direct_gemm and head_weight.dtype != hidden.dtype:
            w_compute = head_weight.to(hidden.dtype)
        else:
            w_compute = head_weight

        block_v_fwd = 2048
        block_v_bwd = 1024

        for start in range(0, seq_len, chunk):
            stop = min(start + chunk, seq_len)
            cur_len = stop - start
            rows = bsz * cur_len

            piece = hidden[:, start:stop].detach()
            labels_slice = target[:, start:stop].contiguous().view(rows)

            if can_direct_gemm:
                piece_2d = piece.reshape(rows, hidden_dim)
                logits_2d = _get_logits_scratch(device, hidden.dtype, rows, w_compute.shape[0])
                torch.matmul(piece_2d, w_compute.t(), out=logits_2d)
                if head_bias is not None:
                    logits_2d.add_(head_bias.to(hidden.dtype))
                vocab_size = logits_2d.shape[1]
            else:
                logits_3d = head(piece)
                vocab_size = logits_3d.shape[-1]
                logits_2d = logits_3d.reshape(rows, vocab_size).contiguous()

            lse_1d = torch.empty((rows,), dtype=torch.float32, device=device)
            ent_1d = torch.empty((rows,), dtype=torch.float32, device=device)
            loss_1d = torch.empty((rows,), dtype=torch.float32, device=device)
            hits_1d = torch.empty((rows,), dtype=torch.long, device=device)

            qwen2_online_ce_entropy_fwd_kernel[(rows,)](
                logits_2d,
                labels_slice,
                lse_1d,
                ent_1d,
                loss_1d,
                hits_1d,
                logits_2d.stride(0),
                vocab_size,
                int(ignore_index),
                BLOCK_V=block_v_fwd,
                num_warps=8,
            )

            total = total + loss_1d.sum()
            token_entropy[:, start:stop] = ent_1d.to(hidden.dtype).view(bsz, cur_len)
            token_loss[:, start:stop] = loss_1d.view(bsz, cur_len)
            correct = correct + hits_1d.sum()

            if with_grad:
                grid_bwd = (rows, triton.cdiv(vocab_size, block_v_bwd))
                qwen2_online_ce_bwd_kernel[grid_bwd](
                    logits_2d,
                    lse_1d,
                    labels_slice,
                    inv_denom,
                    logits_2d.stride(0),
                    vocab_size,
                    int(ignore_index),
                    BLOCK_V=block_v_bwd,
                    num_warps=4,
                )
                if can_direct_gemm:
                    grad_2d = torch.matmul(logits_2d, w_compute)
                    grad[:, start:stop] = grad_2d.view(bsz, cur_len, hidden_dim)
                else:
                    piece_req = piece.detach().requires_grad_(True)
                    with torch.enable_grad():
                        out_req = head(piece_req).reshape(rows, vocab_size)
                        (piece_grad,) = torch.autograd.grad(
                            out_req, piece_req, grad_outputs=logits_2d
                        )
                    grad[:, start:stop] = piece_grad

        ctx.with_grad = with_grad
        if with_grad:
            ctx.save_for_backward(grad)
        ctx.mark_non_differentiable(token_entropy, token_loss, correct)
        return total / denominator, token_entropy, token_loss, correct

    @staticmethod
    def backward(ctx, grad_loss, _grad_entropy, _grad_token_loss, _grad_correct):
        if not ctx.with_grad:
            return (None,) * 7
        (grad,) = ctx.saved_tensors
        return (grad * grad_loss.to(grad.dtype), None, None, None, None, None, None)


def patch_soup_chunked_ce(default_chunk: int = 128):
    """Patch Soup's stream_loss_chunk to use TritonChunkedHeadLossFn and auto-attach on Qwen2 SFT."""
    import soup_cli.utils.stream_loss_chunk as slc
    from soup_cli.trainer import sft as sft_mod

    slc._chunked_function = lambda: TritonChunkedHeadLossFn

    orig_setup_transformers = sft_mod.SFTTrainerWrapper._setup_transformers

    def _patched_setup_transformers(self, cfg, tcfg):
        orig_setup_transformers(self, cfg, tcfg)
        model = getattr(self, "model", None)
        if model is None:
            return
        # Keep frozen non-quantized weights (embed_tokens / lm_head / norms / biases) in fp16 on pre-Ampere CUDA
        # so hidden_states stay fp16 instead of upcasting the entire forward/backward pass to fp32.
        try:
            if torch.cuda.is_available():
                major, _ = torch.cuda.get_device_capability()
                if major < 8:
                    for _name, param in model.named_parameters():
                        if not param.requires_grad and param.dtype == torch.float32:
                            param.data = param.data.to(torch.float16)
                    if hasattr(model, "enable_input_require_grads"):
                        model.enable_input_require_grads()
        except Exception:
            pass

        # Attach ChunkedHeadLoss automatically when output head is frozen and model is on CUDA
        if getattr(self, "_stream_loss_chunk_hook", None) is None:
            try:
                causal_lm = slc.causal_lm_of(model)
                head = causal_lm.get_output_embeddings() if hasattr(causal_lm, "get_output_embeddings") else None
                if (
                    head is not None
                    and isinstance(getattr(head, "weight", None), torch.Tensor)
                    and head.weight.is_cuda
                    and not any(p.requires_grad for p in head.parameters())
                ):
                    self._stream_loss_chunk_hook = slc.ChunkedHeadLoss(causal_lm, chunk=default_chunk)
            except Exception:
                pass

    if not getattr(sft_mod.SFTTrainerWrapper._setup_transformers, "_soup_solution_patched", False):
        _patched_setup_transformers._soup_solution_patched = True
        sft_mod.SFTTrainerWrapper._setup_transformers = _patched_setup_transformers

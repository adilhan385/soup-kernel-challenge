"""Fused chunked online Cross-Entropy + Shannon Entropy + Token Accuracy in Triton."""

import torch
import triton
import triton.language as tl

from solution.nf4_mlp import get_shared_scratch
from solution.rope import pop_unpad_valid_mask


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


class TritonChunkedHeadLossFn(torch.autograd.Function):
    """Fused chunked LM head + Triton online cross-entropy, entropy, and accuracy."""

    @staticmethod
    def forward(ctx, hidden, head, target, chunk, denominator, ignore_index, with_grad):
        bsz, seq_len, hidden_dim = hidden.shape
        device = hidden.device
        total = torch.zeros((), dtype=torch.float32, device=device)
        correct = torch.zeros((), dtype=torch.long, device=device)

        if torch.is_tensor(denominator):
            denom_val = float(denominator.detach().float().clamp(min=1.0).item())
        else:
            denom_val = float(max(denominator, 1))
        inv_denom = 1.0 / denom_val

        # Extract linear weight/bias if available for direct Tensor Core GEMM into shared scratch buffer
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
        block_v_bwd = 2048

        if can_direct_gemm:
            total_tokens = bsz * seq_len
            vocab_size = w_compute.shape[0]
            hidden_flat = hidden.detach().reshape(total_tokens, hidden_dim)
            target_flat = target.reshape(total_tokens)

            token_entropy_flat = torch.zeros((total_tokens,), dtype=hidden.dtype, device=device)
            token_loss_flat = torch.zeros((total_tokens,), dtype=torch.float32, device=device)
            grad_flat = (
                torch.zeros((total_tokens, hidden_dim), dtype=hidden.dtype, device=device)
                if with_grad
                else None
            )

            valid_mask = pop_unpad_valid_mask(hidden)
            active_mask = (
                (target_flat != ignore_index)
                & (target_flat >= 0)
                & (target_flat < vocab_size)
            )
            if valid_mask is not None and valid_mask.numel() == total_tokens:
                inactive_valid_mask = valid_mask & (~active_mask)
            else:
                inactive_valid_mask = ~active_mask

            active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
            inactive_idx = inactive_valid_mask.nonzero(as_tuple=False).squeeze(-1)
            n_active = active_idx.numel()
            all_valid_idx = torch.cat([active_idx, inactive_idx], dim=0)
            n_total_valid = all_valid_idx.numel()
            chunk_rows = min(max(bsz * chunk, 128), 512)

            for start in range(0, n_total_valid, chunk_rows):
                idx_slice = all_valid_idx[start : start + chunk_rows]
                rows = idx_slice.numel()
                piece_2d = hidden_flat.index_select(0, idx_slice)
                labels_slice = target_flat.index_select(0, idx_slice)

                logits_2d = get_shared_scratch(device, hidden.dtype, 0, (rows, vocab_size))
                torch.matmul(piece_2d, w_compute.t(), out=logits_2d)
                if head_bias is not None:
                    logits_2d.add_(head_bias.to(hidden.dtype))

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

                token_entropy_flat.index_copy_(0, idx_slice, ent_1d.to(hidden.dtype))
                bwd_rows = max(0, min(rows, n_active - start))
                if bwd_rows > 0:
                    active_slice = idx_slice[:bwd_rows]
                    total = total + loss_1d[:bwd_rows].sum()
                    token_loss_flat.index_copy_(0, active_slice, loss_1d[:bwd_rows])
                    correct = correct + hits_1d[:bwd_rows].sum()

                    if with_grad:
                        grid_bwd = (bwd_rows, triton.cdiv(vocab_size, block_v_bwd))
                        qwen2_online_ce_bwd_kernel[grid_bwd](
                            logits_2d[:bwd_rows],
                            lse_1d[:bwd_rows],
                            labels_slice[:bwd_rows],
                            inv_denom,
                            logits_2d.stride(0),
                            vocab_size,
                            int(ignore_index),
                            BLOCK_V=block_v_bwd,
                            num_warps=8,
                        )
                        grad_2d = torch.matmul(logits_2d[:bwd_rows], w_compute)
                        grad_flat.index_copy_(0, active_slice, grad_2d)

            token_entropy = token_entropy_flat.view(bsz, seq_len)
            token_loss = token_loss_flat.view(bsz, seq_len)
            grad = grad_flat.view(bsz, seq_len, hidden_dim) if with_grad else None
        else:
            token_entropy = torch.empty((bsz, seq_len), dtype=hidden.dtype, device=device)
            token_loss = torch.empty((bsz, seq_len), dtype=torch.float32, device=device)
            grad = torch.zeros_like(hidden) if with_grad else None

            for start in range(0, seq_len, chunk):
                stop = min(start + chunk, seq_len)
                cur_len = stop - start
                rows = bsz * cur_len

                piece = hidden[:, start:stop].detach()
                labels_slice = target[:, start:stop].contiguous().view(rows)

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
                        num_warps=8,
                    )
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
    """Patch Soup's stream_loss_chunk to use TritonChunkedHeadLossFn and avoid fp32 embedding spikes."""
    import soup_cli.utils.stream_loss_chunk as slc
    from soup_cli.trainer import sft as sft_mod

    slc._chunked_function = lambda: TritonChunkedHeadLossFn

    # Prevent peft.prepare_model_for_kbit_training from upcasting 445 MiB fp16 embed_tokens.weight
    # to 890 MiB float32 on pre-Ampere CUDA (Tesla T4), avoiding a 1.3 GiB transient memory spike.
    try:
        import peft
        import peft.utils.other as peft_other

        orig_prepare_kbit = peft_other.prepare_model_for_kbit_training

        def _fast_prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=None
        ):
            if torch.cuda.is_available():
                try:
                    major, _ = torch.cuda.get_device_capability()
                except Exception:
                    major = 8
                is_kbit = getattr(model, "is_loaded_in_4bit", False) or getattr(
                    model, "is_loaded_in_8bit", False
                )
                if major < 8 and is_kbit:
                    for _, param in model.named_parameters():
                        param.requires_grad = False
                        if (
                            param.dtype in (torch.float32, torch.bfloat16)
                            and param.__class__.__name__ != "Params4bit"
                        ):
                            param.data = param.data.to(torch.float16)
                    if hasattr(model, "enable_input_require_grads"):
                        model.enable_input_require_grads()
                    if use_gradient_checkpointing:
                        if gradient_checkpointing_kwargs is None:
                            gradient_checkpointing_kwargs = {"use_reentrant": True}
                        model.gradient_checkpointing_enable(
                            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                        )
                    return model
            return orig_prepare_kbit(
                model,
                use_gradient_checkpointing=use_gradient_checkpointing,
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
            )

        if not getattr(peft_other.prepare_model_for_kbit_training, "_soup_solution_patched", False):
            _fast_prepare_model_for_kbit_training._soup_solution_patched = True
            peft_other.prepare_model_for_kbit_training = _fast_prepare_model_for_kbit_training
            peft.prepare_model_for_kbit_training = _fast_prepare_model_for_kbit_training
    except Exception:
        pass

    orig_setup_transformers = sft_mod.SFTTrainerWrapper._setup_transformers

    def _patched_setup_transformers(self, cfg, tcfg):
        orig_setup_transformers(self, cfg, tcfg)
        model = getattr(self, "model", None)
        if model is None:
            return
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
                head = (
                    causal_lm.get_output_embeddings()
                    if hasattr(causal_lm, "get_output_embeddings")
                    else None
                )
                if (
                    head is not None
                    and isinstance(getattr(head, "weight", None), torch.Tensor)
                    and head.weight.is_cuda
                    and not any(p.requires_grad for p in head.parameters())
                ):
                    self._stream_loss_chunk_hook = slc.ChunkedHeadLoss(
                        causal_lm, chunk=default_chunk
                    )
            except Exception:
                pass

    if not getattr(sft_mod.SFTTrainerWrapper._setup_transformers, "_soup_solution_patched", False):
        _patched_setup_transformers._soup_solution_patched = True
        sft_mod.SFTTrainerWrapper._setup_transformers = _patched_setup_transformers

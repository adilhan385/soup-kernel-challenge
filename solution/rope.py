"""Fused QK Rotary Position Embedding (RoPE) in Triton + within-batch unpadding (per-sequence causal SDPA)."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def qwen2_rope_fwd_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    qo_ptr,
    ko_ptr,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_cb,
    stride_cs,
    stride_qob,
    stride_qoh,
    stride_qos,
    stride_kob,
    stride_koh,
    stride_kos,
    seq_len,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    half_d: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    b_idx = token_idx // seq_len
    s_idx = token_idx % seq_len

    offs = tl.arange(0, BLOCK_HALF)
    mask = offs < half_d

    c_base = cos_ptr + b_idx * stride_cb + s_idx * stride_cs
    s_base = sin_ptr + b_idx * stride_cb + s_idx * stride_cs

    c_lo = tl.load(c_base + offs, mask=mask, other=0.0).to(tl.float32)
    c_hi = tl.load(c_base + half_d + offs, mask=mask, other=0.0).to(tl.float32)
    s_lo = tl.load(s_base + offs, mask=mask, other=0.0).to(tl.float32)
    s_hi = tl.load(s_base + half_d + offs, mask=mask, other=0.0).to(tl.float32)

    if head_idx < n_qh:
        in_base = q_ptr + b_idx * stride_qb + head_idx * stride_qh + s_idx * stride_qs
        out_base = qo_ptr + b_idx * stride_qob + head_idx * stride_qoh + s_idx * stride_qos
    else:
        kh = head_idx - n_qh
        in_base = k_ptr + b_idx * stride_kb + kh * stride_kh + s_idx * stride_ks
        out_base = ko_ptr + b_idx * stride_kob + kh * stride_koh + s_idx * stride_kos

    x_lo_raw = tl.load(in_base + offs, mask=mask, other=0.0)
    x_hi_raw = tl.load(in_base + half_d + offs, mask=mask, other=0.0)
    x_lo = x_lo_raw.to(tl.float32)
    x_hi = x_hi_raw.to(tl.float32)

    y_lo = x_lo * c_lo - x_hi * s_lo
    y_hi = x_hi * c_hi + x_lo * s_hi

    tl.store(out_base + offs, y_lo.to(x_lo_raw.dtype), mask=mask)
    tl.store(out_base + half_d + offs, y_hi.to(x_hi_raw.dtype), mask=mask)


@triton.jit
def qwen2_rope_bwd_kernel(
    dqo_ptr,
    dko_ptr,
    cos_ptr,
    sin_ptr,
    dq_ptr,
    dk_ptr,
    stride_dqob,
    stride_dqoh,
    stride_dqos,
    stride_dkob,
    stride_dkoh,
    stride_dkos,
    stride_cb,
    stride_cs,
    stride_dqb,
    stride_dqh,
    stride_dqs,
    stride_dkb,
    stride_dkh,
    stride_dks,
    seq_len,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    half_d: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    b_idx = token_idx // seq_len
    s_idx = token_idx % seq_len

    offs = tl.arange(0, BLOCK_HALF)
    mask = offs < half_d

    c_base = cos_ptr + b_idx * stride_cb + s_idx * stride_cs
    s_base = sin_ptr + b_idx * stride_cb + s_idx * stride_cs

    c_lo = tl.load(c_base + offs, mask=mask, other=0.0).to(tl.float32)
    c_hi = tl.load(c_base + half_d + offs, mask=mask, other=0.0).to(tl.float32)
    s_lo = tl.load(s_base + offs, mask=mask, other=0.0).to(tl.float32)
    s_hi = tl.load(s_base + half_d + offs, mask=mask, other=0.0).to(tl.float32)

    if head_idx < n_qh:
        in_base = dqo_ptr + b_idx * stride_dqob + head_idx * stride_dqoh + s_idx * stride_dqos
        out_base = dq_ptr + b_idx * stride_dqb + head_idx * stride_dqh + s_idx * stride_dqs
    else:
        kh = head_idx - n_qh
        in_base = dko_ptr + b_idx * stride_dkob + kh * stride_dkoh + s_idx * stride_dkos
        out_base = dk_ptr + b_idx * stride_dkb + kh * stride_dkh + s_idx * stride_dks

    dy_lo_raw = tl.load(in_base + offs, mask=mask, other=0.0)
    dy_hi_raw = tl.load(in_base + half_d + offs, mask=mask, other=0.0)
    dy_lo = dy_lo_raw.to(tl.float32)
    dy_hi = dy_hi_raw.to(tl.float32)

    dx_lo = dy_lo * c_lo + dy_hi * s_hi
    dx_hi = dy_hi * c_hi - dy_lo * s_lo

    tl.store(out_base + offs, dx_lo.to(dy_lo_raw.dtype), mask=mask)
    tl.store(out_base + half_d + offs, dx_hi.to(dy_hi_raw.dtype), mask=mask)


def _normalize_cos_sin(cos: torch.Tensor, sin: torch.Tensor, bsz: int, seq_len: int, dim: int):
    if cos.dim() == 4:
        cos = cos.squeeze(1)
        sin = sin.squeeze(1)
    elif cos.dim() == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    if cos.stride(-1) != 1:
        cos = cos.contiguous()
    if sin.stride(-1) != 1:
        sin = sin.contiguous()
    stride_cb = 0 if cos.shape[0] == 1 else cos.stride(0)
    stride_cs = cos.stride(1)
    return cos, sin, stride_cb, stride_cs


class FusedQwen2RoPEFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        if q.stride(-1) != 1:
            q = q.contiguous()
        if k.stride(-1) != 1:
            k = k.contiguous()

        bsz, n_qh, seq_len, dim = q.shape
        _, n_kh, _, _ = k.shape
        half_d = dim // 2
        block_half = triton.next_power_of_2(half_d)

        cos_3d, sin_3d, stride_cb, stride_cs = _normalize_cos_sin(cos, sin, bsz, seq_len, dim)

        qo = torch.empty_like(q)
        ko = torch.empty_like(k)

        grid = (bsz * seq_len, n_qh + n_kh)
        qwen2_rope_fwd_kernel[grid](
            q,
            k,
            cos_3d,
            sin_3d,
            qo,
            ko,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            stride_cb,
            stride_cs,
            qo.stride(0),
            qo.stride(1),
            qo.stride(2),
            ko.stride(0),
            ko.stride(1),
            ko.stride(2),
            seq_len,
            n_qh=n_qh,
            n_kh=n_kh,
            half_d=half_d,
            BLOCK_HALF=block_half,
        )
        ctx.save_for_backward(cos_3d, sin_3d)
        ctx.stride_cb = stride_cb
        ctx.stride_cs = stride_cs
        ctx.shapes = (bsz, n_qh, n_kh, seq_len, half_d, block_half)
        return qo, ko

    @staticmethod
    def backward(ctx, dqo: torch.Tensor, dko: torch.Tensor):
        cos_3d, sin_3d = ctx.saved_tensors
        bsz, n_qh, n_kh, seq_len, half_d, block_half = ctx.shapes
        if dqo.stride(-1) != 1:
            dqo = dqo.contiguous()
        if dko.stride(-1) != 1:
            dko = dko.contiguous()

        dq = torch.empty_like(dqo)
        dk = torch.empty_like(dko)

        grid = (bsz * seq_len, n_qh + n_kh)
        qwen2_rope_bwd_kernel[grid](
            dqo,
            dko,
            cos_3d,
            sin_3d,
            dq,
            dk,
            dqo.stride(0),
            dqo.stride(1),
            dqo.stride(2),
            dko.stride(0),
            dko.stride(1),
            dko.stride(2),
            ctx.stride_cb,
            ctx.stride_cs,
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            seq_len,
            n_qh=n_qh,
            n_kh=n_kh,
            half_d=half_d,
            BLOCK_HALF=block_half,
        )
        return dq, dk, None, None


def fused_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids=None,
    unsqueeze_dim: int = 1,
):
    if q.is_cuda and k.is_cuda and q.dim() == 4 and k.dim() == 4 and unsqueeze_dim == 1:
        return FusedQwen2RoPEFn.apply(q, k, cos, sin)
    cos_u = cos.unsqueeze(unsqueeze_dim)
    sin_u = sin.unsqueeze(unsqueeze_dim)
    half_q = q.shape[-1] // 2
    half_k = k.shape[-1] // 2
    q_rot = torch.cat((-q[..., half_q:], q[..., :half_q]), dim=-1)
    k_rot = torch.cat((-k[..., half_k:], k[..., :half_k]), dim=-1)
    return (q * cos_u) + (q_rot * sin_u), (k * cos_u) + (k_rot * sin_u)


class _UnpadState:
    bounds: tuple[int, tuple[tuple[int, int], ...]] | None = None
    last_unpad_info: tuple[int, tuple[int, ...], torch.Tensor] | None = None


_UNPAD_STATE = _UnpadState()


def pop_unpad_valid_mask(hidden: torch.Tensor) -> torch.Tensor | None:
    """Return and clear the flat boolean valid-token mask if ``hidden`` matches the last unpadded output."""
    info = _UNPAD_STATE.last_unpad_info
    _UNPAD_STATE.last_unpad_info = None
    if info is not None and hidden.data_ptr() == info[0] and tuple(hidden.shape) == info[1]:
        return info[2]
    return None


class _ScatterUnpaddedFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        unpad_hidden: torch.Tensor,
        valid_idx: torch.Tensor,
        valid_mask_flat: torch.Tensor,
        bsz: int,
        seq_len: int,
    ):
        hidden_dim = unpad_hidden.shape[-1]
        out = torch.zeros(
            (bsz * seq_len, hidden_dim), dtype=unpad_hidden.dtype, device=unpad_hidden.device
        )
        out.index_copy_(0, valid_idx, unpad_hidden.squeeze(0))
        out_3d = out.view(bsz, seq_len, hidden_dim)
        _UNPAD_STATE.last_unpad_info = (out_3d.data_ptr(), tuple(out_3d.shape), valid_mask_flat)
        ctx.save_for_backward(valid_idx)
        return out_3d

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (valid_idx,) = ctx.saved_tensors
        hidden_dim = grad_out.shape[-1]
        grad_flat = grad_out.reshape(-1, hidden_dim)
        grad_unpad = grad_flat.index_select(0, valid_idx).unsqueeze(0)
        return grad_unpad, None, None, None, None


def fast_repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads while keeping physical memory in [B, S, H, D] order so sequence slices [:, :, s:e, :] are dense."""
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    return (
        hidden_states.transpose(1, 2)[:, :, :, None, :]
        .expand(batch, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(batch, slen, num_key_value_heads * n_rep, head_dim)
        .transpose(1, 2)
    )


def patch_qwen2_rope():
    from transformers.models.qwen2 import modeling_qwen2

    modeling_qwen2.apply_rotary_pos_emb = fused_apply_rotary_pos_emb
    modeling_qwen2.repeat_kv = fast_repeat_kv

    # 1. Per-sequence causal SDPA when within-batch unpadding is active
    if not getattr(F.scaled_dot_product_attention, "_soup_solution_patched", False):
        orig_sdpa = F.scaled_dot_product_attention

        def _unpadded_sdpa(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
            **kwargs,
        ):
            bounds_info = _UNPAD_STATE.bounds
            if (
                bounds_info is not None
                and query.is_cuda
                and query.shape[0] == 1
                and query.shape[2] == bounds_info[0]
            ):
                _, seq_bounds = bounds_info
                if len(seq_bounds) == 1:
                    return orig_sdpa(
                        query,
                        key,
                        value,
                        attn_mask=None,
                        dropout_p=dropout_p,
                        is_causal=True,
                        scale=scale,
                        **kwargs,
                    )
                outs = [
                    orig_sdpa(
                        query[:, :, s:e, :],
                        key[:, :, s:e, :],
                        value[:, :, s:e, :],
                        attn_mask=None,
                        dropout_p=dropout_p,
                        is_causal=True,
                        scale=scale,
                        **kwargs,
                    ).transpose(1, 2)
                    for s, e in seq_bounds
                ]
                return torch.cat(outs, dim=1).transpose(1, 2)
            return orig_sdpa(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                **kwargs,
            )

        _unpadded_sdpa._soup_solution_patched = True
        F.scaled_dot_product_attention = _unpadded_sdpa

    # 2. Within-batch unpadding in Qwen2Model.forward (removes pad tokens inside each batch)
    if not getattr(modeling_qwen2.Qwen2Model.forward, "_soup_solution_patched", False):
        orig_model_fwd = modeling_qwen2.Qwen2Model.forward

        def _fast_qwen2_model_forward(
            self,
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values=None,
            inputs_embeds: torch.FloatTensor | None = None,
            use_cache: bool | None = None,
            output_attentions: bool | None = None,
            output_hidden_states: bool | None = None,
            return_dict: bool | None = None,
            cache_position: torch.LongTensor | None = None,
            **kwargs,
        ):
            effective_use_cache = (
                use_cache if use_cache is not None else getattr(self.config, "use_cache", False)
            )
            if self.training:
                effective_use_cache = False
            if (
                not getattr(self, "gradient_checkpointing", False)
                and input_ids is not None
                and input_ids.is_cuda
                and input_ids.dim() == 2
                and inputs_embeds is None
                and attention_mask is not None
                and attention_mask.dim() == 2
                and attention_mask.shape == input_ids.shape
                and position_ids is None
                and past_key_values is None
                and not effective_use_cache
                and not output_attentions
                and not output_hidden_states
            ):
                bsz, seq_len = input_ids.shape
                if bsz > 1:
                    mask_bool = attention_mask.ne(0)
                    lengths = mask_bool.sum(dim=1).tolist()
                    total_valid = sum(lengths)
                    if 0 < total_valid < bsz * seq_len and all(
                        length > 0 for length in lengths
                    ):
                        valid_mask_flat = mask_bool.reshape(-1)
                        valid_idx = valid_mask_flat.nonzero(as_tuple=False).squeeze(-1)
                        pos_full = (mask_bool.cumsum(dim=1) - 1).clamp(min=0)
                        unpad_input_ids = (
                            input_ids.reshape(-1).index_select(0, valid_idx).unsqueeze(0)
                        )
                        unpad_pos_ids = (
                            pos_full.reshape(-1).index_select(0, valid_idx).unsqueeze(0)
                        )

                        bounds = []
                        offset = 0
                        for length in lengths:
                            bounds.append((offset, offset + length))
                            offset += length

                        prev_bounds = _UNPAD_STATE.bounds
                        _UNPAD_STATE.bounds = (total_valid, tuple(bounds))
                        try:
                            outputs = orig_model_fwd(
                                self,
                                input_ids=unpad_input_ids,
                                attention_mask=None,
                                position_ids=unpad_pos_ids,
                                past_key_values=None,
                                inputs_embeds=None,
                                use_cache=False,
                                output_attentions=output_attentions,
                                output_hidden_states=output_hidden_states,
                                return_dict=return_dict,
                                cache_position=None,
                                **kwargs,
                            )
                        finally:
                            _UNPAD_STATE.bounds = prev_bounds

                        if isinstance(outputs, tuple):
                            unpad_hidden = outputs[0]
                            padded_hidden = _ScatterUnpaddedFn.apply(
                                unpad_hidden, valid_idx, valid_mask_flat, bsz, seq_len
                            )
                            return (padded_hidden,) + outputs[1:]

                        unpad_hidden = outputs.last_hidden_state
                        padded_hidden = _ScatterUnpaddedFn.apply(
                            unpad_hidden, valid_idx, valid_mask_flat, bsz, seq_len
                        )
                        outputs.last_hidden_state = padded_hidden
                        return outputs

            return orig_model_fwd(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs,
            )

        _fast_qwen2_model_forward._soup_solution_patched = True
        modeling_qwen2.Qwen2Model.forward = _fast_qwen2_model_forward

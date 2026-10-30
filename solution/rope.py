"""Fused QK Rotary Position Embedding (RoPE) in Triton for Qwen2."""

import torch
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


def patch_qwen2_rope():
    from transformers.models.qwen2 import modeling_qwen2

    modeling_qwen2.apply_rotary_pos_emb = fused_apply_rotary_pos_emb

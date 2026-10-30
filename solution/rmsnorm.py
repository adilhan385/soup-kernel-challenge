"""Fused RMSNorm in Triton for Qwen2."""

import torch
import triton
import triton.language as tl


@triton.jit
def qwen2_rmsnorm_fwd_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    rstd_ptr,
    stride_x,
    stride_y,
    n_cols,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols

    x_row_ptr = x_ptr + row * stride_x + cols
    x_raw = tl.load(x_row_ptr, mask=mask, other=0.0)
    x = x_raw.to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)
    tl.store(rstd_ptr + row, rstd)

    y = (x * rstd) * w
    tl.store(y_ptr + row * stride_y + cols, y.to(x_raw.dtype), mask=mask)


@triton.jit
def qwen2_rmsnorm_bwd_kernel(
    dy_ptr,
    x_ptr,
    w_ptr,
    rstd_ptr,
    dx_ptr,
    stride_dy,
    stride_x,
    stride_dx,
    n_cols,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols

    dy_raw = tl.load(dy_ptr + row * stride_dy + cols, mask=mask, other=0.0)
    x_raw = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0)
    w_raw = tl.load(w_ptr + cols, mask=mask, other=0.0)

    dy = dy_raw.to(tl.float32)
    x = x_raw.to(tl.float32)
    w = w_raw.to(tl.float32)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)

    x_hat = x * rstd
    wdy = w * dy
    c1 = tl.sum(x_hat * wdy, axis=0) / n_cols
    dx = (wdy - x_hat * c1) * rstd

    tl.store(dx_ptr + row * stride_dx + cols, dx.to(x_raw.dtype), mask=mask)


class FusedRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float):
        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x_2d = x.contiguous().view(-1, n_cols)
        n_rows = x_2d.shape[0]

        w_contig = weight.contiguous()
        y_2d = torch.empty_like(x_2d)
        rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)

        block_n = triton.next_power_of_2(n_cols)
        num_warps = 4 if block_n <= 1024 else 8

        qwen2_rmsnorm_fwd_kernel[(n_rows,)](
            x_2d,
            w_contig,
            y_2d,
            rstd,
            x_2d.stride(0),
            y_2d.stride(0),
            n_cols,
            eps,
            BLOCK_N=block_n,
            num_warps=num_warps,
        )
        ctx.save_for_backward(x_2d, w_contig, rstd)
        ctx.block_n = block_n
        ctx.num_warps = num_warps
        ctx.orig_shape = orig_shape
        ctx.weight_requires_grad = weight.requires_grad
        return y_2d.view(orig_shape)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x_2d, w_contig, rstd = ctx.saved_tensors
        n_rows, n_cols = x_2d.shape
        dy_2d = dy.contiguous().view(n_rows, n_cols)
        dx_2d = torch.empty_like(x_2d)

        qwen2_rmsnorm_bwd_kernel[(n_rows,)](
            dy_2d,
            x_2d,
            w_contig,
            rstd,
            dx_2d,
            dy_2d.stride(0),
            x_2d.stride(0),
            dx_2d.stride(0),
            n_cols,
            BLOCK_N=ctx.block_n,
            num_warps=ctx.num_warps,
        )
        dw = None
        if ctx.weight_requires_grad:
            x_hat = x_2d.float() * rstd.unsqueeze(1)
            dw = (dy_2d.float() * x_hat).sum(dim=0).to(w_contig.dtype)
        return dx_2d.view(ctx.orig_shape), dw, None


def fused_rmsnorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    if hidden_states.is_cuda:
        return FusedRMSNormFn.apply(hidden_states, self.weight, self.variance_epsilon)
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)


def patch_qwen2_rmsnorm():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    Qwen2RMSNorm.forward = fused_rmsnorm_forward

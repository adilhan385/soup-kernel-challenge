"""Custom Triton NF4 blockwise dequantization + memory-efficient QLoRA Linear4bit & MLP."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def qwen2_nf4_dequant_kernel(
    packed_ptr,
    code_ptr,
    absmax_ptr,
    s2_code_ptr,
    s2_absmax_ptr,
    out_ptr,
    offset_val,
    n_elements,
    BLOCKSIZE: tl.constexpr,
    S2_BLOCKSIZE: tl.constexpr,
    IS_NESTED: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    mask = offs < n_elements

    byte_idx = offs >> 1
    is_lo = offs & 1

    packed_byte = tl.load(packed_ptr + byte_idx, mask=mask, other=0).to(tl.int32)
    hi_nib = (packed_byte >> 4) & 0x0F
    lo_nib = packed_byte & 0x0F
    nibble = tl.where(is_lo == 0, hi_nib, lo_nib)

    norm_val = tl.load(code_ptr + nibble, mask=mask, other=0.0).to(tl.float32)

    blk_idx = offs // BLOCKSIZE
    if IS_NESTED:
        q_abs = tl.load(absmax_ptr + blk_idx, mask=mask, other=0).to(tl.int32)
        s2_c = tl.load(s2_code_ptr + q_abs, mask=mask, other=0.0).to(tl.float32)
        s2_blk = blk_idx // S2_BLOCKSIZE
        s2_a = tl.load(s2_absmax_ptr + s2_blk, mask=mask, other=0.0).to(tl.float32)
        scale = s2_c * s2_a + offset_val
    else:
        scale = tl.load(absmax_ptr + blk_idx, mask=mask, other=0.0).to(tl.float32)

    deq = norm_val * scale
    tl.store(out_ptr + offs, deq.to(out_ptr.dtype.element_ty), mask=mask)


_WEIGHT_SCRATCH = {}


def _get_weight_scratch(device: torch.device, dtype: torch.dtype, tag: str, shape: tuple[int, int]) -> torch.Tensor:
    key = (device, dtype, tag, shape)
    buf = _WEIGHT_SCRATCH.get(key)
    if buf is None:
        buf = torch.empty(shape, dtype=dtype, device=device)
        _WEIGHT_SCRATCH[key] = buf
    return buf


def clear_weight_scratch():
    _WEIGHT_SCRATCH.clear()


def dequantize_nf4_triton(weight, dtype: torch.dtype = torch.float16, out: torch.Tensor | None = None) -> torch.Tensor:
    """Dequantize a bitsandbytes Params4bit weight tensor into ``dtype`` using ``qwen2_nf4_dequant_kernel``."""
    qs = weight.quant_state
    shape = tuple(qs.shape) if hasattr(qs, "shape") else tuple(weight.shape)
    n_elements = shape[0] * shape[1]
    if out is None:
        out = torch.empty(shape, dtype=dtype, device=weight.device)
    else:
        out = out.view(shape)

    packed = weight.data.view(-1)
    blocksize = int(qs.blocksize)
    is_nested = bool(getattr(qs, "nested", False))

    if is_nested:
        s2 = qs.state2
        s2_code = s2.code
        s2_absmax = s2.absmax
        s2_blocksize = int(s2.blocksize)
        offset_val = float(qs.offset.item()) if torch.is_tensor(qs.offset) else float(qs.offset or 0.0)
    else:
        s2_code = qs.code
        s2_absmax = qs.absmax
        s2_blocksize = 256
        offset_val = 0.0

    block_elems = 1024
    grid = (triton.cdiv(n_elements, block_elems),)
    qwen2_nf4_dequant_kernel[grid](
        packed,
        qs.code,
        qs.absmax,
        s2_code,
        s2_absmax,
        out,
        offset_val,
        n_elements,
        BLOCKSIZE=blocksize,
        S2_BLOCKSIZE=s2_blocksize,
        IS_NESTED=is_nested,
        BLOCK_ELEMS=block_elems,
        num_warps=4,
    )
    return out


class TritonLinear4bitFn(torch.autograd.Function):
    """Memory-free autograd function for frozen bitsandbytes Linear4bit layers.

    Unlike bitsandbytes' ``MatMul4Bit``, which calls ``ctx.save_for_backward(A, B)``
    even when ``B.requires_grad`` is False, this function saves zero input
    activations when the 4-bit weight is frozen.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight, bias: torch.Tensor | None):
        compute_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.float16
        x_comp = x if x.dtype == compute_dtype else x.to(compute_dtype)

        qs = weight.quant_state
        w_shape = tuple(qs.shape)
        w_buf = _get_weight_scratch(x.device, compute_dtype, "linear", w_shape)
        dequantize_nf4_triton(weight, dtype=compute_dtype, out=w_buf)

        out = torch.matmul(x_comp, w_buf.t())
        if bias is not None:
            out = out + (bias if bias.dtype == compute_dtype else bias.to(compute_dtype))

        ctx.weight = weight
        ctx.input_dtype = x.dtype
        ctx.compute_dtype = compute_dtype
        ctx.bias_requires_grad = bool(bias is not None and bias.requires_grad)
        return out if out.dtype == x.dtype else out.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        compute_dtype = ctx.compute_dtype
        dy = grad_output if grad_output.dtype == compute_dtype else grad_output.to(compute_dtype)

        weight = ctx.weight
        w_shape = tuple(weight.quant_state.shape)
        w_buf = _get_weight_scratch(dy.device, compute_dtype, "linear", w_shape)
        dequantize_nf4_triton(weight, dtype=compute_dtype, out=w_buf)

        grad_x = torch.matmul(dy, w_buf)
        if grad_x.dtype != ctx.input_dtype:
            grad_x = grad_x.to(ctx.input_dtype)

        grad_bias = None
        if ctx.bias_requires_grad:
            grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))
        return grad_x, None, grad_bias


class TritonCompactNF4MLPFn(torch.autograd.Function):
    """Memory-saving Qwen2MLP autograd function for frozen 4-bit gate/up/down projections.

    Saves only the compact MLP input ``x`` ([B, S, H], 6 MB/layer) instead of
    four intermediate [B, S, 8960] tensors (~147 MB/layer in fp16 or ~294 MB/layer in fp32),
    and fuses gate+up GEMMs in both forward and backward passes.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, w_gate, w_up, w_down):
        compute_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.float16
        x_comp = (x if x.dtype == compute_dtype else x.to(compute_dtype)).contiguous()

        orig_shape = x_comp.shape
        hidden_dim = orig_shape[-1]
        rows = x_comp.numel() // hidden_dim
        x_2d = x_comp.view(rows, hidden_dim)

        inter_dim = int(w_gate.quant_state.shape[0])
        w_gu = _get_weight_scratch(x.device, compute_dtype, "mlp_gu", (2 * inter_dim, hidden_dim))
        w_d = _get_weight_scratch(x.device, compute_dtype, "mlp_d", (hidden_dim, inter_dim))

        dequantize_nf4_triton(w_gate, dtype=compute_dtype, out=w_gu[:inter_dim])
        dequantize_nf4_triton(w_up, dtype=compute_dtype, out=w_gu[inter_dim:])
        dequantize_nf4_triton(w_down, dtype=compute_dtype, out=w_d)

        out_2d = torch.empty((rows, hidden_dim), dtype=compute_dtype, device=x.device)
        chunk_rows = 512

        for s in range(0, rows, chunk_rows):
            e = min(s + chunk_rows, rows)
            gu = torch.matmul(x_2d[s:e], w_gu.t())
            g = gu[:, :inter_dim]
            u = gu[:, inter_dim:]
            h = F.silu(g) * u
            torch.matmul(h, w_d.t(), out=out_2d[s:e])

        ctx.save_for_backward(x_comp)
        ctx.w_gate = w_gate
        ctx.w_up = w_up
        ctx.w_down = w_down
        ctx.input_dtype = x.dtype
        ctx.compute_dtype = compute_dtype
        out = out_2d.view(orig_shape)
        return out if out.dtype == x.dtype else out.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x_comp,) = ctx.saved_tensors
        compute_dtype = ctx.compute_dtype
        dy_comp = (grad_out if grad_out.dtype == compute_dtype else grad_out.to(compute_dtype)).contiguous()

        orig_shape = x_comp.shape
        hidden_dim = orig_shape[-1]
        rows = x_comp.numel() // hidden_dim
        x_2d = x_comp.view(rows, hidden_dim)
        dy_2d = dy_comp.view(rows, hidden_dim)

        inter_dim = int(ctx.w_gate.quant_state.shape[0])
        w_gu = _get_weight_scratch(x_comp.device, compute_dtype, "mlp_gu", (2 * inter_dim, hidden_dim))
        w_d = _get_weight_scratch(x_comp.device, compute_dtype, "mlp_d", (hidden_dim, inter_dim))

        dequantize_nf4_triton(ctx.w_gate, dtype=compute_dtype, out=w_gu[:inter_dim])
        dequantize_nf4_triton(ctx.w_up, dtype=compute_dtype, out=w_gu[inter_dim:])
        dequantize_nf4_triton(ctx.w_down, dtype=compute_dtype, out=w_d)

        dx_2d = torch.empty((rows, hidden_dim), dtype=compute_dtype, device=x_comp.device)
        chunk_rows = 512

        for s in range(0, rows, chunk_rows):
            e = min(s + chunk_rows, rows)
            gu = torch.matmul(x_2d[s:e], w_gu.t())
            dh = torch.matmul(dy_2d[s:e], w_d)

            g_f = gu[:, :inter_dim].float()
            u = gu[:, inter_dim:]
            sig = torch.sigmoid(g_f)
            silu_g = (g_f * sig).to(compute_dtype)
            grad_silu = (sig * (1.0 + g_f * (1.0 - sig))).to(compute_dtype)

            dg = dh * u * grad_silu
            du = dh * silu_g
            gu[:, :inter_dim] = dg
            gu[:, inter_dim:] = du
            torch.matmul(gu, w_gu, out=dx_2d[s:e])

        dx = dx_2d.view(orig_shape)
        if dx.dtype != ctx.input_dtype:
            dx = dx.to(ctx.input_dtype)
        return dx, None, None, None


def _is_frozen_nf4_linear(mod) -> bool:
    if type(mod).__name__ != "Linear4bit":
        return False
    w = getattr(mod, "weight", None)
    qs = getattr(w, "quant_state", None)
    if w is None or qs is None or w.requires_grad:
        return False
    bias = getattr(mod, "bias", None)
    if bias is not None and bias.requires_grad:
        return False
    return getattr(qs, "quant_type", "nf4") in ("nf4", "fp4")


def patch_nf4_and_qwen2_mlp():
    """Patch bitsandbytes Linear4bit and Qwen2MLP to use Triton NF4 dequantization and compact activations."""
    try:
        import bitsandbytes as bnb

        orig_linear4bit_fwd = bnb.nn.Linear4bit.forward

        def _fast_linear4bit_forward(self, x: torch.Tensor):
            if x.is_cuda and _is_frozen_nf4_linear(self):
                return TritonLinear4bitFn.apply(x, self.weight, self.bias)
            return orig_linear4bit_fwd(self, x)

        if not getattr(bnb.nn.Linear4bit.forward, "_soup_solution_patched", False):
            _fast_linear4bit_forward._soup_solution_patched = True
            bnb.nn.Linear4bit.forward = _fast_linear4bit_forward
    except ImportError:
        pass

    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    orig_mlp_fwd = Qwen2MLP.forward

    def _fast_qwen2_mlp_forward(self, x: torch.Tensor):
        if (
            x.is_cuda
            and _is_frozen_nf4_linear(self.gate_proj)
            and _is_frozen_nf4_linear(self.up_proj)
            and _is_frozen_nf4_linear(self.down_proj)
            and getattr(self.gate_proj, "bias", None) is None
            and getattr(self.up_proj, "bias", None) is None
            and getattr(self.down_proj, "bias", None) is None
        ):
            return TritonCompactNF4MLPFn.apply(
                x, self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight
            )
        return orig_mlp_fwd(self, x)

    if not getattr(Qwen2MLP.forward, "_soup_solution_patched", False):
        _fast_qwen2_mlp_forward._soup_solution_patched = True
        Qwen2MLP.forward = _fast_qwen2_mlp_forward

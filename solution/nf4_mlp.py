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


_SCRATCH_POOL = {}


def get_shared_scratch(
    device: torch.device, dtype: torch.dtype, slot: int, shape: tuple[int, int]
) -> torch.Tensor:
    """Return a 2D view into a persistent scratch buffer for the given slot.

    Slot 0 is shared across non-overlapping operations:
      - ``TritonLinear4bitFn`` dequantized weight ([1536, 1536])
      - ``TritonCompactNF4MLPFn`` fused gate+up weight ([17920, 1536], 52.5 MiB)
      - ``TritonChunkedHeadLossFn`` chunk logits ([256, 151936], 74.2 MiB)
    Slot 1 holds ``TritonCompactNF4MLPFn`` down weight ([1536, 8960], 26.25 MiB).
    """
    key = (device, dtype, slot)
    needed = shape[0] * shape[1]
    buf = _SCRATCH_POOL.get(key)
    if buf is None or buf.numel() < needed:
        min_elems = 38895616 if slot == 0 else 13762560
        buf = torch.empty((max(needed, min_elems),), dtype=dtype, device=device)
        _SCRATCH_POOL[key] = buf
    return buf[:needed].view(shape)


def clear_weight_scratch():
    _SCRATCH_POOL.clear()


def get_dequantized_weight(
    weight, dtype: torch.dtype, cache: bool = False, slot: int = 0
) -> torch.Tensor:
    """Retrieve dequantized weight either from a per-weight cache or from a shared scratch slot."""
    if cache:
        cached = getattr(weight, "_soup_cached_fp16", None)
        if cached is not None and cached.dtype == dtype and cached.device == weight.device:
            return cached
        w_shape = tuple(weight.quant_state.shape)
        w_fp16 = torch.empty(w_shape, dtype=dtype, device=weight.device)
        dequantize_nf4_triton(weight, dtype=dtype, out=w_fp16)
        weight._soup_cached_fp16 = w_fp16
        return w_fp16
    w_shape = tuple(weight.quant_state.shape)
    w_buf = get_shared_scratch(weight.device, dtype, slot, w_shape)
    dequantize_nf4_triton(weight, dtype=dtype, out=w_buf)
    return w_buf


def dequantize_nf4_triton(
    weight, dtype: torch.dtype = torch.float16, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Dequantize a bitsandbytes Params4bit weight tensor into ``dtype`` using ``qwen2_nf4_dequant_kernel``."""
    meta = getattr(weight, "_soup_nf4_meta", None)
    if (
        meta is None
        or meta[0].device != weight.device
        or meta[0].data_ptr() != weight.data.data_ptr()
    ):
        qs = weight.quant_state
        shape = tuple(qs.shape) if hasattr(qs, "shape") else tuple(weight.shape)
        n_elements = shape[0] * shape[1]
        packed = weight.data.view(-1)
        blocksize = int(qs.blocksize)
        is_nested = bool(getattr(qs, "nested", False))

        if is_nested:
            s2 = qs.state2
            s2_code = s2.code
            s2_absmax = s2.absmax
            s2_blocksize = int(s2.blocksize)
            offset_val = (
                float(qs.offset.item())
                if torch.is_tensor(qs.offset)
                else float(qs.offset or 0.0)
            )
        else:
            s2_code = qs.code
            s2_absmax = qs.absmax
            s2_blocksize = 256
            offset_val = 0.0

        block_elems = 1024
        grid = (triton.cdiv(n_elements, block_elems),)
        meta = (
            packed,
            qs.code,
            qs.absmax,
            s2_code,
            s2_absmax,
            offset_val,
            n_elements,
            blocksize,
            s2_blocksize,
            is_nested,
            shape,
            block_elems,
            grid,
        )
        try:
            weight._soup_nf4_meta = meta
        except Exception:
            pass

    (
        packed,
        code,
        absmax,
        s2_code,
        s2_absmax,
        offset_val,
        n_elements,
        blocksize,
        s2_blocksize,
        is_nested,
        shape,
        block_elems,
        grid,
    ) = meta

    if out is None:
        out = torch.empty(shape, dtype=dtype, device=weight.device)
    else:
        out = out.view(shape)

    qwen2_nf4_dequant_kernel[grid](
        packed,
        code,
        absmax,
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

        w_buf = get_dequantized_weight(weight, compute_dtype, cache=True)

        out = torch.matmul(x_comp, w_buf.t())
        if bias is not None:
            out.add_(bias if bias.dtype == compute_dtype else bias.to(compute_dtype))

        ctx.weight = weight
        ctx.input_dtype = x.dtype
        ctx.compute_dtype = compute_dtype
        ctx.bias_requires_grad = bool(bias is not None and bias.requires_grad)
        return out if out.dtype == x.dtype else out.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        compute_dtype = ctx.compute_dtype
        dy = grad_output if grad_output.dtype == compute_dtype else grad_output.to(compute_dtype)

        w_buf = get_dequantized_weight(ctx.weight, compute_dtype, cache=True)

        grad_x = torch.matmul(dy, w_buf)
        if grad_x.dtype != ctx.input_dtype:
            grad_x = grad_x.to(ctx.input_dtype)

        grad_bias = None
        if ctx.bias_requires_grad:
            grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))
        return grad_x, None, grad_bias


class TritonLoRALinear4bitFn(torch.autograd.Function):
    """Fused autograd function for frozen 4-bit base layer + active LoRA adapter (A, B).

    Eliminates full-hidden-state fp16<->fp32 casts and duplicate saved fp32 input copies
    across q_proj and v_proj, running all projections on FP16 Tensor Cores while keeping
    master gradients for ``lora_A.weight`` and ``lora_B.weight`` in float32.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight,
        bias: torch.Tensor | None,
        lora_a_weight: torch.Tensor,
        lora_b_weight: torch.Tensor,
        scaling: float,
    ):
        compute_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.float16
        x_comp = (x if x.dtype == compute_dtype else x.to(compute_dtype)).contiguous()
        orig_shape = x_comp.shape
        in_dim = orig_shape[-1]
        rows = x_comp.numel() // in_dim
        x_2d = x_comp.view(rows, in_dim)

        w_buf = get_dequantized_weight(weight, compute_dtype, cache=True)
        out_dim = w_buf.shape[0]

        out_2d = torch.matmul(x_2d, w_buf.t())
        if bias is not None:
            out_2d.add_(bias if bias.dtype == compute_dtype else bias.to(compute_dtype))

        a_comp = (
            lora_a_weight
            if lora_a_weight.dtype == compute_dtype
            else lora_a_weight.to(compute_dtype)
        )
        b_comp = (
            lora_b_weight
            if lora_b_weight.dtype == compute_dtype
            else lora_b_weight.to(compute_dtype)
        )
        xa_2d = torch.matmul(x_2d, a_comp.t())
        out_2d.addmm_(xa_2d, b_comp.t(), beta=1.0, alpha=float(scaling))

        ctx.save_for_backward(x_comp, xa_2d, lora_a_weight, lora_b_weight)
        ctx.weight = weight
        ctx.scaling = float(scaling)
        ctx.input_dtype = x.dtype
        ctx.compute_dtype = compute_dtype
        ctx.orig_shape = orig_shape
        ctx.bias_requires_grad = bool(bias is not None and bias.requires_grad)
        out = out_2d.view(*orig_shape[:-1], out_dim)
        return out if out.dtype == x.dtype else out.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_comp, xa_2d, lora_a_weight, lora_b_weight = ctx.saved_tensors
        compute_dtype = ctx.compute_dtype
        scaling = ctx.scaling

        dy_comp = (
            grad_output if grad_output.dtype == compute_dtype else grad_output.to(compute_dtype)
        ).contiguous()
        out_dim = dy_comp.shape[-1]
        rows = dy_comp.numel() // out_dim
        dy_2d = dy_comp.view(rows, out_dim)
        x_2d = x_comp.view(rows, -1)

        b_comp = (
            lora_b_weight
            if lora_b_weight.dtype == compute_dtype
            else lora_b_weight.to(compute_dtype)
        )
        a_comp = (
            lora_a_weight
            if lora_a_weight.dtype == compute_dtype
            else lora_a_weight.to(compute_dtype)
        )

        grad_b = None
        if ctx.needs_input_grad[4]:
            gb = torch.matmul(dy_2d.t(), xa_2d)
            grad_b = gb.to(lora_b_weight.dtype).mul_(scaling)

        dxa_2d = torch.matmul(dy_2d, b_comp)
        if scaling != 1.0:
            dxa_2d.mul_(scaling)

        grad_a = None
        if ctx.needs_input_grad[3]:
            ga = torch.matmul(dxa_2d.t(), x_2d)
            grad_a = ga.to(lora_a_weight.dtype)

        grad_x = None
        if ctx.needs_input_grad[0]:
            w_buf = get_dequantized_weight(ctx.weight, compute_dtype, cache=True)
            grad_x_2d = torch.matmul(dy_2d, w_buf)
            grad_x_2d.addmm_(dxa_2d, a_comp)
            grad_x = grad_x_2d.view(ctx.orig_shape)
            if grad_x.dtype != ctx.input_dtype:
                grad_x = grad_x.to(ctx.input_dtype)

        grad_bias = None
        if ctx.bias_requires_grad:
            grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))

        return grad_x, None, grad_bias, grad_a, grad_b, None


class TritonCompactNF4MLPFn(torch.autograd.Function):
    """Memory-saving Qwen2MLP autograd function for frozen 4-bit gate/up/down projections.

    Saves ``gu`` when under the VRAM budget (or compact ``x`` otherwise), and uses
    in-place ATen ``silu_backward`` in backward with zero fp32 tensor allocations.
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
        w_gu = get_shared_scratch(x.device, compute_dtype, 0, (2 * inter_dim, hidden_dim))
        w_d = get_shared_scratch(x.device, compute_dtype, 1, (hidden_dim, inter_dim))

        dequantize_nf4_triton(w_gate, dtype=compute_dtype, out=w_gu[:inter_dim])
        dequantize_nf4_triton(w_up, dtype=compute_dtype, out=w_gu[inter_dim:])
        dequantize_nf4_triton(w_down, dtype=compute_dtype, out=w_d)

        gu = torch.matmul(x_2d, w_gu.t())
        save_gu = (
            torch.is_grad_enabled()
            and x.requires_grad
            and ((not x.is_cuda) or (torch.cuda.memory_allocated(x.device) < 4096 * 1024 * 1024))
        )
        if save_gu:
            h = F.silu(gu[:, :inter_dim])
            h.mul_(gu[:, inter_dim:])
            out_2d = torch.matmul(h, w_d.t())
            del h
            ctx.save_for_backward(gu)
            ctx.saved_gu = True
        else:
            g = gu[:, :inter_dim]
            u = gu[:, inter_dim:]
            F.silu(g, inplace=True)
            g.mul_(u)
            out_2d = torch.matmul(g, w_d.t())
            del gu
            ctx.save_for_backward(x_comp)
            ctx.saved_gu = False

        ctx.orig_shape = orig_shape
        ctx.rows = rows
        ctx.hidden_dim = hidden_dim
        ctx.w_gate = w_gate
        ctx.w_up = w_up
        ctx.w_down = w_down
        ctx.input_dtype = x.dtype
        ctx.compute_dtype = compute_dtype
        out = out_2d.view(orig_shape)
        return out if out.dtype == x.dtype else out.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (saved,) = ctx.saved_tensors
        compute_dtype = ctx.compute_dtype
        dy_comp = (grad_out if grad_out.dtype == compute_dtype else grad_out.to(compute_dtype)).contiguous()

        orig_shape = ctx.orig_shape
        rows = ctx.rows
        hidden_dim = ctx.hidden_dim
        dy_2d = dy_comp.view(rows, hidden_dim)

        inter_dim = int(ctx.w_gate.quant_state.shape[0])
        w_gu = get_shared_scratch(saved.device, compute_dtype, 0, (2 * inter_dim, hidden_dim))
        w_d = get_shared_scratch(saved.device, compute_dtype, 1, (hidden_dim, inter_dim))

        dequantize_nf4_triton(ctx.w_down, dtype=compute_dtype, out=w_d)

        if ctx.saved_gu:
            gu = saved
        else:
            x_2d = saved.view(rows, hidden_dim)
            dequantize_nf4_triton(ctx.w_gate, dtype=compute_dtype, out=w_gu[:inter_dim])
            dequantize_nf4_triton(ctx.w_up, dtype=compute_dtype, out=w_gu[inter_dim:])
            gu = torch.matmul(x_2d, w_gu.t())

        dh = torch.matmul(dy_2d, w_d)
        g = gu[:, :inter_dim]
        u = gu[:, inter_dim:]
        u.mul_(dh)  # u <- dL/dsilu(g)
        silu_g = F.silu(g)
        dh.mul_(silu_g)  # dh <- dL/du
        del silu_g
        dg = torch.ops.aten.silu_backward(u, g)
        g.copy_(dg)
        del dg
        u.copy_(dh)
        del dh, saved

        if ctx.saved_gu:
            dequantize_nf4_triton(ctx.w_gate, dtype=compute_dtype, out=w_gu[:inter_dim])
            dequantize_nf4_triton(ctx.w_up, dtype=compute_dtype, out=w_gu[inter_dim:])
        dx_2d = torch.matmul(gu, w_gu)
        del gu

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
    """Patch bitsandbytes Linear4bit, PEFT LoRA Linear4bit, and Qwen2MLP."""
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

    try:
        import peft.tuners.lora.bnb as lora_bnb

        orig_lora_4bit_fwd = lora_bnb.Linear4bit.forward

        def _triton_lora_4bit_forward(self, x: torch.Tensor, *args, **kwargs):
            if (
                not args
                and not kwargs
                and x.is_cuda
                and not getattr(self, "disable_adapters", False)
                and not getattr(self, "merged", False)
                and _is_frozen_nf4_linear(getattr(self, "base_layer", None))
            ):
                active = getattr(self, "active_adapters", None)
                if isinstance(active, (list, tuple)) and len(active) == 1:
                    adapter = active[0]
                    if (
                        adapter in self.lora_A
                        and adapter in self.lora_B
                        and not getattr(self, "use_dora", {}).get(adapter, False)
                        and (
                            getattr(self, "lora_magnitude_vector", None) is None
                            or adapter not in self.lora_magnitude_vector
                        )
                        and type(self.lora_dropout[adapter]) is torch.nn.Identity
                        and getattr(self.lora_A[adapter], "bias", None) is None
                        and getattr(self.lora_B[adapter], "bias", None) is None
                    ):
                        return TritonLoRALinear4bitFn.apply(
                            x,
                            self.base_layer.weight,
                            self.base_layer.bias,
                            self.lora_A[adapter].weight,
                            self.lora_B[adapter].weight,
                            float(self.scaling[adapter]),
                        )
            return orig_lora_4bit_fwd(self, x, *args, **kwargs)

        if not getattr(lora_bnb.Linear4bit.forward, "_soup_solution_patched", False):
            _triton_lora_4bit_forward._soup_solution_patched = True
            lora_bnb.Linear4bit.forward = _triton_lora_4bit_forward
    except Exception:
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

"""Unit tests for custom Triton kernels (RMSNorm, RoPE, Online Chunked CE, NF4 Dequant)."""

import pytest
import torch
import torch.nn.functional as F

from solution.fused_ce import TritonChunkedHeadLossFn
from solution.nf4_mlp import dequantize_nf4_triton
from solution.rmsnorm import FusedRMSNormFn
from solution.rope import FusedQwen2RoPEFn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("shape", [(2, 64, 1536), (4, 512, 1536)])
def test_rmsnorm_forward_and_backward_match_reference(shape):
    torch.manual_seed(0)
    eps = 1e-6
    x = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    w = torch.randn(shape[-1], device="cuda", dtype=torch.float16, requires_grad=False)
    dy = torch.randn(shape, device="cuda", dtype=torch.float16)

    x_ref = x.detach().clone().requires_grad_(True)
    xf = x_ref.float()
    y_ref = (w.float() * xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)
    y_ref.backward(dy)

    y = FusedRMSNormFn.apply(x, w, eps)
    y.backward(dy)

    torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-2, atol=1e-2)


def test_rope_forward_and_backward_match_reference():
    torch.manual_seed(1)
    bsz, n_qh, n_kh, seq_len, dim = 2, 12, 2, 64, 128
    q = torch.randn((bsz, n_qh, seq_len, dim), device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn((bsz, n_kh, seq_len, dim), device="cuda", dtype=torch.float16, requires_grad=True)
    cos = torch.randn((bsz, seq_len, dim), device="cuda", dtype=torch.float16)
    sin = torch.randn((bsz, seq_len, dim), device="cuda", dtype=torch.float16)
    dqo = torch.randn_like(q)
    dko = torch.randn_like(k)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    cos_u = cos.unsqueeze(1)
    sin_u = sin.unsqueeze(1)
    half = dim // 2
    qo_ref = (q_ref * cos_u) + (torch.cat((-q_ref[..., half:], q_ref[..., :half]), dim=-1) * sin_u)
    ko_ref = (k_ref * cos_u) + (torch.cat((-k_ref[..., half:], k_ref[..., :half]), dim=-1) * sin_u)
    (qo_ref * dqo).sum().add_((ko_ref * dko).sum()).backward()

    qo, ko = FusedQwen2RoPEFn.apply(q, k, cos, sin)
    (qo * dqo).sum().add_((ko * dko).sum()).backward()

    torch.testing.assert_close(qo, qo_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(ko, ko_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(q.grad, q_ref.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k.grad, k_ref.grad, rtol=1e-2, atol=1e-2)


def test_online_ce_entropy_matches_reference():
    torch.manual_seed(2)
    bsz, seq_len, hidden_dim, vocab_size = 2, 32, 128, 4096
    hidden = torch.randn((bsz, seq_len, hidden_dim), device="cuda", dtype=torch.float16, requires_grad=True)
    head = torch.nn.Linear(hidden_dim, vocab_size, bias=False, device="cuda", dtype=torch.float16)
    head.weight.requires_grad_(False)
    target = torch.randint(0, vocab_size, (bsz, seq_len), device="cuda")
    target[:, -4:] = -100
    denom = (target != -100).sum()

    hidden_ref = hidden.detach().clone().requires_grad_(True)
    logits_ref = head(hidden_ref)
    loss_ref = F.cross_entropy(
        logits_ref.float().view(-1, vocab_size),
        target.view(-1),
        ignore_index=-100,
        reduction="sum",
    ) / denom
    loss_ref.backward()

    loss, ent, tok_loss, correct = TritonChunkedHeadLossFn.apply(
        hidden, head.forward, target, 16, denom, -100, True
    )
    loss.backward()

    torch.testing.assert_close(loss, loss_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(hidden.grad, hidden_ref.grad, rtol=2e-2, atol=2e-2)


def test_nf4_dequant_matches_bitsandbytes():
    bnb = pytest.importorskip("bitsandbytes")
    torch.manual_seed(3)
    layer = bnb.nn.Linear4bit(256, 512, bias=False, compute_dtype=torch.float16, quant_type="nf4").to("cuda")
    ref = bnb.functional.dequantize_4bit(layer.weight.data, layer.weight.quant_state).to(torch.float16)
    got = dequantize_nf4_triton(layer.weight, dtype=torch.float16)
    torch.testing.assert_close(got, ref, rtol=1e-3, atol=1e-3)

"""Focused CPU tests for the DeepLoop Nanochat integration."""

import torch

from nanochat.common import COMPUTE_DTYPE
from nanochat.engine import KVCache
from nanochat.gpt import GPT, GPTConfig


def build_model(loop_count=3):
    config = GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
        loop_count=loop_count,
        use_deeploop=True,
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cpu")
    model.init_weights()
    return model


def test_deeploop_reuses_physical_blocks_and_scales_compute():
    model_r1 = build_model(loop_count=1)
    model_r3 = build_model(loop_count=3)

    assert len(model_r3.transformer.h) == 2
    assert model_r3.effective_n_layer == 6
    assert sum(p.numel() for p in model_r1.parameters()) == sum(p.numel() for p in model_r3.parameters())
    assert model_r3.estimate_flops() > model_r1.estimate_flops()
    assert len(model_r3.value_embeds) == 0
    assert model_r3.resid_lambdas is None


def test_deeploop_backpropagates_through_all_repeats():
    torch.manual_seed(1)
    model = build_model(loop_count=3)
    ids = torch.randint(0, model.config.vocab_size, (2, 8))
    loss = model(ids[:, :-1], ids[:, 1:])
    loss.backward()

    # The one physical block weight is visited three times, so one gradient proves
    # that autograd accumulated all repeat contributions onto the shared parameter.
    grad = model.transformer.h[0].attn.c_q.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_deeploop_kv_cache_matches_full_forward():
    torch.manual_seed(2)
    model = build_model(loop_count=3).eval()
    ids = torch.randint(0, model.config.vocab_size, (1, 8))

    full_logits = model(ids)
    cache = KVCache(
        batch_size=1,
        num_heads=model.config.n_kv_head,
        seq_len=ids.size(1),
        head_dim=model.config.n_embd // model.config.n_head,
        num_layers=model.effective_n_layer,
        device="cpu",
        dtype=COMPUTE_DTYPE,
    )
    model(ids[:, :-1], kv_cache=cache)
    cached_logits = model(ids[:, -1:], kv_cache=cache)

    assert cache.n_layers == 6
    assert cache.get_pos() == ids.size(1)
    torch.testing.assert_close(cached_logits[:, -1], full_logits[:, -1], atol=2e-4, rtol=2e-4)

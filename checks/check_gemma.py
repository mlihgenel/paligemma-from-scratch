"""Do the Gemma parts give the same output as HF?

Parts are tested from small to large:

    GemmaRMSNorm       -> HF GemmaRMSNorm (float32 and bfloat16 input)
    GemmaMLP           -> HF GemmaMLP
    GemmaRotaryEmbedding -> HF GemmaRotaryEmbedding + apply_rotary_pos_emb (cos/sin and rotated Q/K)
    repeat_kv          -> HF repeat_kv (MQA and GQA with n_kv > 1)
    KVCache            -> HF DynamicCache (prefill + token-by-token appends, length)
    GemmaAttention     -> HF GemmaAttention (eager, with and without a causal mask; cached prefill + token by token)
    GemmaDecoderLayer  -> HF GemmaDecoderLayer (attention + MLP + two norms, causal mask)
    GemmaModel         -> HF GemmaModel (embedding scale + RoPE + causal mask + all layers + final norm;
                          cached prefill + token by token, position and mask from the cache length)
    GemmaForCausalLM   -> HF GemmaForCausalLM (logits and weight tying)

Run (from the project root):
    .venv/bin/python checks/check_gemma.py

Nothing is downloaded: the HF modules are built from a small config with random weights, the weights
are copied into our modules and both are compared on the same input. The test with the real weights
is in the Colab notebook.
"""
import sys
from pathlib import Path

import torch
from transformers import DynamicCache
from transformers import GemmaConfig as HFGemmaConfig
from transformers import GemmaForCausalLM as HFGemmaForCausalLM
from transformers.models.gemma.modeling_gemma import GemmaAttention as HFGemmaAttention
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer as HFGemmaDecoderLayer
from transformers.models.gemma.modeling_gemma import GemmaMLP as HFGemmaMLP
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm as HFGemmaRMSNorm
from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding as HFGemmaRotaryEmbedding
from transformers.models.gemma.modeling_gemma import apply_rotary_pos_emb as hf_apply_rotary_pos_emb
from transformers.models.gemma.modeling_gemma import repeat_kv as hf_repeat_kv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import gemma  # noqa: E402

# Small config: the real Gemma 2B has hidden 2048, intermediate 16384. Same logic, fast locally
SMALL = dict(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=1,
    head_dim=16,
)
EPS = 1e-6
B, T = 2, 10
ATOL = 1e-5


def compare(name, ours, theirs, exact=False):
    same_shape = ours.shape == theirs.shape
    same_dtype = ours.dtype == theirs.dtype
    max_diff = (ours.float() - theirs.float()).abs().max().item() if same_shape else float("nan")
    if exact:
        ok = same_shape and same_dtype and torch.equal(ours, theirs)
    else:
        ok = same_shape and same_dtype and torch.allclose(ours, theirs, atol=ATOL)
    print(f"[{'OK ' if ok else 'FAIL'}] {name}: ours {tuple(ours.shape)} {ours.dtype}, "
          f"HF {tuple(theirs.shape)} {theirs.dtype}, max diff {max_diff:.2e}")
    return ok


def copy_weights(name, ours, hf):
    """Compare the keys and, if they match, copy the HF weights into our module."""
    ours_keys, hf_keys = set(ours.state_dict()), set(hf.state_dict())
    if ours_keys != hf_keys:
        print(f"[FAIL] {name}: state_dict keys do not match")
        print(f"   in ours but not in HF: {sorted(ours_keys - hf_keys)}")
        print(f"   in HF but not in ours: {sorted(hf_keys - ours_keys)}")
        return False
    ours.load_state_dict(hf.state_dict())
    return True


def check_rmsnorm():
    hf = HFGemmaRMSNorm(SMALL["hidden_size"], eps=EPS).eval()
    # The weight starts at zero. Left at zero, the scale is 1 on both sides and code that uses
    # the weight wrongly (or not at all) would still pass. So we fill it with random values.
    with torch.no_grad():
        hf.weight.normal_()
    ours = gemma.GemmaRMSNorm(SMALL["hidden_size"], eps=EPS).eval()
    if not copy_weights("GemmaRMSNorm", ours, hf):
        return False

    # Inputs can be large because Gemma multiplies embeddings by sqrt(hidden_size)
    x = torch.randn(B, T, SMALL["hidden_size"]) * 40
    with torch.no_grad():
        ok32 = compare("GemmaRMSNorm float32", ours(x), hf(x))

        # The model runs in bf16. With the same order of operations as HF the result must be bit-exact.
        # Written in Llama order (cast to bf16 first, then multiply) a small difference shows up here.
        ours.to(torch.bfloat16), hf.to(torch.bfloat16)
        x16 = x.to(torch.bfloat16)
        ok16 = compare("GemmaRMSNorm bfloat16 (bit-exact)", ours(x16), hf(x16), exact=True)
    return ok32 and ok16


def check_mlp():
    hf_config = HFGemmaConfig(**SMALL, hidden_act="gelu_pytorch_tanh")
    hf = HFGemmaMLP(hf_config).eval()
    ours = gemma.GemmaMLP(gemma.GemmaConfig(**SMALL)).eval()
    if not copy_weights("GemmaMLP", ours, hf):
        return False

    x = torch.randn(B, T, SMALL["hidden_size"])
    with torch.no_grad():
        return compare("GemmaMLP", ours(x), hf(x))


def check_rope():
    hf_config = HFGemmaConfig(**SMALL, hidden_act="gelu_pytorch_tanh")
    hf = HFGemmaRotaryEmbedding(hf_config)
    ours = gemma.GemmaRotaryEmbedding(SMALL["head_dim"], base=hf_config.rope_parameters["rope_theta"])
    # A buffer, so no copy_weights: we compute the frequencies ourselves and they must match
    ok_freq = compare("RoPE inv_freq", ours.inv_freq, hf.inv_freq)

    # Positions need not start at 0 (with a KV cache a generated token sits at position 260 or later)
    position_ids = torch.arange(T).unsqueeze(0).expand(B, -1) + torch.tensor([[0], [300]])
    q = torch.randn(B, SMALL["num_attention_heads"], T, SMALL["head_dim"])
    k = torch.randn(B, SMALL["num_key_value_heads"], T, SMALL["head_dim"])
    cos, sin = ours(q, position_ids)
    hf_cos, hf_sin = hf(q, position_ids)
    ok_cs = compare("RoPE cos", cos, hf_cos) & compare("RoPE sin", sin, hf_sin)

    q_ours, k_ours = gemma.apply_rotary_pos_emb(q, k, cos, sin)
    q_hf, k_hf = hf_apply_rotary_pos_emb(q, k, hf_cos, hf_sin)
    ok_qk = compare("RoPE rotated Q", q_ours, q_hf) & compare("RoPE rotated K", k_ours, k_hf)

    # Rotation must not change a vector's length, only its direction
    norm_kept = torch.allclose(q_ours.norm(dim=-1), q.norm(dim=-1), atol=ATOL)
    print(f"[{'OK ' if norm_kept else 'FAIL'}] RoPE keeps vector lengths")
    return bool(ok_freq and ok_cs and ok_qk and norm_kept)


def check_repeat_kv():
    n_heads = SMALL["num_attention_heads"]
    ok = True
    # 1: MQA like Gemma 2B. 2: GQA (n_kv > 1, where the grouping order shows). 4: n_rep = 1, nothing should change
    for n_kv in (1, 2, 4):
        n_rep = n_heads // n_kv
        x = torch.randn(B, n_kv, T, SMALL["head_dim"])
        ours, theirs = gemma.repeat_kv(x, n_rep), hf_repeat_kv(x, n_rep)
        ok &= compare(f"repeat_kv n_kv={n_kv}, n_rep={n_rep}", ours, theirs, exact=True)

        # Q head h must use KV head h // n_rep (groups are made of neighboring heads)
        grouped = all(torch.equal(ours[:, h], x[:, h // n_rep]) for h in range(n_heads))
        print(f"[{'OK ' if grouped else 'FAIL'}] repeat_kv n_kv={n_kv}: Q head h -> KV head h // n_rep")
        ok &= grouped
    return bool(ok)


def check_kv_cache():
    n_kv, hd, n_layers = SMALL["num_key_value_heads"], SMALL["head_dim"], SMALL["num_hidden_layers"]
    ours, theirs = gemma.KVCache(), DynamicCache()
    ok = ours.get_seq_length() == 0
    print(f"[{'OK ' if ok else 'FAIL'}] KVCache length is 0 when empty")
    # Prefill first (T tokens), then 3 token-by-token steps. Every step updates all layers
    for step, t_new in enumerate((T, 1, 1, 1)):
        for layer in range(n_layers):
            k, v = torch.randn(B, n_kv, t_new, hd), torch.randn(B, n_kv, t_new, hd)
            ok_k, ok_v = ours.update(k, v, layer)
            hf_k, hf_v = theirs.update(k, v, layer)
            if layer == n_layers - 1:
                ok &= compare(f"KVCache step {step} (T_new={t_new}) K, layer {layer}", ok_k, hf_k, exact=True)
                ok &= compare(f"KVCache step {step} (T_new={t_new}) V, layer {layer}", ok_v, hf_v, exact=True)
            else:
                ok &= torch.equal(ok_k, hf_k) and torch.equal(ok_v, hf_v)
    same_len = ours.get_seq_length() == theirs.get_seq_length() == T + 3
    print(f"[{'OK ' if same_len else 'FAIL'}] KVCache length: ours {ours.get_seq_length()}, "
          f"HF {theirs.get_seq_length()}, expected {T + 3}")
    return bool(ok and same_len)


def check_attention():
    hf_config = HFGemmaConfig(**SMALL, hidden_act="gelu_pytorch_tanh")
    hf_config._attn_implementation = "eager"      # not the sdpa/flash kernels, but the plain math
    hf = HFGemmaAttention(hf_config, layer_idx=0).eval()
    ours = gemma.GemmaAttention(gemma.GemmaConfig(**SMALL), layer_idx=0).eval()
    if not copy_weights("GemmaAttention", ours, hf):
        return False

    hf_rope = HFGemmaRotaryEmbedding(hf_config)
    x = torch.randn(B, T, SMALL["hidden_size"])
    position_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
    cos, sin = hf_rope(x, position_ids)

    # Causal mask: token i may only look at j <= i. Additive form: allowed 0, blocked -inf
    mask = torch.full((T, T), float("-inf")).triu(diagonal=1)[None, None].expand(B, 1, T, T)

    with torch.no_grad():
        ok_mask = compare("GemmaAttention (causal mask)",
                          ours(x, (cos, sin), mask), hf(x, (cos, sin), mask)[0])
        ok_nomask = compare("GemmaAttention (no mask)", ours(x, (cos, sin)), hf(x, (cos, sin), None)[0])

        # Does the causal mask work: changing a future token must not affect earlier outputs
        x2 = x.clone()
        x2[:, -1] += 10.0
        out1, out2 = ours(x, (cos, sin), mask), ours(x2, (cos, sin), mask)
        causal = torch.allclose(out1[:, :-1], out2[:, :-1], atol=ATOL) and not torch.allclose(out1[:, -1], out2[:, -1])
        print(f"[{'OK ' if causal else 'FAIL'}] GemmaAttention: changing the last token leaves earlier outputs unchanged")

        ok_cache = True
        if hasattr(gemma, "KVCache"):
            # With a cache: the first T-3 tokens are the prefill (causal mask), the last 3 go one by one. A single token's
            # mask is all zeros: it may look at the whole past, and the future is not in the cache anyway
            ours_cache, hf_cache = gemma.KVCache(), DynamicCache()
            P = T - 3
            outs = []
            for start, end in [(0, P)] + [(t, t + 1) for t in range(P, T)]:
                step_mask = mask[:, :, start:end, :end]                 # [B, 1, T_new, T_all]
                cs = (cos[:, start:end], sin[:, start:end])
                o = ours(x[:, start:end], cs, step_mask, kv_cache=ours_cache)
                h = hf(x[:, start:end], cs, step_mask, past_key_values=hf_cache)[0]
                ok_cache &= compare(f"GemmaAttention cached, tokens {start}..{end - 1}", o, h)
                outs.append(o)
            # The real claim: the cached steps joined together equal the full sequence processed at once
            ok_cache &= compare("GemmaAttention cached steps = uncached full sequence",
                                torch.cat(outs, dim=1), ours(x, (cos, sin), mask))
    return bool(ok_mask and ok_nomask and causal and ok_cache)


def hf_eager_config():
    cfg = HFGemmaConfig(**SMALL, hidden_act="gelu_pytorch_tanh")
    cfg._attn_implementation = "eager"      # not the sdpa/flash kernels, but the same math we wrote
    return cfg


def check_decoder_layer():
    hf_config = hf_eager_config()
    hf = HFGemmaDecoderLayer(hf_config, layer_idx=0).eval()
    ours = gemma.GemmaDecoderLayer(gemma.GemmaConfig(**SMALL), layer_idx=0).eval()

    # Randomize the norm weights. Left at zero, both norms behave the same and code that
    # swaps input_layernorm and post_attention_layernorm would still pass
    with torch.no_grad():
        hf.input_layernorm.weight.normal_()
        hf.post_attention_layernorm.weight.normal_()
    if not copy_weights("GemmaDecoderLayer", ours, hf):
        return False

    x = torch.randn(B, T, SMALL["hidden_size"]) * 40
    position_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
    cos, sin = HFGemmaRotaryEmbedding(hf_config)(x, position_ids)
    mask = gemma.make_causal_mask(T, x.dtype, x.device)
    with torch.no_grad():
        return compare("GemmaDecoderLayer", ours(x, (cos, sin), mask),
                       hf(x, attention_mask=mask, position_embeddings=(cos, sin)))


def randomize_norms(module):
    """Randomize all RMSNorm weights (left at zero, they all behave the same)."""
    with torch.no_grad():
        for n, p in module.named_parameters():
            if "norm" in n:
                p.normal_()


def scaled_embeds(ours_model):
    """Embeddings for the HF model with the scale already applied.

    transformers 5.x applies the sqrt(hidden_size) scale inside the embedding table
    (GemmaTextScaledWordEmbedding) and does not scale inputs_embeds passed in directly. Our
    GemmaModel applies the scale at the start of forward, on inputs_embeds (the older HF approach).
    To compare like with like: our model gets the raw embeddings, HF gets them multiplied by sqrt(d).
    """
    input_ids = torch.randint(0, SMALL["vocab_size"], (B, T))
    raw = ours_model.embed_tokens(input_ids)                       # [B, T, D], unscaled
    return raw, raw * torch.tensor(SMALL["hidden_size"] ** 0.5, dtype=raw.dtype)


def check_model():
    hf_config = hf_eager_config()
    hf = HFGemmaForCausalLM(hf_config).model.eval()   # the GemmaModel part
    ours = gemma.GemmaModel(gemma.GemmaConfig(**SMALL)).eval()
    randomize_norms(hf)
    if not copy_weights("GemmaModel", ours, hf):
        return False

    with torch.no_grad():
        raw, scaled = scaled_embeds(ours)
        ok = compare("GemmaModel", ours(raw), hf(inputs_embeds=scaled).last_hidden_state)

        if hasattr(gemma, "KVCache"):
            # No position or mask passed: both must follow from the cache length. HF does the same internally
            ours_cache, hf_cache = gemma.KVCache(), DynamicCache()
            P = T - 3
            outs = []
            for start, end in [(0, P)] + [(t, t + 1) for t in range(P, T)]:
                o = ours(raw[:, start:end], kv_cache=ours_cache)
                h = hf(inputs_embeds=scaled[:, start:end], past_key_values=hf_cache, use_cache=True).last_hidden_state
                ok &= compare(f"GemmaModel cached, tokens {start}..{end - 1}", o, h)
                outs.append(o)
            ok &= compare("GemmaModel cached steps = uncached full sequence", torch.cat(outs, dim=1), ours(raw))

            # Several new tokens with a past (past_len > 0, T > 1): the mask must be rectangular
            rect = gemma.make_causal_mask(3, torch.float32, torch.device("cpu"), past_len=4)[0, 0]
            expected = gemma.make_causal_mask(7, torch.float32, torch.device("cpu"))[0, 0, 4:]
            ok &= compare("make_causal_mask past_len=4, T=3 = last 3 rows of the square mask", rect, expected, exact=True)
        return bool(ok)


def check_causal_lm():
    hf_config = hf_eager_config()
    hf = HFGemmaForCausalLM(hf_config).eval()
    ours = gemma.GemmaForCausalLM(gemma.GemmaConfig(**SMALL)).eval()
    ours.tie_weights()   # in our code PaliGemma calls tying, here we do it by hand
    randomize_norms(hf)
    if not copy_weights("GemmaForCausalLM", ours, hf):
        return False

    # load_state_dict copies values but keeps the tying: both names must still be the same Parameter object
    tied = ours.lm_head.weight is ours.model.embed_tokens.weight
    print(f"[{'OK ' if tied else 'FAIL'}] weight tying: lm_head.weight is {'the same' if tied else 'a DIFFERENT'} "
          f"Parameter object as embed_tokens.weight")

    with torch.no_grad():
        raw, scaled = scaled_embeds(ours.model)
        ok = compare("GemmaForCausalLM logits", ours(raw), hf(inputs_embeds=scaled).logits)
    return tied and ok


def main():
    torch.manual_seed(0)
    checks = [
        ("GemmaRMSNorm", check_rmsnorm),
        ("GemmaMLP", check_mlp),
        ("GemmaRotaryEmbedding", check_rope),
        ("repeat_kv", check_repeat_kv),
        ("KVCache", check_kv_cache),
        ("GemmaAttention", check_attention),
        ("GemmaDecoderLayer", check_decoder_layer),
        ("GemmaModel", check_model),
        ("GemmaForCausalLM", check_causal_lm),
    ]

    results = []
    for cls_name, fn in checks:
        if not hasattr(gemma, cls_name):
            print(f"[ -- ] {cls_name}: not in gemma.py, skipped")
            continue
        try:
            results.append(fn())
        except Exception as e:
            print(f"[FAIL] {cls_name}: {type(e).__name__}: {e}")
            results.append(False)

    ok = bool(results) and all(results)
    print(f"\nResult: {'all parts match HF ✅' if ok else 'mismatch ❌'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

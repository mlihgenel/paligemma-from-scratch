"""Gemma experiments with the classes in gemma.py, at toy size.

Toy config: hidden 256, intermediate 2048, 6 layers. Embeddings follow the real Gemma's element
distribution (std ≈ 0.186), RMSNorm weights come from the first 256 dims of the real checkpoint
(pg_slices.json), MLP weights are random (std 0.02, HF initializer_range). Attention is replaced by
a random linear stub: these experiments measure the residual stream, not attention.
The experiments with the real GemmaAttention are in export_attention.py.

For the "what breaks?" experiments some variants rewrite the layer's forward here, but
the submodules (norms, MLP) are always the project's classes.

Run: .venv/bin/python handbook/export/export_gemma_toy.py
(run fetch_paligemma_slices.py first)
"""
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from common import OUT, Writer, small_list

import gemma  # noqa: E402

TOY = dict(vocab_size=512, hidden_size=256, intermediate_size=2048, num_hidden_layers=6,
           num_attention_heads=4, num_key_value_heads=1, head_dim=64)
D = TOY["hidden_size"]
T = 12


def load_slice(doc, key):
    x = doc[key] if isinstance(key, str) else key
    a = np.fromfile(OUT / x["$bin"], dtype={"f32": np.float32, "f16": np.float16}[x["dtype"]])
    return a.reshape(x["shape"]).astype(np.float32)


class Stub(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, x, *args, **kwargs):   # the layer passes (x, position_embeddings, attention_mask)
        return self.proj(x)


def layer_inputs(h):
    """The extra inputs the layer passes to attention. The stub ignores them, but the signature stays real."""
    B_, T_, _ = h.shape
    pe = gemma.GemmaRotaryEmbedding(TOY["head_dim"])(h, torch.arange(T_).unsqueeze(0).expand(B_, -1))
    return pe, gemma.make_causal_mask(T_, h.dtype, h.device)


def build_model(slices):
    torch.manual_seed(0)
    cfg = gemma.GemmaConfig(**TOY)
    # Initializing the real GemmaAttention's q/k/v/o would consume random numbers and shift the MLPs'
    # init. To produce the same numbers as the attention-free version we put an Identity there while
    # building, then swap in the Stub.
    real_attention = gemma.GemmaAttention
    gemma.GemmaAttention = lambda config, layer_idx: nn.Identity()
    try:
        model = gemma.GemmaModel(cfg).eval()
    finally:
        gemma.GemmaAttention = real_attention
    il = load_slice(slices, slices["norms"]["input_layernorm"])
    pa = load_slice(slices, slices["norms"]["post_attention_layernorm"])
    fn = load_slice(slices, slices["norms"]["final"])
    with torch.no_grad():
        model.embed_tokens.weight.normal_(0, 0.186)          # element std of the real Gemma embedding
        for i, layer in enumerate(model.layers):
            layer.self_attn = Stub(D)
            for lin in (layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj):
                nn.init.normal_(lin.weight, std=0.02)
            layer.input_layernorm.weight.copy_(torch.from_numpy(il[i, :D]))
            layer.post_attention_layernorm.weight.copy_(torch.from_numpy(pa[i, :D]))
        model.norm.weight.copy_(torch.from_numpy(fn[:D]))
    return cfg, model


# --- Layer variants (the submodules are the project's) ------------------------------------------
def layer_normal(layer, h):
    return layer(h, *layer_inputs(h))                 # the project's GemmaDecoderLayer.forward


def layer_residual_after_norm(layer, h):
    h = layer.input_layernorm(h)
    residual = h                                      # BUG: the residual is saved AFTER the norm
    h = residual + layer.self_attn(h)
    h = layer.post_attention_layernorm(h)
    residual = h
    return residual + layer.mlp(h)


def layer_post_norm(layer, h):
    # Original Transformer (post-norm): add first, then normalize
    h = layer.input_layernorm(h + layer.self_attn(h))
    return layer.post_attention_layernorm(h + layer.mlp(h))


def run_stream(model, emb, layer_fn, scale=True):
    """Trace the stream layer by layer: norm, similarity to the embedding, norm of the updates the sub-blocks write."""
    h = emb * torch.tensor(D ** 0.5, dtype=emb.dtype) if scale else emb
    x0 = h.clone()
    norms, cos0, attn_upd, mlp_upd = [h.norm(dim=-1).mean().item()], [1.0], [], []
    for layer in model.layers:
        before = h
        if layer_fn is layer_normal:
            a = layer.self_attn(layer.input_layernorm(h))
            mid = h + a
            m = layer.mlp(layer.post_attention_layernorm(mid))
            attn_upd.append(a.norm(dim=-1).mean().item())
            mlp_upd.append(m.norm(dim=-1).mean().item())
        h = layer_fn(layer, before)
        norms.append(h.norm(dim=-1).mean().item())
        cos0.append(F.cosine_similarity(h, x0, dim=-1).mean().item())
    out = model.norm(h)
    return {"norms": norms, "cos_to_embed": cos0, "attn_update": attn_upd, "mlp_update": mlp_upd,
            "final_norm": out.norm(dim=-1).mean().item()}


def rms_llama_order(norm, x):
    """Llama order: normalize in float32, cast back to the input dtype FIRST, THEN multiply."""
    out = norm._norm(x.float()).type_as(x)
    return out * (1.0 + norm.weight.to(x.dtype))


def rms_ignores_weight(norm, x):
    return norm._norm(x.float()).type_as(x)


def main():
    slices = json.loads((OUT / "pg_slices.json").read_text())
    cfg, model = build_model(slices)
    w = Writer("gemma_toy")
    doc = {"toy_config": TOY, "seq_len": T}
    torch.manual_seed(1)
    ids = torch.randint(0, TOY["vocab_size"], (1, T))

    with torch.no_grad():
        emb = model.embed_tokens(ids)                               # [1, T, D] ham

        # A) The residual stream and its variants
        doc["stream"] = {
            "normal": run_stream(model, emb, layer_normal),
            "residual_after_norm": run_stream(model, emb, layer_residual_after_norm),
            "post_norm": run_stream(model, emb, layer_post_norm),
            "no_sqrt_d": run_stream(model, emb, layer_normal, scale=False),
        }
        # Same as the project's GemmaModel.forward? (checks the normal variant)
        ours_out = model(emb)
        h = emb * torch.tensor(D ** 0.5)
        for layer in model.layers:
            h = layer(h, *layer_inputs(h))
        doc["stream_check_maxdiff"] = (model.norm(h) - ours_out).abs().max().item()

        # B) √d: embedding magnitudes (from real rows) and the scale
        real_rows = load_slice(slices, slices["embed_sample_rows"])            # [400, 2048]
        d_real = real_rows.shape[1]
        doc["sqrt_d"] = {
            "d": d_real,
            "sqrt_exact": d_real ** 0.5,
            "sqrt_bf16": torch.tensor(d_real ** 0.5, dtype=torch.bfloat16).item(),
            "elem_std_raw": float(real_rows.std()),
            "elem_std_scaled": float((real_rows * d_real ** 0.5).std()),
            "row_norm_raw": small_list(np.linalg.norm(real_rows, axis=1), 3),
        }
        named = load_slice(slices, slices["embed_named_rows"])
        doc["sqrt_d"]["named_norms"] = [
            {**m, "norm": float(np.linalg.norm(r))} for m, r in zip(slices["embed_named"], named)]

        # The OLD image-token path (no longer in the code): ÷√d (paligemma.py) then ×√d (gemma.py), in float32 and bf16.
        # The current code does not scale image features at all; this measurement shows why the path was shortened
        torch.manual_seed(2)
        img = torch.randn(1, 4, 2048) * 3
        rt = {}
        for dt in (torch.float32, torch.bfloat16):
            x = img.to(dt)
            back = (x / (2048 ** 0.5)) * torch.tensor(2048 ** 0.5, dtype=dt)
            rt[str(dt).replace("torch.", "")] = {
                "maxdiff": (back.float() - x.float()).abs().max().item(),
                "rel": ((back.float() - x.float()).abs() / x.float().abs().clamp_min(1e-6)).max().item(),
                "n_diff": int((back.float() != x.float()).sum()), "n": x.numel(),
            }
        doc["sqrt_d"]["roundtrip"] = rt

        # dtype of the normalizer: bf16 tensor, Python float or float32 tensor?
        xb = (torch.randn(1, T, 2048) * 0.186).to(torch.bfloat16)
        ref = xb * torch.tensor(2048 ** 0.5, dtype=torch.bfloat16)               # the project's code
        variants = {
            "python_float": xb * (2048 ** 0.5),
            "float32_tensor": xb * torch.tensor(2048 ** 0.5, dtype=torch.float32),
        }
        doc["sqrt_d"]["normalizer_variants"] = {
            k: {"dtype": str(v.dtype).replace("torch.", ""),
                "n_diff": int((v.float() != ref.float()).sum()), "n": ref.numel(),
                "maxdiff": (v.float() - ref.float()).abs().max().item()}
            for k, v in variants.items()}

        # C) RMSNorm: order of operations (Gemma vs Llama) in bf16
        norm = model.layers[0].input_layernorm
        x = torch.randn(4, T, D) * 40
        nb = gemma.GemmaRMSNorm(D, eps=cfg.rms_norm_eps)
        nb.load_state_dict(norm.state_dict())
        nb = nb.to(torch.bfloat16)
        xb = x.to(torch.bfloat16)
        g, l = nb(xb), rms_llama_order(nb, xb)
        doc["rms_order"] = {"n": g.numel(), "n_diff": int((g != l).sum()),
                            "maxdiff": (g.float() - l.float()).abs().max().item()}

        # D) Are buggy RMSNorms caught by a test with zero vs random weights?
        z = gemma.GemmaRMSNorm(D)
        r = gemma.GemmaRMSNorm(D)
        r.weight.normal_()
        xx = torch.randn(2, T, D) * 40
        doc["rms_zero_init_trap"] = {
            "zero_w": {"ignores_weight": (z(xx) - rms_ignores_weight(z, xx)).abs().max().item()},
            "random_w": {"ignores_weight": (r(xx) - rms_ignores_weight(r, xx)).abs().max().item()},
            "out_rms_zero_w": z(xx).pow(2).mean(-1).sqrt().mean().item(),
        }

        # E) If the two norms are swapped (invisible with zero weights, visible with real ones)
        layer = model.layers[0]
        hin = torch.randn(1, T, D) * 40

        def swapped(layer, h):
            residual = h
            h = residual + layer.self_attn(layer.post_attention_layernorm(h))
            residual = h
            return residual + layer.mlp(layer.input_layernorm(h))

        zero_layer = gemma.GemmaDecoderLayer(cfg, 0)
        zero_layer.self_attn = layer.self_attn
        zero_layer.mlp = layer.mlp
        doc["norm_swap"] = {
            "real_w": (layer(hin, *layer_inputs(hin)) - swapped(layer, hin)).abs().max().item(),
            "zero_w": (zero_layer(hin, *layer_inputs(hin)) - swapped(zero_layer, hin)).abs().max().item(),
        }

        # F) GeGLU internals (layer 0, with the real stream as input)
        mlp = model.layers[0].mlp
        hn = model.layers[0].post_attention_layernorm(emb * D ** 0.5)
        gate = mlp.gate_proj(hn)
        up = mlp.up_proj(hn)
        act = F.gelu(gate, approximate="tanh")
        prod = act * up
        k = 48                                          # the first 48 intermediate dims, for visualization
        doc["geglu"] = {
            "gate": w.arr(gate[0, :, :k]), "act": w.arr(act[0, :, :k]),
            "up": w.arr(up[0, :, :k]), "prod": w.arr(prod[0, :, :k]),
            "frac_closed": float((act.abs() < 0.05 * act.abs().max()).float().mean()),
            "frac_negative_act": float((act < 0).float().mean()),
            "token_independence_maxdiff": max(
                (mlp(hn[:, t:t + 1]) - mlp(hn)[:, t:t + 1]).abs().max().item() for t in range(T)),
        }

        # G) Container comparison: ModuleList, plain list, Sequential
        class WithModuleList(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(2, 2, bias=False) for _ in range(2)])

        class WithList(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [nn.Linear(2, 2, bias=False) for _ in range(2)]

        class WithSequential(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(*[nn.Linear(2, 2, bias=False) for _ in range(2)])

        doc["containers"] = {
            c.__name__: {"keys": list(c().state_dict().keys()),
                         "n_params": sum(p.numel() for p in c().parameters())}
            for c in (WithModuleList, WithList, WithSequential)}

        # H) Weight tying step by step (the project's GemmaForCausalLM)
        lm = gemma.GemmaForCausalLM(cfg)
        before = {"is_same": lm.lm_head.weight is lm.model.embed_tokens.weight,
                  "n_params": sum(p.numel() for p in lm.parameters())}
        lm.tie_weights()
        after = {"is_same": lm.lm_head.weight is lm.model.embed_tokens.weight,
                 "n_params": sum(p.numel() for p in lm.parameters()),
                 "state_dict_keys": [k for k in lm.state_dict() if "embed" in k or "lm_head" in k],
                 "same_storage": lm.state_dict()["lm_head.weight"].data_ptr()
                 == lm.state_dict()["model.embed_tokens.weight"].data_ptr()}
        sd = {k: v.clone() + 1 for k, v in lm.state_dict().items()}
        lm.load_state_dict(sd)
        after_load = {"is_same": lm.lm_head.weight is lm.model.embed_tokens.weight}
        lm.model.embed_tokens.weight.data[0, 0] = 123.0
        after_load["edit_visible_in_lm_head"] = lm.lm_head.weight[0, 0].item() == 123.0
        doc["tying"] = {"before": before, "after": after, "after_load": after_load,
                        "embed_params": TOY["vocab_size"] * D}

    w.save(doc)
    print(json.dumps({k: doc[k] for k in ("stream_check_maxdiff", "rms_order", "norm_swap", "rms_zero_init_trap", "tying")}, indent=1, ensure_ascii=False))
    print(json.dumps(doc["sqrt_d"]["normalizer_variants"]), doc["sqrt_d"]["roundtrip"])
    for k, v in doc["stream"].items():
        print(k, [round(x, 1) for x in v["norms"]], [round(x, 2) for x in v["cos_to_embed"]])
    print("geglu", doc["geglu"]["frac_closed"], doc["geglu"]["token_independence_maxdiff"])


if __name__ == "__main__":
    main()

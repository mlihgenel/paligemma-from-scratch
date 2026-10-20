"""Aşama 4 deneyleri: senin gemma.py sınıflarınla, oyuncak boyutta.

Oyuncak config: hidden 256, intermediate 2048, 6 katman. Embedding'ler gerçek Gemma'nın eleman
dağılımıyla (std ≈ 0.186), RMSNorm ağırlıkları gerçek checkpoint'in ilk 256 boyutundan
(pg_slices.json), MLP ağırlıkları rastgele (std 0.02, HF initializer_range). Attention yerine
check_gemma.py'deki gibi rastgele bir linear stub.

"Ne bozulur?" deneyleri için katmanın forward'ı bazı varyantlarda burada yeniden yazılıyor ama
alt modüller (norm'lar, MLP) her zaman senin sınıfların.

Çalıştır: .venv/bin/python handbook/export/export_gemma_toy.py
(önce fetch_paligemma_slices.py çalışmış olmalı)
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

    def forward(self, x):
        return self.proj(x)


def build_model(slices):
    torch.manual_seed(0)
    cfg = gemma.GemmaConfig(**TOY)
    model = gemma.GemmaModel(cfg).eval()
    il = load_slice(slices, slices["norms"]["input_layernorm"])
    pa = load_slice(slices, slices["norms"]["post_attention_layernorm"])
    fn = load_slice(slices, slices["norms"]["final"])
    with torch.no_grad():
        model.embed_tokens.weight.normal_(0, 0.186)          # gerçek Gemma embedding eleman std'si
        for i, layer in enumerate(model.layers):
            layer.self_attn = Stub(D)
            for lin in (layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj):
                nn.init.normal_(lin.weight, std=0.02)
            layer.input_layernorm.weight.copy_(torch.from_numpy(il[i, :D]))
            layer.post_attention_layernorm.weight.copy_(torch.from_numpy(pa[i, :D]))
        model.norm.weight.copy_(torch.from_numpy(fn[:D]))
    return cfg, model


# --- Katman varyantları (alt modüller senin) --------------------------------------------------
def layer_normal(layer, h):
    return layer(h)                                   # senin GemmaDecoderLayer.forward'ın


def layer_residual_after_norm(layer, h):
    h = layer.input_layernorm(h)
    residual = h                                      # HATA: residual norm'dan SONRA saklanıyor
    h = residual + layer.self_attn(h)
    h = layer.post_attention_layernorm(h)
    residual = h
    return residual + layer.mlp(h)


def layer_post_norm(layer, h):
    # Orijinal Transformer (post-norm): önce topla, sonra normalize et
    h = layer.input_layernorm(h + layer.self_attn(h))
    return layer.post_attention_layernorm(h + layer.mlp(h))


def run_stream(model, emb, layer_fn, scale=True):
    """Akışı katman katman izle: norm, embedding'e benzerlik, alt blokların yazdığı güncellemelerin normu."""
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
    """Llama sırası: float32'de normalize, ÖNCE girdi dtype'ına dön, SONRA çarp."""
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

        # A) Residual akışı ve varyantları
        doc["stream"] = {
            "normal": run_stream(model, emb, layer_normal),
            "residual_after_norm": run_stream(model, emb, layer_residual_after_norm),
            "post_norm": run_stream(model, emb, layer_post_norm),
            "no_sqrt_d": run_stream(model, emb, layer_normal, scale=False),
        }
        # Senin GemmaModel.forward'ın ile aynı mı? (normal varyant doğrulaması)
        ours_out = model(emb)
        h = emb * torch.tensor(D ** 0.5)
        for layer in model.layers:
            h = layer(h)
        doc["stream_check_maxdiff"] = (model.norm(h) - ours_out).abs().max().item()

        # B) √d: embedding büyüklükleri (gerçek satırlardan) ve ölçek
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

        # Görsel token yolu: ÷√d (paligemma.py) sonra ×√d (gemma.py), float32 ve bf16'da
        torch.manual_seed(2)
        img = torch.randn(1, 4, 2048) * 3
        rt = {}
        for dt in (torch.float32, torch.bfloat16):
            x = img.to(dt)
            back = (x / (2048 ** 0.5)) * torch.tensor(2048 ** 0.5, dtype=dt)
            rt[str(dt).replace("torch.", "")] = {
                "maxdiff": (back.float() - x.float()).abs().max().item(),
                "rel": ((back.float() - x.float()).abs() / x.float().abs().clamp_min(1e-6)).max().item(),
            }
        doc["sqrt_d"]["roundtrip"] = rt

        # normalizer'ın dtype'ı: bf16 tensör mü, Python float mı, float32 tensör mü?
        xb = (torch.randn(1, T, 2048) * 0.186).to(torch.bfloat16)
        ref = xb * torch.tensor(2048 ** 0.5, dtype=torch.bfloat16)               # senin kodun
        variants = {
            "python_float": xb * (2048 ** 0.5),
            "float32_tensor": xb * torch.tensor(2048 ** 0.5, dtype=torch.float32),
        }
        doc["sqrt_d"]["normalizer_variants"] = {
            k: {"dtype": str(v.dtype).replace("torch.", ""),
                "n_diff": int((v.float() != ref.float()).sum()), "n": ref.numel(),
                "maxdiff": (v.float() - ref.float()).abs().max().item()}
            for k, v in variants.items()}

        # C) RMSNorm: işlem sırası (Gemma ve Llama) bf16'da
        norm = model.layers[0].input_layernorm
        x = torch.randn(4, T, D) * 40
        nb = gemma.GemmaRMSNorm(D, eps=cfg.rms_norm_eps)
        nb.load_state_dict(norm.state_dict())
        nb = nb.to(torch.bfloat16)
        xb = x.to(torch.bfloat16)
        g, l = nb(xb), rms_llama_order(nb, xb)
        doc["rms_order"] = {"n": g.numel(), "n_diff": int((g != l).sum()),
                            "maxdiff": (g.float() - l.float()).abs().max().item()}

        # D) Sıfır ve rastgele ağırlıkla hatalı RMSNorm'lar testte yakalanıyor mu?
        z = gemma.GemmaRMSNorm(D)
        r = gemma.GemmaRMSNorm(D)
        r.weight.normal_()
        xx = torch.randn(2, T, D) * 40
        doc["rms_zero_init_trap"] = {
            "zero_w": {"ignores_weight": (z(xx) - rms_ignores_weight(z, xx)).abs().max().item()},
            "random_w": {"ignores_weight": (r(xx) - rms_ignores_weight(r, xx)).abs().max().item()},
            "out_rms_zero_w": z(xx).pow(2).mean(-1).sqrt().mean().item(),
        }

        # E) İki norm yer değiştirirse (sıfır ağırlıkta görünmez, gerçek ağırlıkta görünür)
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
            "real_w": (layer(hin) - swapped(layer, hin)).abs().max().item(),
            "zero_w": (zero_layer(hin) - swapped(zero_layer, hin)).abs().max().item(),
        }

        # F) GeGLU iç değerleri (katman 0, gerçek akış girdisiyle)
        mlp = model.layers[0].mlp
        hn = model.layers[0].post_attention_layernorm(emb * D ** 0.5)
        gate = mlp.gate_proj(hn)
        up = mlp.up_proj(hn)
        act = F.gelu(gate, approximate="tanh")
        prod = act * up
        k = 48                                          # görselleştirme için ilk 48 ara boyut
        doc["geglu"] = {
            "gate": w.arr(gate[0, :, :k]), "act": w.arr(act[0, :, :k]),
            "up": w.arr(up[0, :, :k]), "prod": w.arr(prod[0, :, :k]),
            "frac_closed": float((act.abs() < 0.05 * act.abs().max()).float().mean()),
            "frac_negative_act": float((act < 0).float().mean()),
            "token_independence_maxdiff": max(
                (mlp(hn[:, t:t + 1]) - mlp(hn)[:, t:t + 1]).abs().max().item() for t in range(T)),
        }

        # G) Konteyner karşılaştırması: ModuleList, düz list, Sequential
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

        # H) Weight tying adım adım (senin GemmaForCausalLM'in)
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

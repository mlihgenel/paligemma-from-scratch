"""Attention sayfalarının (RoPE, MQA, GemmaAttention) deneyleri: senin gemma.py sınıflarınla.

- RoPE: gerçek head_dim = 256 için frekanslar (senin GemmaRotaryEmbedding'in), "ilk yarı / ikinci yarı"
  ile "komşu çiftler" düzeninin farkı, skorun yalnızca konum farkına bağlı olduğunun ölçümü.
- repeat_kv: expand'in veri kopyalamadığı, reshape'in kopyaladığı (data_ptr).
- GemmaAttention: oyuncak boyutta, rastgele ama belirgin ağırlıklarla. Maskeli ve maskesiz attention
  ağırlıkları elle yeniden hesaplanıp senin forward'ının çıktısıyla karşılaştırılıyor. Nedensel maske
  gerçekten işe yarıyor mu (son token değişince öncekiler değişiyor mu)?

Çalıştır: .venv/bin/python handbook/export/export_attention.py
"""
import torch
import torch.nn.functional as F

from common import Writer, small_list

import gemma  # noqa: E402

TOY = dict(vocab_size=128, hidden_size=64, intermediate_size=256, num_hidden_layers=2,
           num_attention_heads=4, num_key_value_heads=1, head_dim=16)
T = 8


def adjacent_pairs_rope(x, cos_half, sin_half):
    """Makaledeki yazım: (x0, x1), (x2, x3)... komşu çiftler. cos/sin: [T, d/2]."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos_half - x2 * sin_half
    out[..., 1::2] = x2 * cos_half + x1 * sin_half
    return out


def main():
    torch.manual_seed(3)
    w = Writer("attention_toy")
    doc = {"toy_config": TOY, "seq_len": T}

    with torch.no_grad():
        # --- RoPE: gerçek head_dim = 256 frekansları (senin sınıfın) ---
        rope_real = gemma.GemmaRotaryEmbedding(256, base=10000.0)
        doc["rope_real"] = {"head_dim": 256, "base": 10000.0, "inv_freq": small_list(rope_real.inv_freq, 8)}

        # --- RoPE: iki düzen. Aynı q, aynı açılar; biri ilk/ikinci yarı (senin kodun), biri komşu çiftler ---
        d = TOY["head_dim"]
        rope = gemma.GemmaRotaryEmbedding(d)
        pos = torch.arange(T).unsqueeze(0)                          # [1, T]
        q = torch.randn(1, 1, T, d)
        x_dummy = torch.zeros(1, T, d)
        cos, sin = rope(x_dummy, pos)                               # [1, T, d]
        k_dummy = torch.zeros(1, 1, T, d)
        ours, _ = gemma.apply_rotary_pos_emb(q, k_dummy, cos, sin)
        cos_half, sin_half = cos[0, :, : d // 2], sin[0, :, : d // 2]
        adj = adjacent_pairs_rope(q[0, 0], cos_half, sin_half)      # [T, d] komşu çift düzeni
        # Aynı dönüşüm, boyutlar yeniden sıralanmış halde: komşu çift (2i, 2i+1) -> (i, i + d/2)
        perm = torch.cat([torch.arange(0, d, 2), torch.arange(1, d, 2)])
        # 1) Aynı q'ya iki düzeni uygula: farklı çiftler döndüğü için sonuç farklı
        # 2) q'nun boyutlarını yeniden sırala (komşu çiftler -> ilk/ikinci yarı), senin kodundan geçir,
        #    komşu çift sonucunun aynı sırayla yeniden sıralanmışıyla karşılaştır: aynı döndürme
        q_perm = q[0, 0][:, perm]                                   # [T, d], çiftler artık (i, i + d/2)
        ours_perm, _ = gemma.apply_rotary_pos_emb(q_perm[None, None], k_dummy, cos, sin)
        doc["rope_layout"] = {
            "half_vs_adjacent_maxdiff": (ours[0, 0] - adj).abs().max().item(),
            "half_vs_adjacent_permuted_maxdiff": (ours_perm[0, 0] - adj[:, perm]).abs().max().item(),
        }

        # --- RoPE: skor yalnızca konum farkına bağlı mı? (senin kodunla) ---
        q1 = torch.randn(1, 1, 1, d)
        k1 = torch.randn(1, 1, 1, d)

        def score(m, n):
            cm, sm = rope(torch.zeros(1, 1, d), torch.tensor([[m]]))
            cn, sn = rope(torch.zeros(1, 1, d), torch.tensor([[n]]))
            qm, _ = gemma.apply_rotary_pos_emb(q1, k1, cm, sm)
            _, kn = gemma.apply_rotary_pos_emb(q1, k1, cn, sn)
            return (qm * kn).sum().item()

        base = [score(m + 5, m) for m in (0, 10, 300, 3000)]       # hep aynı fark: 5
        doc["rope_relative"] = {"scores_diff5": base, "spread": max(base) - min(base),
                                "score_diff5": base[0], "score_diff6": score(6, 0), "score_diff_neg5": score(0, 5)}

        # --- repeat_kv: expand veri kopyalamıyor. reshape: n_kv = 1 (MQA) iken de kopyalamıyor (bir görünüm),
        #     n_kv > 1 iken kopyalıyor (eksenleri birleştirmek için bellek düzeni sürekli olmak zorunda) ---
        rk = {}
        for n_kv in (1, 2):
            x = torch.randn(1, n_kv, T, d)
            n_rep = 4 // n_kv
            expanded = x[:, :, None, :, :].expand(1, n_kv, n_rep, T, d)
            repeated = gemma.repeat_kv(x, n_rep)
            rk[n_kv] = {
                "expand_shares_memory": expanded.data_ptr() == x.data_ptr(),
                "expand_stride_rep": expanded.stride(2),            # 0: aynı veri n_rep kez "görünüyor"
                "repeat_shares_memory": repeated.data_ptr() == x.data_ptr(),
                "repeat_shape": list(repeated.shape),
                "all_equal": all(torch.equal(repeated[:, h], x[:, h // n_rep]) for h in range(4)),
            }
        doc["repeat_kv"] = rk

        # --- GemmaAttention: gerçek sınıf, belirgin (std 0.3) ağırlıklar ---
        cfg = gemma.GemmaConfig(**TOY)
        attn = gemma.GemmaAttention(cfg, layer_idx=0).eval()
        for lin in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
            lin.weight.normal_(0, 0.3)
        x = torch.randn(1, T, cfg.hidden_size)
        rope_m = gemma.GemmaRotaryEmbedding(cfg.head_dim)
        pe = rope_m(x, torch.arange(T).unsqueeze(0))
        causal = gemma.make_causal_mask(T, x.dtype, x.device)

        def manual_weights(x, mask):
            """forward'ın içindekiyle aynı adımlar, ara değerleri görebilmek için dışarıda."""
            B_, T_, _ = x.shape
            q = attn.q_proj(x).view(B_, T_, attn.num_heads, attn.head_dim).transpose(1, 2)
            k = attn.k_proj(x).view(B_, T_, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
            q, k = gemma.apply_rotary_pos_emb(q, k, *pe)
            k = gemma.repeat_kv(k, attn.num_kv_groups)
            scores = q @ k.transpose(2, 3) * attn.scaling
            if mask is not None:
                scores = scores + mask
            return F.softmax(scores, dim=-1, dtype=torch.float32), scores

        wm, sm_ = manual_weights(x, causal)
        wn, sn_ = manual_weights(x, None)
        v = attn.v_proj(x).view(1, T, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
        out_manual = attn.o_proj((wm @ gemma.repeat_kv(v, attn.num_kv_groups)).transpose(1, 2).reshape(1, T, -1))
        doc["attention"] = {
            "weights_causal": w.arr(wm[0]),                         # [4 head, T, T]
            "weights_nomask": w.arr(wn[0]),
            "scores_causal": w.arr(torch.nan_to_num(sm_[0].clamp_min(-1e4))),   # maskeli yerler -1e4 ile sınırlı
            "manual_vs_forward_maxdiff": (out_manual - attn(x, pe, causal)).abs().max().item(),
            "row_sums_maxdiff": (wm.sum(-1) - 1).abs().max().item(),
            "upper_triangle_max": wm.triu(1).max().item(),
        }

        # Nedensel maske işe yarıyor mu: son token'ı değiştir, önceki çıktılar değişiyor mu?
        x2 = x.clone()
        x2[:, -1] += 5.0
        for name, mask in (("with_mask", causal), ("without_mask", None)):
            o1, o2 = attn(x, pe, mask), attn(x2, pe, mask)
            doc["attention"][f"leak_{name}"] = (o1[:, :-1] - o2[:, :-1]).abs().max().item()

        # Ölçek: head_dim büyüdükçe iç çarpımların yayılımı √head_dim kadar büyür
        stats = {}
        for hd in (16, 64, 256):
            qq, kk = torch.randn(4000, hd), torch.randn(4000, hd)
            raw = (qq * kk).sum(-1)
            stats[hd] = {"std_raw": raw.std().item(), "std_scaled": (raw * hd ** -0.5).std().item()}
        doc["scale_stats"] = stats

        # Gerçek config'teki parametre sayıları (meta cihazı gerekmeden, formülle)
        D_, H, KV, HD = 2048, 8, 1, 256
        doc["params_real"] = {"q": D_ * H * HD, "k": D_ * KV * HD, "v": D_ * KV * HD, "o": H * HD * D_}

    w.save(doc)
    print({k: doc[k] for k in ("rope_layout", "rope_relative", "repeat_kv", "scale_stats", "params_real")})
    print({k: v for k, v in doc["attention"].items() if not isinstance(v, dict)})


if __name__ == "__main__":
    main()

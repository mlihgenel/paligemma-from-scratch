"""Experiments for the attention pages (RoPE, MQA, GemmaAttention), using the classes in gemma.py.

- RoPE: frequencies for the real head_dim = 256 (from GemmaRotaryEmbedding), the difference between the
  "first half / second half" and "adjacent pairs" layouts, and a measurement that scores depend only on position differences.
- repeat_kv: expand copies no data, reshape does (data_ptr).
- GemmaAttention: toy size, with random but pronounced weights. Masked and unmasked attention weights
  are recomputed by hand and compared with the forward output. Does the causal mask
  really work (do earlier outputs change when the last token changes)?

Run: .venv/bin/python handbook/export/export_attention.py
"""
import torch
import torch.nn.functional as F

from common import Writer, small_list

import gemma  # noqa: E402

TOY = dict(vocab_size=128, hidden_size=64, intermediate_size=256, num_hidden_layers=2,
           num_attention_heads=4, num_key_value_heads=1, head_dim=16)
T = 8


def adjacent_pairs_rope(x, cos_half, sin_half):
    """The paper's notation: (x0, x1), (x2, x3)... adjacent pairs. cos/sin: [T, d/2]."""
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
        # --- RoPE: frequencies for the real head_dim = 256 (the project's class) ---
        rope_real = gemma.GemmaRotaryEmbedding(256, base=10000.0)
        doc["rope_real"] = {"head_dim": 256, "base": 10000.0, "inv_freq": small_list(rope_real.inv_freq, 8)}

        # --- RoPE: two layouts. Same q, same angles; one first/second half (the project's code), one adjacent pairs ---
        d = TOY["head_dim"]
        rope = gemma.GemmaRotaryEmbedding(d)
        pos = torch.arange(T).unsqueeze(0)                          # [1, T]
        q = torch.randn(1, 1, T, d)
        x_dummy = torch.zeros(1, T, d)
        cos, sin = rope(x_dummy, pos)                               # [1, T, d]
        k_dummy = torch.zeros(1, 1, T, d)
        ours, _ = gemma.apply_rotary_pos_emb(q, k_dummy, cos, sin)
        cos_half, sin_half = cos[0, :, : d // 2], sin[0, :, : d // 2]
        adj = adjacent_pairs_rope(q[0, 0], cos_half, sin_half)      # [T, d] adjacent-pair layout
        # The same transform with the dims reordered: adjacent pair (2i, 2i+1) -> (i, i + d/2)
        perm = torch.cat([torch.arange(0, d, 2), torch.arange(1, d, 2)])
        # 1) Apply both layouts to the same q: different pairs rotate, so the results differ
        # 2) Reorder q's dims (adjacent pairs -> first/second half), run it through the project's code,
        #    and compare with the adjacent-pair result reordered the same way: the same rotation
        q_perm = q[0, 0][:, perm]                                   # [T, d], pairs are now (i, i + d/2)
        ours_perm, _ = gemma.apply_rotary_pos_emb(q_perm[None, None], k_dummy, cos, sin)
        doc["rope_layout"] = {
            "half_vs_adjacent_maxdiff": (ours[0, 0] - adj).abs().max().item(),
            "half_vs_adjacent_permuted_maxdiff": (ours_perm[0, 0] - adj[:, perm]).abs().max().item(),
        }

        # --- RoPE: does the score depend only on the position difference? (with the project's code) ---
        q1 = torch.randn(1, 1, 1, d)
        k1 = torch.randn(1, 1, 1, d)

        def score(m, n):
            cm, sm = rope(torch.zeros(1, 1, d), torch.tensor([[m]]))
            cn, sn = rope(torch.zeros(1, 1, d), torch.tensor([[n]]))
            qm, _ = gemma.apply_rotary_pos_emb(q1, k1, cm, sm)
            _, kn = gemma.apply_rotary_pos_emb(q1, k1, cn, sn)
            return (qm * kn).sum().item()

        base = [score(m + 5, m) for m in (0, 10, 300, 3000)]       # always the same difference: 5
        doc["rope_relative"] = {"scores_diff5": base, "spread": max(base) - min(base),
                                "score_diff5": base[0], "score_diff6": score(6, 0), "score_diff_neg5": score(0, 5)}

        # --- repeat_kv: expand copies no data. reshape: with n_kv = 1 (MQA) it does not copy either (a view),
        #     with n_kv > 1 it copies (merging axes needs a contiguous memory layout) ---
        rk = {}
        for n_kv in (1, 2):
            x = torch.randn(1, n_kv, T, d)
            n_rep = 4 // n_kv
            expanded = x[:, :, None, :, :].expand(1, n_kv, n_rep, T, d)
            repeated = gemma.repeat_kv(x, n_rep)
            rk[n_kv] = {
                "expand_shares_memory": expanded.data_ptr() == x.data_ptr(),
                "expand_stride_rep": expanded.stride(2),            # 0: the same data "appears" n_rep times
                "repeat_shares_memory": repeated.data_ptr() == x.data_ptr(),
                "repeat_shape": list(repeated.shape),
                "all_equal": all(torch.equal(repeated[:, h], x[:, h // n_rep]) for h in range(4)),
            }
        doc["repeat_kv"] = rk

        # --- GemmaAttention: the real class, pronounced (std 0.3) weights ---
        cfg = gemma.GemmaConfig(**TOY)
        attn = gemma.GemmaAttention(cfg, layer_idx=0).eval()
        for lin in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
            lin.weight.normal_(0, 0.3)
        x = torch.randn(1, T, cfg.hidden_size)
        rope_m = gemma.GemmaRotaryEmbedding(cfg.head_dim)
        pe = rope_m(x, torch.arange(T).unsqueeze(0))
        causal = gemma.make_causal_mask(T, x.dtype, x.device)

        def manual_weights(x, mask):
            """The same steps as inside forward, done outside so the intermediate values are visible."""
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
            "scores_causal": w.arr(torch.nan_to_num(sm_[0].clamp_min(-1e4))),   # masked positions clamped at -1e4
            "manual_vs_forward_maxdiff": (out_manual - attn(x, pe, causal)).abs().max().item(),
            "row_sums_maxdiff": (wm.sum(-1) - 1).abs().max().item(),
            "upper_triangle_max": wm.triu(1).max().item(),
        }

        # Does the causal mask work: change the last token, do earlier outputs change?
        x2 = x.clone()
        x2[:, -1] += 5.0
        for name, mask in (("with_mask", causal), ("without_mask", None)):
            o1, o2 = attn(x, pe, mask), attn(x2, pe, mask)
            doc["attention"][f"leak_{name}"] = (o1[:, :-1] - o2[:, :-1]).abs().max().item()

        # Scale: as head_dim grows, the spread of dot products grows by √head_dim
        stats = {}
        for hd in (16, 64, 256):
            qq, kk = torch.randn(4000, hd), torch.randn(4000, hd)
            raw = (qq * kk).sum(-1)
            stats[hd] = {"std_raw": raw.std().item(), "std_scaled": (raw * hd ** -0.5).std().item()}
        doc["scale_stats"] = stats

        # Parameter counts of the real config (by formula, no meta device needed)
        D_, H, KV, HD = 2048, 8, 1, 256
        doc["params_real"] = {"q": D_ * H * HD, "k": D_ * KV * HD, "v": D_ * KV * HD, "o": H * HD * D_}

    w.save(doc)
    print({k: doc[k] for k in ("rope_layout", "rope_relative", "repeat_kv", "scale_stats", "params_real")})
    print({k: v for k, v in doc["attention"].items() if not isinstance(v, dict)})


if __name__ == "__main__":
    main()

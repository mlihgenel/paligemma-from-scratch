"""Gemma parçaları HF ile aynı çıktıyı veriyor mu?

Parçalar küçükten büyüğe test edilir. src/gemma.py'de henüz yazılmamış sınıflar atlanır,
yani her yeni sınıfı yazdıkça script'i yeniden çalıştırabilirsin:

    GemmaRMSNorm       -> HF GemmaRMSNorm (float32 ve bfloat16 girdiyle)
    GemmaMLP           -> HF GemmaMLP
    GemmaRotaryEmbedding -> HF GemmaRotaryEmbedding + apply_rotary_pos_emb (cos/sin ve döndürülmüş Q/K)
    repeat_kv          -> HF repeat_kv (MQA ve n_kv > 1 olan GQA durumları)
    KVCache            -> HF DynamicCache (prefill + token token ekleme, uzunluk)
    GemmaAttention     -> HF GemmaAttention (eager, nedensel maskeli ve maskesiz; cache'le prefill + token token)
    GemmaDecoderLayer  -> HF GemmaDecoderLayer (attention + MLP + iki norm, nedensel maskeyle)
    GemmaModel         -> HF GemmaModel (embedding ölçeği + RoPE + nedensel maske + bütün katmanlar + son norm;
                          cache'le prefill + token token, konum ve maske cache uzunluğundan)
    GemmaForCausalLM   -> HF GemmaForCausalLM (logit'ler ve weight tying)

Çalıştır (proje kökünden):
    .venv/bin/python checks/check_gemma.py

Hiçbir şey indirilmez: HF modülleri küçük bir config ve rastgele ağırlıklarla kurulur, ağırlıklar
bizim modüllere kopyalanır ve aynı girdiyle karşılaştırılır. Gerçek ağırlıklarla test tam model
kurulunca (Colab) yapılacak.
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

# Küçük config: gerçek Gemma 2B'de hidden 2048, intermediate 16384. Mantık aynı, yerelde hızlı
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
    print(f"[{'OK ' if ok else 'HATA'}] {name}: bizim {tuple(ours.shape)} {ours.dtype}, "
          f"HF {tuple(theirs.shape)} {theirs.dtype}, max fark {max_diff:.2e}")
    return ok


def copy_weights(name, ours, hf):
    """Anahtarları karşılaştır, tutuyorsa HF ağırlıklarını bizim modüle kopyala."""
    ours_keys, hf_keys = set(ours.state_dict()), set(hf.state_dict())
    if ours_keys != hf_keys:
        print(f"[HATA] {name}: state_dict anahtarları tutmuyor")
        print(f"   bizde olup HF'de olmayan: {sorted(ours_keys - hf_keys)}")
        print(f"   HF'de olup bizde olmayan: {sorted(hf_keys - ours_keys)}")
        return False
    ours.load_state_dict(hf.state_dict())
    return True


def check_rmsnorm():
    hf = HFGemmaRMSNorm(SMALL["hidden_size"], eps=EPS).eval()
    # Ağırlık sıfırla başlıyor. Sıfır kalırsa ölçek her iki tarafta da 1 olur ve weight'i yanlış
    # kullanan (ya da hiç kullanmayan) bir kod da testi geçer. Bu yüzden rastgele dolduruyoruz.
    with torch.no_grad():
        hf.weight.normal_()
    ours = gemma.GemmaRMSNorm(SMALL["hidden_size"], eps=EPS).eval()
    if not copy_weights("GemmaRMSNorm", ours, hf):
        return False

    # Gemma'da embedding'ler sqrt(hidden_size) ile çarpıldığı için girdiler büyük olabiliyor
    x = torch.randn(B, T, SMALL["hidden_size"]) * 40
    with torch.no_grad():
        ok32 = compare("GemmaRMSNorm float32", ours(x), hf(x))

        # Model bf16'da çalışacak. İşlem sırası HF ile aynıysa sonuç bit bit aynı olmalı.
        # Llama sırasıyla (önce bf16'ya dön, sonra çarp) yazılmışsa burada küçük bir fark çıkar.
        ours.to(torch.bfloat16), hf.to(torch.bfloat16)
        x16 = x.to(torch.bfloat16)
        ok16 = compare("GemmaRMSNorm bfloat16 (birebir)", ours(x16), hf(x16), exact=True)
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
    # Buffer olduğu için copy_weights yok: frekansları kendimiz hesaplıyoruz, aynı çıkmalı
    ok_freq = compare("RoPE inv_freq", ours.inv_freq, hf.inv_freq)

    # Konumlar 0'dan başlamak zorunda değil (KV cache'te üretilen token 260. konumda olacak)
    position_ids = torch.arange(T).unsqueeze(0).expand(B, -1) + torch.tensor([[0], [300]])
    q = torch.randn(B, SMALL["num_attention_heads"], T, SMALL["head_dim"])
    k = torch.randn(B, SMALL["num_key_value_heads"], T, SMALL["head_dim"])
    cos, sin = ours(q, position_ids)
    hf_cos, hf_sin = hf(q, position_ids)
    ok_cs = compare("RoPE cos", cos, hf_cos) & compare("RoPE sin", sin, hf_sin)

    q_ours, k_ours = gemma.apply_rotary_pos_emb(q, k, cos, sin)
    q_hf, k_hf = hf_apply_rotary_pos_emb(q, k, hf_cos, hf_sin)
    ok_qk = compare("RoPE döndürülmüş Q", q_ours, q_hf) & compare("RoPE döndürülmüş K", k_ours, k_hf)

    # Döndürme vektörün uzunluğunu değiştirmemeli: sadece yön değişiyor
    norm_kept = torch.allclose(q_ours.norm(dim=-1), q.norm(dim=-1), atol=ATOL)
    print(f"[{'OK ' if norm_kept else 'HATA'}] RoPE vektör uzunluğu korunuyor")
    return bool(ok_freq and ok_cs and ok_qk and norm_kept)


def check_repeat_kv():
    n_heads = SMALL["num_attention_heads"]
    ok = True
    # 1: Gemma 2B gibi MQA. 2: GQA (n_kv > 1, gruplama sırası burada belli olur). 4: n_rep = 1, hiçbir şey değişmemeli
    for n_kv in (1, 2, 4):
        n_rep = n_heads // n_kv
        x = torch.randn(B, n_kv, T, SMALL["head_dim"])
        ours, theirs = gemma.repeat_kv(x, n_rep), hf_repeat_kv(x, n_rep)
        ok &= compare(f"repeat_kv n_kv={n_kv}, n_rep={n_rep}", ours, theirs, exact=True)

        # Q head'i h, KV head'i h // n_rep'i kullanmalı (gruplar komşu head'lerden oluşuyor)
        grouped = all(torch.equal(ours[:, h], x[:, h // n_rep]) for h in range(n_heads))
        print(f"[{'OK ' if grouped else 'HATA'}] repeat_kv n_kv={n_kv}: Q head h -> KV head h // n_rep")
        ok &= grouped
    return bool(ok)


def check_kv_cache():
    n_kv, hd, n_layers = SMALL["num_key_value_heads"], SMALL["head_dim"], SMALL["num_hidden_layers"]
    ours, theirs = gemma.KVCache(), DynamicCache()
    ok = ours.get_seq_length() == 0
    print(f"[{'OK ' if ok else 'HATA'}] KVCache boşken uzunluk 0")
    # Önce prefill (T token), sonra 3 adım token token üretim. Her adımda bütün katmanlar güncelleniyor
    for step, t_new in enumerate((T, 1, 1, 1)):
        for layer in range(n_layers):
            k, v = torch.randn(B, n_kv, t_new, hd), torch.randn(B, n_kv, t_new, hd)
            ok_k, ok_v = ours.update(k, v, layer)
            hf_k, hf_v = theirs.update(k, v, layer)
            if layer == n_layers - 1:
                ok &= compare(f"KVCache adım {step} (T_yeni={t_new}) K, katman {layer}", ok_k, hf_k, exact=True)
                ok &= compare(f"KVCache adım {step} (T_yeni={t_new}) V, katman {layer}", ok_v, hf_v, exact=True)
            else:
                ok &= torch.equal(ok_k, hf_k) and torch.equal(ok_v, hf_v)
    same_len = ours.get_seq_length() == theirs.get_seq_length() == T + 3
    print(f"[{'OK ' if same_len else 'HATA'}] KVCache uzunluk: bizim {ours.get_seq_length()}, "
          f"HF {theirs.get_seq_length()}, beklenen {T + 3}")
    return bool(ok and same_len)


def check_attention():
    hf_config = HFGemmaConfig(**SMALL, hidden_act="gelu_pytorch_tanh")
    hf_config._attn_implementation = "eager"      # sdpa/flash kernel'leri değil, yukarıda yazılan matematiğin kendisi
    hf = HFGemmaAttention(hf_config, layer_idx=0).eval()
    ours = gemma.GemmaAttention(gemma.GemmaConfig(**SMALL), layer_idx=0).eval()
    if not copy_weights("GemmaAttention", ours, hf):
        return False

    hf_rope = HFGemmaRotaryEmbedding(hf_config)
    x = torch.randn(B, T, SMALL["hidden_size"])
    position_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
    cos, sin = hf_rope(x, position_ids)

    # Nedensel maske: token i sadece j <= i olanlara bakabilir. Toplanan biçim: izinli 0, yasak -inf
    mask = torch.full((T, T), float("-inf")).triu(diagonal=1)[None, None].expand(B, 1, T, T)

    with torch.no_grad():
        ok_mask = compare("GemmaAttention (nedensel maske)",
                          ours(x, (cos, sin), mask), hf(x, (cos, sin), mask)[0])
        ok_nomask = compare("GemmaAttention (maskesiz)", ours(x, (cos, sin)), hf(x, (cos, sin), None)[0])

        # Nedensel maske gerçekten işe yarıyor mu: gelecekteki token'ı değiştirmek geçmiş çıktıyı etkilememeli
        x2 = x.clone()
        x2[:, -1] += 10.0
        out1, out2 = ours(x, (cos, sin), mask), ours(x2, (cos, sin), mask)
        causal = torch.allclose(out1[:, :-1], out2[:, :-1], atol=ATOL) and not torch.allclose(out1[:, -1], out2[:, -1])
        print(f"[{'OK ' if causal else 'HATA'}] GemmaAttention: son token değişince önceki çıktılar aynı kalıyor")

        ok_cache = True
        if hasattr(gemma, "KVCache"):
            # Cache'le: ilk T-3 token prefill (nedensel maske), kalan 3 token tek tek. Tek token'ın maskesi tamamen 0:
            # geçmişin hepsine bakabilir, gelecek zaten cache'te yok. Konumu (RoPE) cache uzunluğu belirliyor
            ours_cache, hf_cache = gemma.KVCache(), DynamicCache()
            P = T - 3
            outs = []
            for start, end in [(0, P)] + [(t, t + 1) for t in range(P, T)]:
                step_mask = mask[:, :, start:end, :end]                 # [B, 1, T_yeni, T_top]
                cs = (cos[:, start:end], sin[:, start:end])
                o = ours(x[:, start:end], cs, step_mask, kv_cache=ours_cache)
                h = hf(x[:, start:end], cs, step_mask, past_key_values=hf_cache)[0]
                ok_cache &= compare(f"GemmaAttention cache'li, token {start}..{end - 1}", o, h)
                outs.append(o)
            # Asıl iddia: cache'li adımları birleştirince tam diziyi tek seferde işlemekle aynı sonuç
            ok_cache &= compare("GemmaAttention cache'li adımlar = cache'siz tam dizi",
                                torch.cat(outs, dim=1), ours(x, (cos, sin), mask))
    return bool(ok_mask and ok_nomask and causal and ok_cache)


def hf_eager_config():
    cfg = HFGemmaConfig(**SMALL, hidden_act="gelu_pytorch_tanh")
    cfg._attn_implementation = "eager"      # sdpa/flash kernel'leri değil, bizim yazdığımız matematiğin kendisi
    return cfg


def check_decoder_layer():
    hf_config = hf_eager_config()
    hf = HFGemmaDecoderLayer(hf_config, layer_idx=0).eval()
    ours = gemma.GemmaDecoderLayer(gemma.GemmaConfig(**SMALL), layer_idx=0).eval()

    # Norm ağırlıklarını rastgele yap. Sıfır kalırsa iki norm aynı davranır ve
    # input_layernorm ile post_attention_layernorm'u karıştıran bir kod da testi geçer
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
    """Bütün RMSNorm ağırlıklarını rastgele yap (sıfır kalırsa hepsi aynı davranır)."""
    with torch.no_grad():
        for n, p in module.named_parameters():
            if "norm" in n:
                p.normal_()


def scaled_embeds(ours_model):
    """HF modeline vereceğimiz, ölçeği zaten uygulanmış embedding'ler.

    Kurulu HF sürümü (transformers 5.x) sqrt(hidden_size) ölçeğini embedding tablosunun içinde
    (GemmaTextScaledWordEmbedding) uyguluyor ve inputs_embeds verilirse ölçeklemiyor. Bizim
    GemmaModel ise ölçeği forward'ın başında, inputs_embeds üzerinde uyguluyor (Umar / eski HF yöntemi).
    Aynı şeyi karşılaştırmak için: bizim modele ham embedding, HF'ye sqrt(d) ile çarpılmış hali veriliyor.
    """
    input_ids = torch.randint(0, SMALL["vocab_size"], (B, T))
    raw = ours_model.embed_tokens(input_ids)                       # [B, T, D], ölçeksiz
    return raw, raw * torch.tensor(SMALL["hidden_size"] ** 0.5, dtype=raw.dtype)


def check_model():
    hf_config = hf_eager_config()
    hf = HFGemmaForCausalLM(hf_config).model.eval()   # GemmaModel kısmı
    ours = gemma.GemmaModel(gemma.GemmaConfig(**SMALL)).eval()
    randomize_norms(hf)
    if not copy_weights("GemmaModel", ours, hf):
        return False

    with torch.no_grad():
        raw, scaled = scaled_embeds(ours)
        ok = compare("GemmaModel", ours(raw), hf(inputs_embeds=scaled).last_hidden_state)

        if hasattr(gemma, "KVCache"):
            # Konum ve maske verilmiyor: ikisi de cache uzunluğundan çıkmalı. HF de aynısını kendi içinde yapıyor
            ours_cache, hf_cache = gemma.KVCache(), DynamicCache()
            P = T - 3
            outs = []
            for start, end in [(0, P)] + [(t, t + 1) for t in range(P, T)]:
                o = ours(raw[:, start:end], kv_cache=ours_cache)
                h = hf(inputs_embeds=scaled[:, start:end], past_key_values=hf_cache, use_cache=True).last_hidden_state
                ok &= compare(f"GemmaModel cache'li, token {start}..{end - 1}", o, h)
                outs.append(o)
            ok &= compare("GemmaModel cache'li adımlar = cache'siz tam dizi", torch.cat(outs, dim=1), ours(raw))

            # Geçmiş varken birden fazla yeni token (past_len > 0, T > 1): maske dikdörtgen olmalı
            rect = gemma.make_causal_mask(3, torch.float32, torch.device("cpu"), past_len=4)[0, 0]
            expected = gemma.make_causal_mask(7, torch.float32, torch.device("cpu"))[0, 0, 4:]
            ok &= compare("make_causal_mask past_len=4, T=3 = kare maskenin son 3 satırı", rect, expected, exact=True)
        return bool(ok)


def check_causal_lm():
    hf_config = hf_eager_config()
    hf = HFGemmaForCausalLM(hf_config).eval()
    ours = gemma.GemmaForCausalLM(gemma.GemmaConfig(**SMALL)).eval()
    ours.tie_weights()   # bizde tying'i PaliGemma çağırıyor, burada elle
    randomize_norms(hf)
    if not copy_weights("GemmaForCausalLM", ours, hf):
        return False

    # load_state_dict değerleri kopyalar ama tying'i bozmaz: iki ad hâlâ aynı Parameter nesnesi olmalı
    tied = ours.lm_head.weight is ours.model.embed_tokens.weight
    print(f"[{'OK ' if tied else 'HATA'}] weight tying: lm_head.weight {'aynı' if tied else 'FARKLI'} "
          f"Parameter nesnesi (embed_tokens.weight)")

    with torch.no_grad():
        raw, scaled = scaled_embeds(ours.model)
        ok = compare("GemmaForCausalLM logit'ler", ours(raw), hf(inputs_embeds=scaled).logits)
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
            print(f"[ -- ] {cls_name}: gemma.py'de henüz yok, atlandı")
            continue
        try:
            results.append(fn())
        except Exception as e:
            print(f"[HATA] {cls_name}: {type(e).__name__}: {e}")
            results.append(False)

    ok = bool(results) and all(results)
    print(f"\nSonuç: {'yazılan parçalar HF ile aynı ✅' if ok else 'fark var ❌'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

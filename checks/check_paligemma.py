"""PaliGemma parçaları doğru çalışıyor mu?

src/paligemma.py'de henüz yazılmamış parçalar atlanır, yani her yeni parçayı yazdıkça
script'i yeniden çalıştırabilirsin:

    make_prefix_lm_mask -> elle yazılmış beklenen desenle
    load_hf_weights, convert_hf_key -> HF ağırlıkları bizim modele eksiksiz ve doğru adla yükleniyor mu
    PaliGemmaForConditionalGeneration -> HF PaliGemmaForConditionalGeneration (küçük rastgele config, logit'ler)
    generate (src/generate.py) -> HF generate(do_sample=False): token token aynı, eos'ta durma dahil

Çalıştır (proje kökünden):
    .venv/bin/python checks/check_paligemma.py
"""
import sys
from pathlib import Path

import torch
from transformers import GemmaConfig as HFGemmaConfig
from transformers import PaliGemmaConfig as HFPaliGemmaConfig
from transformers import PaliGemmaForConditionalGeneration as HFPaliGemma
from transformers import SiglipVisionConfig as HFSiglipVisionConfig

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import generate  # noqa: E402
import paligemma  # noqa: E402


def compare(name, ours, theirs):
    same_shape = ours.shape == theirs.shape
    max_diff = (ours - theirs).abs().max().item() if same_shape else float("nan")
    ok = same_shape and torch.allclose(ours, theirs, atol=ATOL)
    print(f"[{'OK ' if ok else 'HATA'}] {name}: bizim {tuple(ours.shape)}, HF {tuple(theirs.shape)}, max fark {max_diff:.2e}")
    return ok


def check_prefix_lm_mask():
    # 5 token: ilk 3'ü prefix (görüntü + komut), son 2'si suffix (üretilen cevap)
    total_len, prefix_len = 5, 3
    mask = paligemma.make_prefix_lm_mask(total_len, prefix_len, torch.float32, torch.device("cpu"))
    allowed = (mask[0, 0] == 0)

    # 1 = bakabilir. Prefix satırları: prefix'in hepsini görüyor, suffix'i görmüyor.
    # Suffix satırları: prefix'in hepsini + kendinden öncekileri ve kendini görüyor
    expected = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
    ], dtype=torch.bool)

    print("izinli yerler (1 = bakabilir):")
    for row in allowed.int().tolist():
        print("  ", row)

    ok_pattern = torch.equal(allowed, expected)
    print(f"[{'OK ' if ok_pattern else 'HATA'}] prefix-LM maskesi: prefix iki yönlü, suffix nedensel")

    # Yasak yerler en küçük sayı, izinli yerler tam 0 olmalı (attention'da skora toplanıyor)
    ok_values = bool((mask[0, 0][~expected] == torch.finfo(torch.float32).min).all()
                     and (mask[0, 0][expected] == 0).all())
    print(f"[{'OK ' if ok_values else 'HATA'}] maske değerleri: izinli 0, yasak finfo.min")

    # prefix_len = 1 nedensel maskenin aynısı olmalı, prefix_len = total_len tam attention
    from gemma import make_causal_mask
    ok_causal = torch.equal(paligemma.make_prefix_lm_mask(total_len, 1, torch.float32, torch.device("cpu")),
                            make_causal_mask(total_len, torch.float32, torch.device("cpu")))
    full = paligemma.make_prefix_lm_mask(total_len, total_len, torch.float32, torch.device("cpu"))
    ok_full = bool((full == 0).all())
    print(f"[{'OK ' if ok_causal else 'HATA'}] prefix_len=1: nedensel maskeyle aynı")
    print(f"[{'OK ' if ok_full else 'HATA'}] prefix_len=total_len: tam attention (hiçbir yer yasak değil)")
    return ok_pattern and ok_values and ok_causal and ok_full


# Küçük config: gerçek modelde vision 1152/27 katman, text 2048/18 katman. Mantık aynı, yerelde hızlı
VISION = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
              patch_size=14, image_size=28)                      # 28/14 = 2x2 = 4 görüntü token'ı
TEXT = dict(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=1, head_dim=16, pad_token_id=0)
IMAGE_TOKEN = 257
B, N_TEXT = 2, 6
ATOL = 1e-4


def build_models():
    hf_config = HFPaliGemmaConfig(
        vision_config=HFSiglipVisionConfig(**VISION, projection_dim=TEXT["hidden_size"]),
        text_config=HFGemmaConfig(**TEXT, hidden_act="gelu_pytorch_tanh"),
        image_token_index=IMAGE_TOKEN, projection_dim=TEXT["hidden_size"],
        hidden_size=TEXT["hidden_size"], vocab_size=TEXT["vocab_size"],
    )
    hf_config._attn_implementation = "eager"
    hf_config.text_config._attn_implementation = "eager"
    hf_config.vision_config._attn_implementation = "eager"
    hf = HFPaliGemma(hf_config).eval()
    # Başlangıç ağırlıkları çok küçük (std 0.02): logit'ler neredeyse sıfır olur ve yanlış bir mask ya da
    # position testi bile geçer. Hepsini daha büyük rastgele değerlerle dolduruyoruz (norm ağırlıkları dahil)
    with torch.no_grad():
        for p in hf.parameters():
            p.normal_(0, 0.2)

    ours_config = paligemma.PaliGemmaConfig(
        vision_config=paligemma.SiglipVisionConfig(**VISION),
        text_config=paligemma.GemmaConfig(**TEXT),
        image_token_index=IMAGE_TOKEN, projection_dim=TEXT["hidden_size"],
        hidden_size=TEXT["hidden_size"], vocab_size=TEXT["vocab_size"],
    )
    return hf, paligemma.PaliGemmaForConditionalGeneration(ours_config).eval()


def build_loaded_pair():
    """HF modeli + ağırlıkları src/paligemma.py'deki yükleyiciyle kopyalanmış bizim model."""
    hf, ours = build_models()
    paligemma.load_hf_weights(ours, hf.state_dict())
    return hf, ours


def check_loader():
    hf, ours = build_models()
    paligemma.load_hf_weights(ours, hf.state_dict())

    # Her ağırlık HF'dekiyle aynı mı (head.* hariç) ve weight tying kurulu mu
    ours_sd = ours.state_dict()
    same = all(torch.equal(ours_sd[paligemma.convert_hf_key(k)], v)
               for k, v in hf.state_dict().items() if paligemma.convert_hf_key(k) is not None)
    tied = ours.language_model.lm_head.weight is ours.language_model.model.embed_tokens.weight
    print(f"[{'OK ' if same else 'HATA'}] load_hf_weights: bütün ağırlıklar HF ile aynı")
    print(f"[{'OK ' if tied else 'HATA'}] load_hf_weights: lm_head.weight embedding ile aynı Parameter")

    # Eski (4.x) düzendeki adlar da bizim adlara çıkmalı, head atlanmalı
    old_layout = {"vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight": "vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight",
                  "language_model.model.layers.0.self_attn.q_proj.weight": "language_model.model.layers.0.self_attn.q_proj.weight",
                  "language_model.lm_head.weight": "language_model.lm_head.weight",
                  "vision_tower.vision_model.head.probe": None,
                  "model.vision_tower.head.probe": None}
    old_ok = all(paligemma.convert_hf_key(k) == v for k, v in old_layout.items())
    print(f"[{'OK ' if old_ok else 'HATA'}] convert_hf_key: eski düzen aynen geçiyor, head atlanıyor")

    # Eksik bir anahtar sessizce geçmemeli
    broken = {k: v for k, v in hf.state_dict().items() if "post_layernorm" not in k}
    try:
        paligemma.load_hf_weights(ours, broken)
        raised = False
    except KeyError:
        raised = True
    print(f"[{'OK ' if raised else 'HATA'}] load_hf_weights: eksik ağırlıkta hata veriyor")
    return bool(same and tied and old_ok and raised)


def make_inputs():
    # [<image> x4, metin...]. Token id'leri 0'dan (pad) ve görüntü token'ından farklı
    input_ids = torch.cat([torch.full((B, 4), IMAGE_TOKEN), torch.randint(5, 250, (B, N_TEXT))], dim=1)
    pixel_values = torch.randn(B, 3, VISION["image_size"], VISION["image_size"])
    return input_ids, pixel_values, torch.ones_like(input_ids)


def check_model():
    hf, ours = build_loaded_pair()
    if hf is None:
        return False

    input_ids, pixel_values, attention_mask = make_inputs()
    T = input_ids.shape[1]

    ok = True
    with torch.no_grad():
        # 1) Bütün girdi prefix (cevap yok): HF'ye token_type_ids = 0 vererek prefix'i iki yönlü yaptırıyoruz
        hf_logits = hf(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask,
                       token_type_ids=torch.zeros(B, T, dtype=torch.long)).logits
        ok &= compare("PaliGemma logit'ler (hepsi prefix)", ours(input_ids, pixel_values, attention_mask), hf_logits)

        # 2) Son 3 token suffix (cevap): HF'de token_type_ids = 1
        prefix_len = T - 3
        token_type_ids = torch.cat([torch.zeros(B, prefix_len), torch.ones(B, 3)], dim=1).long()
        hf_logits_suffix = hf(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask,
                              token_type_ids=token_type_ids).logits
        ok &= compare("PaliGemma logit'ler (prefix + suffix)",
                      ours(input_ids, pixel_values, attention_mask, prefix_len=prefix_len), hf_logits_suffix)

        # 3) Bulgu: token_type_ids verilmezse HF prefix'i de nedensel yapıyor. Bizim prefix-LM sonucumuz ondan
        # farklı olmalı (testin duyarlı olduğunu da gösteriyor: yanlış mask yanlış sonuç verirdi)
        hf_causal = hf(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask).logits
        gap = (hf_causal - hf_logits).abs().max().item()
        differs = gap > 1e-2
        print(f"[{'OK ' if differs else 'HATA'}] HF token_type_ids'siz (nedensel) ile prefix-LM logit'leri farklı: max fark {gap:.2e}")
        ok &= differs

        # 4) Konum 0'dan da 1'den da başlasa sonuç aynı mı (RoPE yalnızca konum farkına bağlı)
        lm = ours.language_model
        embeds = torch.randn(B, T, TEXT["hidden_size"])
        mask = paligemma.make_prefix_lm_mask(T, prefix_len, embeds.dtype, embeds.device)
        pos0 = torch.arange(T).unsqueeze(0).expand(B, -1)
        ok &= compare("konum 0'dan vs 1'den başlayınca logit'ler",
                      lm(embeds, mask, pos0), lm(embeds, mask, pos0 + 1))
    return bool(ok)


def check_generate():
    hf, ours = build_loaded_pair()
    if hf is None:
        return False
    input_ids, pixel_values, attention_mask = make_inputs()
    T = input_ids.shape[1]
    steps = 8

    def hf_generate(eos):
        # token_type_ids = 0: prefix iki yönlü. HF üretim sırasında yeni token'lara kendisi 1 ekliyor (suffix)
        out = hf.generate(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask,
                          token_type_ids=torch.zeros_like(input_ids), max_new_tokens=steps, do_sample=False,
                          eos_token_id=eos, pad_token_id=TEXT["pad_token_id"])
        return out[:, T:]                                    # HF komutu da geri döndürüyor

    # 1) eos'suz: hepsi max_new_tokens kadar üretiliyor
    ours_tokens = generate.generate(ours, input_ids, pixel_values, attention_mask, steps, eos_token_id=-1)
    hf_tokens = hf_generate(eos=None)
    ok_plain = torch.equal(ours_tokens, hf_tokens)
    print(f"[{'OK ' if ok_plain else 'HATA'}] greedy üretim, eos yok: {ours_tokens[0].tolist()}")
    if not ok_plain:
        print(f"   HF: {hf_tokens[0].tolist()}")

    # 2) eos: ilk örneğin 3. token'ını eos ilan et. O örnek orada durmalı, sonrası pad, diğer örnek devam etmeli
    eos = int(ours_tokens[0, 2])
    ours_eos = generate.generate(ours, input_ids, pixel_values, attention_mask, steps, eos_token_id=eos)
    hf_eos = hf_generate(eos=eos)
    ok_eos = torch.equal(ours_eos, hf_eos)
    print(f"[{'OK ' if ok_eos else 'HATA'}] greedy üretim, eos={eos}: ilk örnek {ours_eos[0].tolist()}")
    if not ok_eos:
        print(f"   HF: {hf_eos[0].tolist()}")
    return bool(ok_plain and ok_eos)


def main():
    torch.manual_seed(0)
    checks = [
        ("make_prefix_lm_mask", check_prefix_lm_mask),
        ("load_hf_weights", check_loader),
        ("PaliGemmaForConditionalGeneration", check_model),
        ("generate", check_generate),
    ]

    results = []
    for name, fn in checks:
        if not hasattr(paligemma, name) and not hasattr(generate, name):
            print(f"[ -- ] {name}: paligemma.py'de henüz yok, atlandı")
            continue
        try:
            results.append(fn())
        except Exception as e:
            print(f"[HATA] {name}: {type(e).__name__}: {e}")
            results.append(False)

    ok = bool(results) and all(results)
    print(f"\nSonuç: {'yazılan parçalar beklenen gibi ✅' if ok else 'fark var ❌'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

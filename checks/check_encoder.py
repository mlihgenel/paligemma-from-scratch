"""SigLIP encoder parçaları HF ile aynı çıktıyı veriyor mu?

Parçalar küçükten büyüğe test edilir. src/siglip.py'de henüz yazılmamış sınıflar atlanır,
yani her yeni sınıfı yazdıkça script'i yeniden çalıştırabilirsin:

    SiglipAttention          -> HF encoder.layers[0].self_attn
    SiglipMLP                -> HF encoder.layers[0].mlp
    SiglipEncoderLayer       -> HF encoder.layers[0]
    SiglipEncoder            -> HF encoder (bütün katmanlar)
    SiglipVisionTransformer  -> HF modelin tamamı (embeddings + encoder + post_layernorm)

Çalıştır (proje kökünden):
    .venv/bin/python checks/check_encoder.py                                    # base: 768 boyut, 12 katman
    .venv/bin/python checks/check_encoder.py google/siglip-so400m-patch14-224  # PaliGemma'nın kodlayıcısı
"""
import dataclasses
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipVisionModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import siglip  # noqa: E402

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "google/siglip-base-patch16-224"
IMAGE = Path.home() / "Desktop/yz50-research/papers/01-paligemma2/experiment/data/images/c00.jpg"
# Toleranslar: katmanlar arttıkça float32 yuvarlama farkları birikir ve encoder'ın derin katmanlarında
# değerler ~200'e kadar büyür. Bu yüzden mutlak farkın yanında göreli farka (rtol) da izin veriyoruz.
ATOL, RTOL = 1e-4, 1e-4


def first(out):
    """HF attention (çıktı, attention ağırlıkları) ikilisi döndürüyor. Bizimki tek tensör de döndürebilir."""
    return out[0] if isinstance(out, tuple) else out


def compare(name, ours, theirs):
    same_shape = ours.shape == theirs.shape
    max_diff = (ours - theirs).abs().max().item() if same_shape else float("nan")
    rel_diff = max_diff / theirs.abs().max().item() if same_shape else float("nan")
    ok = same_shape and torch.allclose(ours, theirs, atol=ATOL, rtol=RTOL)
    print(f"[{'OK ' if ok else 'HATA'}] {name}: bizim {tuple(ours.shape)}, HF {tuple(theirs.shape)}, "
          f"max fark {max_diff:.2e} (göreli {rel_diff:.1e})")
    return ok


def build_config(hf_cfg):
    """Bizim config dataclass'ında hangi alanlar varsa onları HF config'inden doldur.

    Henüz eklemediğin alanlar (ör. num_attention_heads) varsayılan değerinde kalır
    ve bir uyarıyla listelenir.
    """
    ours_fields = {f.name for f in dataclasses.fields(siglip.SiglipVisionConfig)}
    wanted = ["num_channels", "hidden_size", "image_size", "patch_size", "intermediate_size",
              "num_hidden_layers", "num_attention_heads", "layer_norm_eps", "attention_dropout"]
    missing = [k for k in wanted if k not in ours_fields]
    if missing:
        print("UYARI: config'inde henüz olmayan alanlar:", missing)
    return siglip.SiglipVisionConfig(**{k: getattr(hf_cfg, k) for k in wanted if k in ours_fields})


def load(ours, hf_module, name):
    """HF ağırlıklarını bizim modüle kopyala; anahtar uyuşmazlığını okunur şekilde göster."""
    hf_sd = {k: v for k, v in hf_module.state_dict().items() if not k.startswith("head.")}  # PaliGemma head'i kullanmıyor
    ours_keys, hf_keys = set(ours.state_dict()), set(hf_sd)
    if ours_keys != hf_keys:
        print(f"[HATA] {name}: state_dict anahtarları tutmuyor")
        print("   bizde olup HF'de olmayan:", sorted(ours_keys - hf_keys)[:8])
        print("   HF'de olup bizde olmayan:", sorted(hf_keys - ours_keys)[:8])
        return False
    ours.load_state_dict(hf_sd)
    return True


def main():
    torch.manual_seed(0)
    print("Model:", MODEL_ID)
    # attn_implementation="eager": HF attention'ı bizim yazdığımız gibi açık matmul + softmax ile hesaplasın.
    # Varsayılan "sdpa" (PyTorch'un hızlı çekirdeği) matematiksel olarak aynı ama işlem sırası farklı.
    # transformers 5'te yapı: embeddings, encoder, post_layernorm, head
    hf = SiglipVisionModel.from_pretrained(MODEL_ID, attn_implementation="eager").eval()
    config = build_config(hf.config)
    print("Bizim config:", config, "\n")

    # Gerçek görüntünün embedding'leri: encoder parçalarına girdi olarak bunu veriyoruz,
    # böylece rastgele değil, modelin gerçekten göreceği değerlerle test ediyoruz.
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    pixel_values = processor(images=Image.open(IMAGE).convert("RGB"), return_tensors="pt").pixel_values
    with torch.no_grad():
        x = hf.embeddings(pixel_values)  # [1, num_patches, hidden_size]
    hf_layer0 = hf.encoder.layers[0]

    # (bizim sınıf adı, HF karşılığı, testte çalışacak fonksiyonlar)
    tests = [
        ("SiglipAttention", hf_layer0.self_attn,
         lambda m: first(m(x)), lambda h: first(h(x))),
        ("SiglipMLP", hf_layer0.mlp,
         lambda m: m(x), lambda h: h(x)),
        ("SiglipEncoderLayer", hf_layer0,
         lambda m: first(m(x)), lambda h: first(h(x, None))),
        ("SiglipEncoder", hf.encoder,
         lambda m: first(m(x)), lambda h: h(x).last_hidden_state),
        ("SiglipVisionTransformer", hf,
         lambda m: first(m(pixel_values)), lambda h: h(pixel_values).last_hidden_state),
    ]

    results = []
    for cls_name, hf_module, run_ours, run_hf in tests:
        cls = getattr(siglip, cls_name, None)
        if cls is None:
            print(f"[ -- ] {cls_name}: henüz yazılmadı, atlandı")
            continue
        ours = cls(config).eval()
        if not load(ours, hf_module, cls_name):
            results.append(False)
            continue
        with torch.no_grad():
            results.append(compare(cls_name, run_ours(ours), run_hf(hf_module)))

    if results:
        print(f"\nSonuç: {sum(results)}/{len(results)} test geçti", "✅" if all(results) else "❌")


if __name__ == "__main__":
    main()

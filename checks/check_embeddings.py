"""Kendi SiglipVisionEmbeddings'imiz HF ile aynı çıktıyı veriyor mu?

Çalıştır (proje kökünden):
    .venv/bin/python checks/check_embeddings.py

İlk çalıştırmada google/siglip-base-patch16-224 indirilir (~800 MB, giriş gerekmez).
"""
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipVisionModel

# src/ klasörünü import yoluna ekle, böylece `from siglip import ...` çalışır
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from siglip import SiglipVisionConfig, SiglipVisionEmbeddings  # noqa: E402

MODEL_ID = "google/siglip-base-patch16-224"
IMAGE = Path.home() / "Desktop/yz50-research/papers/01-paligemma2/experiment/data/images/c00.jpg"


def compare(name, ours, theirs):
    """Şekli, en büyük mutlak farkı ve allclose sonucunu yazdır."""
    same_shape = ours.shape == theirs.shape
    max_diff = (ours - theirs).abs().max().item() if same_shape else float("nan")
    ok = same_shape and torch.allclose(ours, theirs, atol=1e-5)
    print(f"[{'OK ' if ok else 'HATA'}] {name}: bizim {tuple(ours.shape)}, HF {tuple(theirs.shape)}, max fark {max_diff:.2e}")
    return ok


def main():
    torch.manual_seed(0)

    # 1) HF modelini yükle. Bize sadece içindeki embeddings modülü lazım.
    #    transformers 5'te yapı: SiglipVisionModel -> embeddings, encoder, post_layernorm, head
    #    (4.x sürümlerinde arada bir `vision_model` katmanı vardı; Tunga'nın videosu o sürümde)
    #    Checkpoint görüntü+metin modelinin tamamı olduğu için yüklerken text_model.* ağırlıkları
    #    "UNEXPECTED" diye listelenir. Normal: biz sadece görüntü tarafını alıyoruz.
    hf_model = SiglipVisionModel.from_pretrained(MODEL_ID).eval()
    hf_emb = hf_model.embeddings

    # 2) Kendi modülümüzü HF config'indeki sayılarla kur.
    #    HF config'inden okumak, sayıları elle yazıp yanlış yazma riskini ortadan kaldırır.
    hf_cfg = hf_model.config
    config = SiglipVisionConfig(
        num_channels=hf_cfg.num_channels,
        hidden_size=hf_cfg.hidden_size,
        image_size=hf_cfg.image_size,
        patch_size=hf_cfg.patch_size,
    )
    ours = SiglipVisionEmbeddings(config).eval()

    # 3) Ağırlıkları kopyala. Modül adlarımız HF ile aynıysa anahtarlar birebir tutar.
    #    strict=True (varsayılan): eksik ya da fazla anahtar varsa burada hata verir.
    #    Adı yanlış bir modül (ör. patch_embeddings) bu satırda yakalanır.
    print("HF state_dict anahtarları:", list(hf_emb.state_dict().keys()))
    ours.load_state_dict(hf_emb.state_dict())

    # 4a) Rastgele girdiyle karşılaştır: sadece mimarinin aynı olup olmadığını test eder.
    x = torch.randn(2, 3, config.image_size, config.image_size)
    with torch.no_grad():
        ok1 = compare("rastgele girdi", ours(x), hf_emb(x))

    # 4b) Gerçek görüntüyle karşılaştır. pixel_values HF image processor'dan geliyor
    #     (kendi ön işlememiz src/processing.py'da, check_processing.py ile test ediliyor).
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)  # sadece görüntü tarafı, tokenizer gerekmez
    image = Image.open(IMAGE).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    print(f"pixel_values: şekil {tuple(pixel_values.shape)}, aralık [{pixel_values.min():.2f}, {pixel_values.max():.2f}]")
    with torch.no_grad():
        ok2 = compare(f"gerçek görüntü ({IMAGE.name})", ours(pixel_values), hf_emb(pixel_values))

    print("\nSonuç:", "embeddings HF ile aynı ✅" if ok1 and ok2 else "fark var ❌")


if __name__ == "__main__":
    main()

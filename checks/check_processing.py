"""Kendi process_images ve PaliGemmaProcessor'ımız HF ile aynı pixel_values ve input_ids'i veriyor mu?

Çalıştır (proje kökünden):
    .venv/bin/python checks/check_processing.py

Görüntü tarafı için google/siglip-base-patch16-224'ün preprocessor_config.json dosyası yeterli (giriş gerekmez).
Metin tarafı için google/paligemma-3b-pt-224'ün tokenizer dosyaları iner (~22 MB, gated: `hf auth login` gerekir).
PaliGemma'nın görüntü ön işlemesi de aynı: 224x224, bicubic, mean=std=0.5.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoImageProcessor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import processing  # noqa: E402

MODEL_ID = "google/siglip-base-patch16-224"
IMAGE_DIR = Path.home() / "Desktop/yz50-research/papers/01-paligemma2/experiment/data/images"
IMAGES = ["c00.jpg", "c01.jpg", "c02.jpg"]  # farklı boyutlarda birkaç fotoğraf

# Bizim fonksiyona verilecek ayarlar (HF processor'ın config'indeki değerlerle aynı)
SIZE = (224, 224)  # (yükseklik, genişlik)
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


def main():
    # Fonksiyonun adı process_images (çoğul) olmalı; eski ad varsa onu da kabul et
    fn = getattr(processing, "process_images", None) or getattr(processing, "process_image", None)
    if fn is None:
        print("[HATA] processing.py içinde process_images fonksiyonu bulunamadı")
        sys.exit(1)

    hf = AutoImageProcessor.from_pretrained(MODEL_ID)
    print(f"HF processor: {type(hf).__name__}, size={hf.size}, resample={hf.resample}, "
          f"mean={hf.image_mean}, std={hf.image_std}\n")

    images = [Image.open(IMAGE_DIR / name).convert("RGB") for name in IMAGES]

    ours = fn(
        images=images,
        size=SIZE,
        resample=Image.Resampling.BICUBIC,
        rescale_factor=1 / 255.0,
        image_mean=MEAN,
        image_std=STD,
    )
    ours = np.stack(ours)  # liste -> [B, 3, H, W]
    theirs = hf(images=images, return_tensors="np").pixel_values

    ok_all = True
    for i, name in enumerate(IMAGES):
        o, t = ours[i], theirs[i]
        same_shape = o.shape == t.shape
        max_diff = float(np.abs(o - t).max()) if same_shape else float("nan")
        ok = same_shape and o.dtype == np.float32 and np.allclose(o, t, atol=1e-5)
        ok_all &= ok
        print(f"[{'OK ' if ok else 'HATA'}] {name} ({images[i].size[0]}x{images[i].size[1]}): "
              f"bizim {o.shape} {o.dtype}, aralık [{o.min():.2f}, {o.max():.2f}] | "
              f"HF {t.shape}, max fark {max_diff:.2e}")
        if same_shape and o.dtype != np.float32:
            print(f"   dtype float32 olmalı, bizimki {o.dtype}")

    print(f"\nGörüntü tarafı: {'HF ile aynı ✅' if ok_all else 'fark var ❌'}\n")

    ok_all &= check_processor(images[0])
    print(f"\nSonuç: {'processing HF ile aynı ✅' if ok_all else 'fark var ❌'}")
    sys.exit(0 if ok_all else 1)


def check_processor(image):
    """Metin + görüntü: PaliGemmaProcessor'ımız HF PaliGemmaProcessor ile aynı tensörleri veriyor mu?

    google/paligemma-3b-pt-224 gated bir repo: `hf auth login` gerekir. Sadece tokenizer ve
    config dosyaları iner (~22 MB), model ağırlıkları inmez.
    """
    if not hasattr(processing, "PaliGemmaProcessor"):
        print("[ -- ] PaliGemmaProcessor: henüz yazılmadı, atlandı")
        return True

    from transformers import AutoProcessor, AutoTokenizer

    pg_id = "google/paligemma-3b-pt-224"
    hf_proc = AutoProcessor.from_pretrained(pg_id)
    tokenizer = AutoTokenizer.from_pretrained(pg_id)
    ours = processing.PaliGemmaProcessor(tokenizer, num_image_tokens=256, image_size=224)

    prompts = ["caption tr", "answer tr Tabelada ne yazıyor?", "ocr", "detect kedi"]
    ok_all = True
    for prompt in prompts:
        o = ours(text=[prompt], images=[image])
        # HF'ye <image> token'larını kendimiz vermiyoruz; eksikse kendisi ekliyor (uyarı basar)
        t = hf_proc(text="<image>" + prompt, images=image, return_tensors="pt")

        problems = []
        for key in ["input_ids", "attention_mask", "pixel_values"]:
            if key not in o:
                problems.append(f"{key} dönmüyor")
                continue
            a, b = o[key], t[key]
            if tuple(a.shape) != tuple(b.shape):
                problems.append(f"{key} şekli {tuple(a.shape)}, HF {tuple(b.shape)}")
            elif key == "pixel_values":
                if not np.allclose(a.numpy(), b.numpy(), atol=1e-5):
                    problems.append(f"pixel_values farklı (max fark {float((a - b).abs().max()):.2e})")
            elif not (a == b).all():
                problems.append(f"{key} farklı")

        ok = not problems
        ok_all &= ok
        ids = o["input_ids"][0].tolist() if "input_ids" in o else []
        print(f"[{'OK ' if ok else 'HATA'}] {prompt!r}: {len(ids)} token")
        if ids:
            print(f"   son token'lar (bizim): {tokenizer.convert_ids_to_tokens(ids[254:])}")
        if not ok:
            hf_ids = t["input_ids"][0].tolist()
            print(f"   son token'lar (HF)   : {tokenizer.convert_ids_to_tokens(hf_ids[254:])}")
            for p in problems:
                print(f"   {p}")
    return ok_all


if __name__ == "__main__":
    main()

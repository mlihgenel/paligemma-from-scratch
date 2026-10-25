"""Do our process_images and PaliGemmaProcessor give the same pixel_values and input_ids as HF?

Run (from the project root):
    .venv/bin/python checks/check_processing.py

The image side only needs the preprocessor_config.json of google/siglip-base-patch16-224 (no login).
The text side downloads the tokenizer files of google/paligemma-3b-pt-224 (~22 MB, gated: needs `hf auth login`).
PaliGemma's image preprocessing is the same: 224x224, bicubic, mean=std=0.5.
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
IMAGE = ROOT / "assets/sample_sign.jpg"

# Settings for our function (the same values as in the HF processor config)
SIZE = (224, 224)  # (height, width)
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


def test_images():
    """The sample photo plus two crops with other sizes and aspect ratios, so resizing is tested on more than one shape."""
    image = Image.open(IMAGE).convert("RGB")
    w, h = image.size
    wide = image.crop((0, h // 4, w, h // 4 + w // 2))                 # landscape, 2:1
    small = image.crop((w // 4, h // 4, w // 4 + 150, h // 4 + 150))   # small square, upscaled to 224
    return [("sample", image), ("wide crop", wide), ("small crop", small)]


def main():
    fn = getattr(processing, "process_images", None)
    if fn is None:
        print("[FAIL] no process_images function in processing.py")
        sys.exit(1)

    hf = AutoImageProcessor.from_pretrained(MODEL_ID)
    print(f"HF processor: {type(hf).__name__}, size={hf.size}, resample={hf.resample}, "
          f"mean={hf.image_mean}, std={hf.image_std}\n")

    named = test_images()
    images = [img for _, img in named]

    ours = fn(
        images=images,
        size=SIZE,
        resample=Image.Resampling.BICUBIC,
        rescale_factor=1 / 255.0,
        image_mean=MEAN,
        image_std=STD,
    )
    ours = np.stack(ours)  # list -> [B, 3, H, W]
    theirs = hf(images=images, return_tensors="np").pixel_values

    ok_all = True
    for i, (name, img) in enumerate(named):
        o, t = ours[i], theirs[i]
        same_shape = o.shape == t.shape
        max_diff = float(np.abs(o - t).max()) if same_shape else float("nan")
        ok = same_shape and o.dtype == np.float32 and np.allclose(o, t, atol=1e-5)
        ok_all &= ok
        print(f"[{'OK ' if ok else 'FAIL'}] {name} ({img.size[0]}x{img.size[1]}): "
              f"ours {o.shape} {o.dtype}, range [{o.min():.2f}, {o.max():.2f}] | "
              f"HF {t.shape}, max diff {max_diff:.2e}")
        if same_shape and o.dtype != np.float32:
            print(f"   dtype should be float32, ours is {o.dtype}")

    print(f"\nImage side: {'matches HF ✅' if ok_all else 'mismatch ❌'}\n")

    ok_all &= check_processor(images[0])
    print(f"\nResult: {'processing matches HF ✅' if ok_all else 'mismatch ❌'}")
    sys.exit(0 if ok_all else 1)


def check_processor(image):
    """Text + image: does our PaliGemmaProcessor give the same tensors as HF's PaliGemmaProcessor?

    google/paligemma-3b-pt-224 is a gated repo: needs `hf auth login`. Only the tokenizer and
    config files are downloaded (~22 MB), not the model weights.
    """
    from transformers import AutoProcessor, AutoTokenizer

    pg_id = "google/paligemma-3b-pt-224"
    hf_proc = AutoProcessor.from_pretrained(pg_id)
    tokenizer = AutoTokenizer.from_pretrained(pg_id)
    ours = processing.PaliGemmaProcessor(tokenizer, num_image_tokens=256, image_size=224)

    # Task prefixes the pt model was trained with. The Turkish prompts also exercise non-ASCII tokenization
    prompts = ["caption en", "answer tr Tabelada ne yazıyor?", "ocr", "detect sign"]
    ok_all = True
    for prompt in prompts:
        o = ours(text=[prompt], images=[image])
        t = hf_proc(text="<image>" + prompt, images=image, return_tensors="pt")

        problems = []
        for key in ["input_ids", "attention_mask", "pixel_values"]:
            if key not in o:
                problems.append(f"{key} is not returned")
                continue
            a, b = o[key], t[key]
            if tuple(a.shape) != tuple(b.shape):
                problems.append(f"{key} shape {tuple(a.shape)}, HF {tuple(b.shape)}")
            elif key == "pixel_values":
                if not np.allclose(a.numpy(), b.numpy(), atol=1e-5):
                    problems.append(f"pixel_values differ (max diff {float((a - b).abs().max()):.2e})")
            elif not (a == b).all():
                problems.append(f"{key} differs")

        ok = not problems
        ok_all &= ok
        ids = o["input_ids"][0].tolist() if "input_ids" in o else []
        print(f"[{'OK ' if ok else 'FAIL'}] {prompt!r}: {len(ids)} tokens")
        if ids:
            print(f"   last tokens (ours): {tokenizer.convert_ids_to_tokens(ids[254:])}")
        if not ok:
            hf_ids = t["input_ids"][0].tolist()
            print(f"   last tokens (HF)  : {tokenizer.convert_ids_to_tokens(hf_ids[254:])}")
            for p in problems:
                print(f"   {p}")
    return ok_all


if __name__ == "__main__":
    main()

"""Does our SiglipVisionEmbeddings give the same output as HF?

Run (from the project root):
    .venv/bin/python checks/check_embeddings.py

The first run downloads google/siglip-base-patch16-224 (~800 MB, no login needed).
"""
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipVisionModel

# Add src/ to the import path so that `from siglip import ...` works
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from siglip import SiglipVisionConfig, SiglipVisionEmbeddings  # noqa: E402

MODEL_ID = "google/siglip-base-patch16-224"
IMAGE = ROOT / "assets/sample_sign.jpg"


def compare(name, ours, theirs):
    """Print the shape, the largest absolute difference and the allclose result."""
    same_shape = ours.shape == theirs.shape
    max_diff = (ours - theirs).abs().max().item() if same_shape else float("nan")
    ok = same_shape and torch.allclose(ours, theirs, atol=1e-5)
    print(f"[{'OK ' if ok else 'FAIL'}] {name}: ours {tuple(ours.shape)}, HF {tuple(theirs.shape)}, max diff {max_diff:.2e}")
    return ok


def main():
    torch.manual_seed(0)

    # 1) Load the HF model. We only need its embeddings module.
    #    transformers 5 layout: SiglipVisionModel -> embeddings, encoder, post_layernorm, head
    #    (4.x versions had an extra `vision_model` level in between)
    #    The checkpoint is the full image + text model, so loading lists the text_model.* weights
    #    as "UNEXPECTED". That is expected: we only take the image side.
    hf_model = SiglipVisionModel.from_pretrained(MODEL_ID).eval()
    hf_emb = hf_model.embeddings

    # 2) Build our module with the numbers from the HF config.
    #    Reading them from the config removes the risk of typing a number wrong.
    hf_cfg = hf_model.config
    config = SiglipVisionConfig(
        num_channels=hf_cfg.num_channels,
        hidden_size=hf_cfg.hidden_size,
        image_size=hf_cfg.image_size,
        patch_size=hf_cfg.patch_size,
    )
    ours = SiglipVisionEmbeddings(config).eval()

    # 3) Copy the weights. If our module names match HF, the keys match one to one.
    #    strict=True (the default) raises here on missing or extra keys,
    #    so a misnamed module (e.g. patch_embeddings) is caught on this line.
    print("HF state_dict keys:", list(hf_emb.state_dict().keys()))
    ours.load_state_dict(hf_emb.state_dict())

    # 4a) Compare on random input: this only tests that the architecture is the same.
    x = torch.randn(2, 3, config.image_size, config.image_size)
    with torch.no_grad():
        ok1 = compare("random input", ours(x), hf_emb(x))

    # 4b) Compare on a real image. pixel_values come from the HF image processor
    #     (our own preprocessing is in src/processing.py and tested by check_processing.py).
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)  # image side only, no tokenizer needed
    image = Image.open(IMAGE).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    print(f"pixel_values: shape {tuple(pixel_values.shape)}, range [{pixel_values.min():.2f}, {pixel_values.max():.2f}]")
    with torch.no_grad():
        ok2 = compare(f"real image ({IMAGE.name})", ours(pixel_values), hf_emb(pixel_values))

    print("\nResult:", "embeddings match HF ✅" if ok1 and ok2 else "mismatch ❌")
    sys.exit(0 if ok1 and ok2 else 1)


if __name__ == "__main__":
    main()

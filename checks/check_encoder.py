"""Do the SigLIP encoder parts give the same output as HF?

Parts are tested from small to large:

    SiglipAttention          -> HF encoder.layers[0].self_attn
    SiglipMLP                -> HF encoder.layers[0].mlp
    SiglipEncoderLayer       -> HF encoder.layers[0]
    SiglipEncoder            -> HF encoder (all layers)
    SiglipVisionTransformer  -> the whole HF model (embeddings + encoder + post_layernorm)

Run (from the project root):
    .venv/bin/python checks/check_encoder.py                                    # base: 768 dims, 12 layers
    .venv/bin/python checks/check_encoder.py google/siglip-so400m-patch14-224  # PaliGemma's encoder
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
IMAGE = ROOT / "assets/sample_sign.jpg"
# Tolerances: float32 rounding differences pile up over the layers, and values in the deep
# encoder layers grow to ~200. So besides the absolute difference we also allow a relative one (rtol).
ATOL, RTOL = 1e-4, 1e-4


def first(out):
    """HF attention returns (output, attention weights). Ours may return a single tensor."""
    return out[0] if isinstance(out, tuple) else out


def compare(name, ours, theirs):
    same_shape = ours.shape == theirs.shape
    max_diff = (ours - theirs).abs().max().item() if same_shape else float("nan")
    rel_diff = max_diff / theirs.abs().max().item() if same_shape else float("nan")
    ok = same_shape and torch.allclose(ours, theirs, atol=ATOL, rtol=RTOL)
    print(f"[{'OK ' if ok else 'FAIL'}] {name}: ours {tuple(ours.shape)}, HF {tuple(theirs.shape)}, "
          f"max diff {max_diff:.2e} (relative {rel_diff:.1e})")
    return ok


def build_config(hf_cfg):
    """Fill the fields of our config dataclass from the HF config."""
    ours_fields = {f.name for f in dataclasses.fields(siglip.SiglipVisionConfig)}
    wanted = ["num_channels", "hidden_size", "image_size", "patch_size", "intermediate_size",
              "num_hidden_layers", "num_attention_heads", "layer_norm_eps", "attention_dropout"]
    missing = [k for k in wanted if k not in ours_fields]
    if missing:
        print("WARNING: fields missing from our config:", missing)
    return siglip.SiglipVisionConfig(**{k: getattr(hf_cfg, k) for k in wanted if k in ours_fields})


def load(ours, hf_module, name):
    """Copy the HF weights into our module and show any key mismatch readably."""
    hf_sd = {k: v for k, v in hf_module.state_dict().items() if not k.startswith("head.")}  # PaliGemma does not use the head
    ours_keys, hf_keys = set(ours.state_dict()), set(hf_sd)
    if ours_keys != hf_keys:
        print(f"[FAIL] {name}: state_dict keys do not match")
        print("   in ours but not in HF:", sorted(ours_keys - hf_keys)[:8])
        print("   in HF but not in ours:", sorted(hf_keys - ours_keys)[:8])
        return False
    ours.load_state_dict(hf_sd)
    return True


def main():
    torch.manual_seed(0)
    print("Model:", MODEL_ID)
    # attn_implementation="eager": HF computes attention with an explicit matmul + softmax, like ours.
    # The default "sdpa" (PyTorch's fast kernel) is the same math in a different order.
    # transformers 5 layout: embeddings, encoder, post_layernorm, head
    hf = SiglipVisionModel.from_pretrained(MODEL_ID, attn_implementation="eager").eval()
    config = build_config(hf.config)
    print("Our config:", config, "\n")

    # Embeddings of a real image are the input to the encoder parts,
    # so the test uses values the model actually sees instead of random ones.
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    pixel_values = processor(images=Image.open(IMAGE).convert("RGB"), return_tensors="pt").pixel_values
    with torch.no_grad():
        x = hf.embeddings(pixel_values)  # [1, num_patches, hidden_size]
    hf_layer0 = hf.encoder.layers[0]

    # (our class name, HF counterpart, functions run in the test)
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
        cls = getattr(siglip, cls_name)
        ours = cls(config).eval()
        if not load(ours, hf_module, cls_name):
            results.append(False)
            continue
        with torch.no_grad():
            results.append(compare(cls_name, run_ours(ours), run_hf(hf_module)))

    print(f"\nResult: {sum(results)}/{len(results)} tests passed", "✅" if all(results) else "❌")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()

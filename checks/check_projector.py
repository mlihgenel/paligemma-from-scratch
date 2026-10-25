"""Does our PaliGemmaMultiModalProjector give the same output as HF?

Run (from the project root):
    .venv/bin/python checks/check_projector.py

Nothing is downloaded: the HF projector is built from the PaliGemma 3B config with random weights,
the weights are copied into our module and both are compared on the same input. The test with
the real weights is in the Colab notebook.
"""
import sys
from pathlib import Path

import torch
from transformers import PaliGemmaConfig
from transformers.models.paligemma.modeling_paligemma import PaliGemmaMultiModalProjector as HFProjector

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import paligemma  # noqa: E402
from siglip import SiglipVisionConfig  # noqa: E402

# paligemma-3b-pt-224 values: so400m (1152) -> Gemma 2B (2048), 256 image tokens
VISION_HIDDEN = 1152
PROJECTION_DIM = 2048
NUM_TOKENS = 256


def main():
    torch.manual_seed(0)

    if not hasattr(paligemma, "PaliGemmaMultiModalProjector"):
        print("[FAIL] no PaliGemmaMultiModalProjector class in paligemma.py")
        sys.exit(1)

    hf_config = PaliGemmaConfig(vision_config={"hidden_size": VISION_HIDDEN, "projection_dim": PROJECTION_DIM})
    hf = HFProjector(hf_config).eval()

    config = SiglipVisionConfig(hidden_size=VISION_HIDDEN, projection_dim=PROJECTION_DIM)
    ours = paligemma.PaliGemmaMultiModalProjector(config).eval()

    # Same keys? (in the real checkpoint: multi_modal_projector.linear.weight / .bias)
    ours_keys, hf_keys = set(ours.state_dict()), set(hf.state_dict())
    if ours_keys != hf_keys:
        print("[FAIL] state_dict keys do not match")
        print(f"   in ours but not in HF: {sorted(ours_keys - hf_keys)}")
        print(f"   in HF but not in ours: {sorted(hf_keys - ours_keys)}")
        sys.exit(1)
    ours.load_state_dict(hf.state_dict())

    x = torch.randn(1, NUM_TOKENS, VISION_HIDDEN)  # stands in for the SigLIP output
    with torch.no_grad():
        o, t = ours(x), hf(x)

    same_shape = o.shape == t.shape
    max_diff = (o - t).abs().max().item() if same_shape else float("nan")
    ok = same_shape and torch.allclose(o, t, atol=1e-5)
    print(f"keys: {sorted(ours_keys)}")
    print(f"[{'OK ' if ok else 'FAIL'}] projector: input {tuple(x.shape)} -> ours {tuple(o.shape)}, "
          f"HF {tuple(t.shape)}, max diff {max_diff:.2e}")
    print(f"\nResult: {'projector matches HF ✅' if ok else 'mismatch ❌'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

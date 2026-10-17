"""PaliGemmaMultiModalProjector'ımız HF ile aynı çıktıyı veriyor mu?

Çalıştır (proje kökünden):
    .venv/bin/python checks/check_projector.py

Hiçbir şey indirilmez: HF projeksiyonu PaliGemma 3B config'iyle rastgele ağırlıklarla kurulur,
ağırlıklar bizim modüle kopyalanır ve aynı girdiyle karşılaştırılır. Gerçek ağırlıklarla test
tam model kurulunca (Colab) yapılacak.
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

# paligemma-3b-pt-224'ün değerleri: so400m (1152) -> Gemma 2B (2048), 256 görsel token
VISION_HIDDEN = 1152
PROJECTION_DIM = 2048
NUM_TOKENS = 256


def main():
    torch.manual_seed(0)

    if not hasattr(paligemma, "PaliGemmaMultiModalProjector"):
        print("[HATA] paligemma.py içinde PaliGemmaMultiModalProjector sınıfı bulunamadı")
        sys.exit(1)

    hf_config = PaliGemmaConfig(vision_config={"hidden_size": VISION_HIDDEN, "projection_dim": PROJECTION_DIM})
    hf = HFProjector(hf_config).eval()

    config = SiglipVisionConfig(hidden_size=VISION_HIDDEN, projection_dim=PROJECTION_DIM)
    ours = paligemma.PaliGemmaMultiModalProjector(config).eval()

    # Anahtarlar aynı mı? (gerçek checkpoint'te multi_modal_projector.linear.weight / .bias)
    ours_keys, hf_keys = set(ours.state_dict()), set(hf.state_dict())
    if ours_keys != hf_keys:
        print("[HATA] state_dict anahtarları tutmuyor")
        print(f"   bizde olup HF'de olmayan: {sorted(ours_keys - hf_keys)}")
        print(f"   HF'de olup bizde olmayan: {sorted(hf_keys - ours_keys)}")
        sys.exit(1)
    ours.load_state_dict(hf.state_dict())

    x = torch.randn(1, NUM_TOKENS, VISION_HIDDEN)  # SigLIP çıktısı yerine
    with torch.no_grad():
        o, t = ours(x), hf(x)

    same_shape = o.shape == t.shape
    max_diff = (o - t).abs().max().item() if same_shape else float("nan")
    ok = same_shape and torch.allclose(o, t, atol=1e-5)
    print(f"anahtarlar: {sorted(ours_keys)}")
    print(f"[{'OK ' if ok else 'HATA'}] projeksiyon: girdi {tuple(x.shape)} -> bizim {tuple(o.shape)}, "
          f"HF {tuple(t.shape)}, max fark {max_diff:.2e}")
    print(f"\nSonuç: {'projeksiyon HF ile aynı ✅' if ok else 'fark var ❌'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

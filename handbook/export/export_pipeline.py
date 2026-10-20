"""Görüntü + prompt hattının ara değerleri: senin processing.py ve siglip.py'n ile.

- processing.process_images ve PaliGemmaProcessor (tokenizer: google/paligemma-3b-pt-224, cache'te)
- siglip.SiglipVisionTransformer, google/siglip-base-patch16-224 ağırlıklarıyla (cache'te, 16 px yama,
  196 token). PaliGemma'nın kendi kodlayıcısı so400m (14 px, 256 token); onun konum embedding'i ve
  filtreleri fetch_paligemma_slices.py'den geliyor.

Çalıştır: .venv/bin/python handbook/export/export_pipeline.py
"""
import dataclasses
import shutil

import numpy as np
import torch
from PIL import Image

from common import IMAGE_DIR, OUT, PG_ID, Writer, small_list

import processing  # noqa: E402  (src/)
import siglip      # noqa: E402

IMAGES = ["c02", "c09", "c26", "c19"]
PROMPTS = ["caption tr", "caption en", "ocr", "answer tr Tabelada ne yazıyor?", "detect sign"]
SIGLIP_ID = "google/siglip-base-patch16-224"
ATTN_LAYERS = [0, 5, 11]          # head bazında saklanacak katmanlar


def hist(a, lo, hi, bins=64):
    return [np.histogram(a[..., c], bins=bins, range=(lo, hi))[0].tolist() for c in range(a.shape[-1])]


def load_siglip():
    from transformers import SiglipVisionModel

    hf = SiglipVisionModel.from_pretrained(SIGLIP_ID, attn_implementation="eager").eval()
    fields = {f.name for f in dataclasses.fields(siglip.SiglipVisionConfig)}
    cfg = siglip.SiglipVisionConfig(**{k: getattr(hf.config, k) for k in fields if hasattr(hf.config, k)})
    ours = siglip.SiglipVisionTransformer(cfg).eval()
    sd = {}
    for k, v in hf.state_dict().items():
        k = k.removeprefix("vision_model.")
        if not k.startswith("head."):
            sd[k] = v
    ours.load_state_dict(sd)
    return ours, cfg


def tokenize_prompts(tok, proc):
    """PaliGemmaProcessor (senin) ile prompt'ları diziye çevir. Görüntü token'ları tek blok olarak özetlenir."""
    out = []
    dummy = Image.new("RGB", (224, 224))
    for p in PROMPTS:
        enc = proc(text=[p], images=[dummy])
        ids = enc["input_ids"][0].tolist()
        pieces = tok.convert_ids_to_tokens(ids)
        n_img = sum(1 for i in ids if i == proc.image_token_id)
        out.append({
            "prompt": p,
            "length": len(ids),
            "num_image_tokens": n_img,
            "text_ids": ids[n_img:],                      # <bos> + prompt + \n
            "text_pieces": pieces[n_img:],
            "attention_mask": enc["attention_mask"][0].tolist()[n_img:],
        })
    return out


def main():
    from transformers import AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(PG_ID)
    proc = processing.PaliGemmaProcessor(tok, num_image_tokens=256, image_size=224)
    model, cfg = load_siglip()
    H = cfg.num_attention_heads
    hd = cfg.hidden_size // H
    n_side = cfg.image_size // cfg.patch_size

    w = Writer("pipeline")
    doc = {
        "siglip_id": SIGLIP_ID,
        "siglip_config": dataclasses.asdict(cfg),
        "image_token_id": proc.image_token_id,
        "bos": tok.bos_token, "bos_id": tok.bos_token_id,
        "prompts": tokenize_prompts(tok, proc),
        "attn_layers": ATTN_LAYERS,
        "images": [],
    }

    for name in IMAGES:
        src = IMAGE_DIR / f"{name}.jpg"
        img = Image.open(src).convert("RGB")
        thumb = img.copy()
        thumb.thumbnail((640, 640))
        thumb.save(OUT / f"img_{name}.jpg", quality=88)

        # Senin process_images adımları, ara sonuçlarıyla
        resized = processing.resize(img, (224, 224), resample=Image.Resampling.BICUBIC)
        resized.save(OUT / f"img_{name}_224.png")
        u8 = np.array(resized)                                           # [224, 224, 3] uint8
        rescaled = processing.rescale(u8, 1 / 255.0)                     # [0, 1]
        normalized = processing.normalize(rescaled, [0.5] * 3, [0.5] * 3)  # [-1, 1]
        pv = processing.process_images([img], size=(224, 224), resample=Image.Resampling.BICUBIC,
                                       rescale_factor=1 / 255.0, image_mean=[0.5] * 3, image_std=[0.5] * 3)[0]
        assert np.allclose(pv, normalized.transpose(2, 0, 1))
        assert pv.shape == (3, 224, 224)

        entry = {
            "name": name,
            "orig_size": list(img.size),
            "hist_u8": hist(u8.astype(np.float32), 0, 256),
            "hist_rescaled": hist(rescaled, 0, 1),
            "hist_normalized": hist(normalized, -1, 1),
            "pixel_values": w.arr(pv, "f16"),                            # [3, 224, 224]
        }

        # SigLIP: hook'larla her katmanın girişi, layer_norm1 çıktısı (attention'ı yeniden hesaplamak için)
        x = torch.from_numpy(pv)[None]
        cap = {"ln1": [], "out": []}
        hooks = []
        for layer in model.encoder.layers:
            hooks.append(layer.layer_norm1.register_forward_hook(lambda m, i, o: cap["ln1"].append(o)))
            hooks.append(layer.register_forward_hook(lambda m, i, o: cap["out"].append(o if torch.is_tensor(o) else o[0])))
        with torch.no_grad():
            emb = model.embeddings(x)                                    # [1, 196, 768]
            patch_only = model.embeddings.patch_embedding(x)             # [1, 768, 14, 14]
            last = model(x)                                              # [1, 196, 768]
        for h in hooks:
            h.remove()
        assert patch_only.shape == (1, cfg.hidden_size, n_side, n_side)
        assert last.shape == (1, n_side * n_side, cfg.hidden_size)

        # Senin SiglipAttention'ının içindeki softmax(QK^T * scale): q_proj/k_proj ile yeniden hesap
        attn_mean, attn_heads = [], []
        with torch.no_grad():
            for li, (layer, h_in) in enumerate(zip(model.encoder.layers, cap["ln1"])):
                q = layer.self_attn.q_proj(h_in).view(1, -1, H, hd).transpose(1, 2)
                k = layer.self_attn.k_proj(h_in).view(1, -1, H, hd).transpose(1, 2)
                a = torch.softmax(q @ k.transpose(-2, -1) * hd ** -0.5, dim=-1)[0]   # [H, 196, 196]
                attn_mean.append(a.mean(0).numpy())
                if li in ATTN_LAYERS:
                    attn_heads.append(a.numpy())

        # Yama embedding'lerinin PCA'sı -> RGB (ilk 3 bileşen)
        def pca_rgb(t):
            z = t[0].numpy()
            z = z - z.mean(0)
            _, _, vt = np.linalg.svd(z, full_matrices=False)
            p = z @ vt[:3].T
            p = (p - p.min(0)) / (p.max(0) - p.min(0) + 1e-8)
            return p

        hidden = [emb] + cap["out"]
        entry.update({
            "attn_mean": w.arr(np.stack(attn_mean), "f16"),             # [12, 196, 196]
            "attn_heads": w.arr(np.stack(attn_heads), "f16"),           # [3, 12, 196, 196]
            "pca_embed": small_list(pca_rgb(emb), 3),                    # [196, 3]
            "pca_last": small_list(pca_rgb(last), 3),
            "hidden_norms": small_list(np.stack([h[0].norm(dim=-1).numpy() for h in hidden]), 3),  # [13, 196]
            "hidden_absmax": [float(h.abs().max()) for h in hidden],
            "post_ln_norm_mean": float(last[0].norm(dim=-1).mean()),
            "patch_value_stats": {"mean": float(patch_only.mean()), "std": float(patch_only.std())},
        })
        doc["images"].append(entry)
        print(f"{name}: tamam")

    # Konum embedding'leri (siglip-base) kosinüs benzerliği [196, 196]
    pe = model.embeddings.position_embedding.weight.detach().numpy()
    pe = pe / np.linalg.norm(pe, axis=1, keepdims=True)
    doc["base_pos_cos"] = w.arr(pe @ pe.T, "f16")
    w.save(doc)


if __name__ == "__main__":
    main()

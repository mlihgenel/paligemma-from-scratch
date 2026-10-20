"""Gerçek boyutlu PaliGemma'nın modül ağacı, parametre sayıları, tensör şekilleri ve state_dict anahtarları.

Model `meta` cihazında kuruluyor: parametrelerin şekli var ama bellekte değeri yok, 3B model 0 bayt tutuyor.
Forward da meta tensörlerle çalıştırılıyor; hook'lar her modülün giriş/çıkış şeklini yakalıyor.
Anahtarlar gerçek checkpoint'in (model.safetensors.index.json) anahtarlarıyla karşılaştırılıyor.

Çalıştır: .venv/bin/python handbook/export/export_model_tree.py
"""
import json

import torch
from huggingface_hub import hf_hub_download

from common import PG_ID, Writer

import paligemma  # noqa: E402


def shape_of(x):
    if torch.is_tensor(x):
        return {"shape": list(x.shape), "dtype": str(x.dtype).replace("torch.", "")}
    if isinstance(x, (tuple, list)) and x and torch.is_tensor(x[0]):
        return shape_of(x[0])
    return None


def main():
    config = paligemma.PaliGemmaConfig()
    with torch.device("meta"):
        model = paligemma.PaliGemmaForConditionalGeneration(config)

    shapes = {}
    hooks = []
    for name, mod in model.named_modules():
        def hook(m, args, kwargs, out, name=name):
            first = args[0] if args else next(iter(kwargs.values()), None)
            if name not in shapes:   # 18 katman aynı şekilde: ilk çağrı yeterli
                shapes[name] = {"in": shape_of(first), "out": shape_of(out)}
        hooks.append(mod.register_forward_hook(hook, with_kwargs=True))

    T = 260
    input_ids = torch.full((1, T), 108, dtype=torch.long, device="meta")
    pixel_values = torch.empty(1, 3, 224, 224, device="meta")
    forward_ok, forward_err = True, None
    # model.forward'daki assert (attention_mask == 1) meta tensörde .item() istiyor. Aynı adımları
    # forward'ın sırasıyla, senin metodlarınla çağırıyoruz
    try:
        with torch.no_grad():
            inputs_embeds = model.language_model.get_input_embeddings()(input_ids)
            image_features = model.vision_tower(pixel_values.to(inputs_embeds.dtype))
            image_features = model.multi_modal_projector(image_features)
            inputs_embeds = model._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids)
            model.language_model(inputs_embeds)
    except Exception as e:   # meta cihazda desteklenmeyen bir işlem olursa şekiller kısmi kalır
        forward_ok, forward_err = False, f"{type(e).__name__}: {e}"
    for h in hooks:
        h.remove()

    nodes = []
    for name, mod in model.named_modules():
        own = sum(p.numel() for p in mod.parameters(recurse=False))
        total = sum(p.numel() for p in mod.parameters())   # parameters() aynı nesneyi bir kez sayar (tying)
        nodes.append({
            "name": name or "(kök)",
            "cls": type(mod).__name__,
            "own_params": own,
            "params": total,
            "children": [n for n, _ in mod.named_children()],
            "param_shapes": {n: list(p.shape) for n, p in mod.named_parameters(recurse=False)},
            **shapes.get(name, {}),
        })

    ours_keys = list(model.state_dict().keys())
    ckpt_keys = list(json.load(open(hf_hub_download(PG_ID, "model.safetensors.index.json")))["weight_map"])
    so, sc = set(ours_keys), set(ckpt_keys)

    n_unique = sum(p.numel() for p in model.parameters())
    n_with_dup = sum(v.numel() for v in model.state_dict().values())
    w = Writer("model_tree")
    w.save({
        "config": {
            "vision": vars(config.vision_config),
            "text": vars(config.text_config),
            "image_token_index": config.image_token_index,
            "projection_dim": config.projection_dim,
        },
        "forward_ok": forward_ok,
        "forward_err": forward_err,
        "seq_len": T,
        "nodes": nodes,
        "params_unique": n_unique,
        "params_state_dict": n_with_dup,      # lm_head ve embed_tokens ayrı ayrı sayılırsa
        "tied": model.language_model.lm_head.weight is model.language_model.model.embed_tokens.weight,
        "keys_ours": ours_keys,
        "keys_ckpt": ckpt_keys,
        "keys_only_ours": sorted(so - sc),
        "keys_only_ckpt": sorted(sc - so),
    })
    print(f"forward_ok={forward_ok} {forward_err or ''}")
    print(f"parametre: {n_unique/1e9:.3f}B tekil, state_dict toplamı {n_with_dup/1e9:.3f}B")
    print(f"sadece bizde: {len(so - sc)} {sorted(so - sc)[:3]}  sadece checkpoint'te: {len(sc - so)} {sorted(sc - so)[:3]}")


if __name__ == "__main__":
    main()

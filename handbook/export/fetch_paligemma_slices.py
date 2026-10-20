"""google/paligemma-3b-pt-224 ağırlıklarından sadece küçük parçaları HTTP Range ile çeker.

Model ~6 GB, ama kitap için gereken: norm ağırlıkları, birkaç yüz embedding satırı, SigLIP'in
konum embedding'i ve yama filtreleri, projektör istatistikleri. safetensors dosyası
[8 bayt header uzunluğu][JSON header][ham veri] düzeninde olduğu için her tensörün bayt aralığı
header'dan okunup sadece o aralık indirilebiliyor. Toplam birkaç MB.

Gated repo: `hf auth login` gerekir (tokenizer zaten bu repodan indi, yani erişim var).
"""
import json
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from huggingface_hub import hf_hub_download, hf_hub_url
from huggingface_hub.utils import build_hf_headers

from common import PG_ID, Writer, small_list

HEADERS = build_hf_headers()
_header_cache: dict[str, tuple[int, dict]] = {}


def get_range(fname: str, start: int, end: int) -> bytes:
    """[start, end) bayt aralığını indir."""
    url = hf_hub_url(PG_ID, fname)
    r = requests.get(url, headers={**HEADERS, "Range": f"bytes={start}-{end - 1}"}, timeout=120)
    r.raise_for_status()
    assert len(r.content) == end - start, (fname, start, end, len(r.content))
    return r.content


def header(fname: str):
    if fname not in _header_cache:
        n = struct.unpack("<Q", get_range(fname, 0, 8))[0]
        meta = json.loads(get_range(fname, 8, 8 + n))
        _header_cache[fname] = (8 + n, meta)
    return _header_cache[fname]


def bf16_to_f32(buf: bytes) -> np.ndarray:
    u16 = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32)
    return (u16 << 16).view(np.float32)


def tensor_info(name: str, weight_map: dict):
    fname = weight_map[name]
    base, meta = header(fname)
    t = meta[name]
    s, e = t["data_offsets"]
    return fname, base + s, base + e, t["shape"], t["dtype"]


def fetch(name: str, weight_map: dict) -> np.ndarray:
    fname, s, e, shape, dtype = tensor_info(name, weight_map)
    assert dtype in ("BF16", "F32"), dtype
    buf = get_range(fname, s, e)
    a = bf16_to_f32(buf) if dtype == "BF16" else np.frombuffer(buf, dtype=np.float32)
    return a.reshape(shape), dtype


def fetch_rows(name: str, weight_map: dict, rows: list[int]) -> np.ndarray:
    """Büyük bir 2D tensörden sadece seçilen satırları indir (embedding tablosu)."""
    fname, s, _, shape, dtype = tensor_info(name, weight_map)
    width = shape[1]
    bpe = 2 if dtype == "BF16" else 4
    row_bytes = width * bpe

    def one(r):
        buf = get_range(fname, s + r * row_bytes, s + (r + 1) * row_bytes)
        return bf16_to_f32(buf) if dtype == "BF16" else np.frombuffer(buf, dtype=np.float32)

    with ThreadPoolExecutor(16) as ex:
        return np.stack(list(ex.map(one, rows))), shape, dtype


def main():
    from transformers import AutoTokenizer

    weight_map = json.load(open(hf_hub_download(PG_ID, "model.safetensors.index.json")))["weight_map"]
    tok = AutoTokenizer.from_pretrained(PG_ID)
    w = Writer("pg_slices")
    doc = {"model": PG_ID}

    # Checkpoint'te lm_head yok: weight tying'in kanıtı
    doc["has_lm_head_key"] = any("lm_head" in k for k in weight_map)
    doc["num_keys"] = len(weight_map)

    # 1) Dil modelinin bütün RMSNorm ağırlıkları: 18 katman x (input, post_attention) + son norm
    n_layers = 18
    norms = {"input_layernorm": [], "post_attention_layernorm": []}
    for i in range(n_layers):
        for kind in norms:
            a, dtype = fetch(f"language_model.model.layers.{i}.{kind}.weight", weight_map)
            norms[kind].append(a)
    final, _ = fetch("language_model.model.norm.weight", weight_map)
    doc["norm_dtype"] = dtype
    doc["norms"] = {
        "input_layernorm": w.arr(np.stack(norms["input_layernorm"])),           # [18, 2048]
        "post_attention_layernorm": w.arr(np.stack(norms["post_attention_layernorm"])),
        "final": w.arr(final),                                                   # [2048]
    }
    print("norm ağırlıkları alındı")

    # 2) Embedding satırları: örnek token'lar + rastgele bir örneklem (norm dağılımı için)
    words = ["<bos>", "<pad>", "<eos>", "<image>", "\n", "caption", "answer", "ocr", "detect", "segment",
             " en", " tr", " kedi", " cat", " dog", " köpek", " tabela", " sign", " acil", " çıkış",
             " İstanbul", " the", " a", ",", ".", "<loc0000>", "<loc0512>", "<loc1023>", "<seg000>", "<seg127>"]
    named = []
    for wd in words:
        ids = tok.convert_tokens_to_ids(wd) if wd.startswith("<") or wd == "\n" else None
        if ids is None or ids == tok.unk_token_id:
            enc = tok(wd, add_special_tokens=False)["input_ids"]
            ids = enc[0] if enc else None
        if ids is not None:
            named.append((wd, int(ids), tok.convert_ids_to_tokens(int(ids))))
    rng = np.random.default_rng(0)
    vocab = tok.vocab_size if hasattr(tok, "vocab_size") else 257216
    sample_ids = sorted(set(rng.choice(256000, 400, replace=False).tolist()))
    all_ids = [i for _, i, _ in named] + sample_ids
    rows, shape, dtype = fetch_rows("language_model.model.embed_tokens.weight", weight_map, all_ids)
    doc["embed_shape"] = shape
    doc["embed_dtype"] = dtype
    doc["embed_named"] = [{"text": t, "id": i, "piece": p} for t, i, p in named]
    doc["embed_named_rows"] = w.arr(rows[: len(named)], "f32")     # [n, 2048]
    doc["embed_sample_ids"] = sample_ids
    doc["embed_sample_rows"] = w.arr(rows[len(named):], "f16")     # [400, 2048]
    print(f"{len(all_ids)} embedding satırı alındı")

    # 3) SigLIP konum embedding'i [256, 1152] ve yama filtreleri [1152, 3, 14, 14]
    pos, _ = fetch("vision_tower.vision_model.embeddings.position_embedding.weight", weight_map)
    pn = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    doc["pos_shape"] = list(pos.shape)
    doc["pos_cos"] = w.arr(pn @ pn.T, "f16")                                    # [256, 256]
    patch_w, _ = fetch("vision_tower.vision_model.embeddings.patch_embedding.weight", weight_map)
    doc["patch_w_shape"] = list(patch_w.shape)
    # En büyük normlu 64 filtre, görselleştirme için
    order = np.argsort(-np.linalg.norm(patch_w.reshape(len(patch_w), -1), axis=1))[:64]
    doc["patch_filters_idx"] = order.tolist()
    doc["patch_filters"] = w.arr(patch_w[order], "f32")                        # [64, 3, 14, 14]
    pb, _ = fetch("vision_tower.vision_model.embeddings.patch_embedding.bias", weight_map)
    doc["patch_bias_first"] = small_list(pb[:8])
    print("SigLIP embedding parçaları alındı")

    # 4) Projektör: ağırlık [2048, 1152] sadece istatistik, bias tam
    pw, _ = fetch("multi_modal_projector.linear.weight", weight_map)
    pbias, _ = fetch("multi_modal_projector.linear.bias", weight_map)
    doc["proj_shape"] = list(pw.shape)
    doc["proj_stats"] = {"mean": float(pw.mean()), "std": float(pw.std()), "absmax": float(np.abs(pw).max())}
    hist, edges = np.histogram(pw, bins=80)
    doc["proj_hist"] = {"counts": hist.tolist(), "edges": small_list(edges)}
    doc["proj_bias"] = w.arr(pbias)
    print("projektör alındı")

    w.save(doc)


if __name__ == "__main__":
    main()

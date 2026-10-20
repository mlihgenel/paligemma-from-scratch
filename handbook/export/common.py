"""Export scriptlerinin ortak yardımcıları.

Bütün scriptler proje kökünden .venv ile çalışır:
    .venv/bin/python handbook/export/<script>.py

Çıktı: handbook/public/data/<ad>.json (+ büyük diziler için <ad>__<dizi>.bin).
JSON içinde bir dizi {"$bin": dosya, "shape": [...], "dtype": "f32"|"f16"|"u8"} olarak durur,
kitaptaki lib/data.ts bunu otomatik Float32Array'e çevirir.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]          # proje kökü (paligemma2/)
SRC = ROOT / "src"
OUT = ROOT / "handbook" / "public" / "data"
OUT.mkdir(parents=True, exist_ok=True)
IMAGE_DIR = Path.home() / "Desktop/yz50-research/papers/01-paligemma2/experiment/data/images"
PG_ID = "google/paligemma-3b-pt-224"

sys.path.insert(0, str(SRC))   # checks/ ile aynı: senin modüllerin import ediliyor


class Writer:
    """Bir JSON belgesi ve ona bağlı .bin dosyalarını yazar."""

    def __init__(self, name: str):
        self.name = name
        self.n = 0

    def arr(self, a, dtype: str = "f32"):
        a = np.asarray(a)
        np_dtype = {"f32": np.float32, "f16": np.float16, "u8": np.uint8, "i32": np.int32}[dtype]
        fname = f"{self.name}__{self.n}.bin"
        self.n += 1
        np.ascontiguousarray(a.astype(np_dtype)).tofile(OUT / fname)
        return {"$bin": fname, "shape": list(a.shape), "dtype": dtype}

    def save(self, doc: dict):
        path = OUT / f"{self.name}.json"
        path.write_text(json.dumps(doc, ensure_ascii=False, default=_default))
        print(f"yazıldı: {path.relative_to(ROOT)} (+{self.n} bin)")


def small_list(a, nd: int = 5):
    """Küçük dizileri JSON'a okunur sayı listesi olarak koy."""
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))

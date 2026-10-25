"""Shared helpers for the export scripts.

All scripts run from the project root with the .venv:
    .venv/bin/python handbook/export/<script>.py

Output: handbook/public/data/<name>.json (+ <name>__<array>.bin for large arrays).
Inside the JSON an array is stored as {"$bin": file, "shape": [...], "dtype": "f32"|"f16"|"u8"},
and the handbook's lib/data.ts turns it into a Float32Array automatically.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]          # project root
SRC = ROOT / "src"
OUT = ROOT / "handbook" / "public" / "data"
OUT.mkdir(parents=True, exist_ok=True)
IMAGE_DIR = Path(__file__).resolve().parent / "images"     # example photos (credits in images/CREDITS.md)
PG_ID = "google/paligemma-3b-pt-224"

sys.path.insert(0, str(SRC))   # same as checks/: the project's own modules are imported


class Writer:
    """Writes a JSON document and the .bin files it points to."""

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
        print(f"wrote: {path.relative_to(ROOT)} (+{self.n} bin)")


def small_list(a, nd: int = 5):
    """Put small arrays into the JSON as a readable list of numbers."""
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))

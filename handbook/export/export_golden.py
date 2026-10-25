"""Golden values for lib/tensor.ts and lib/bf16.ts (from PyTorch).

The handbook's interactive math runs in JS; vitest compares it with the PyTorch results produced here.
RMSNorm is computed here with the project's GemmaRMSNorm.

Run: .venv/bin/python handbook/export/export_golden.py
"""
import json

import torch
import torch.nn.functional as F

from common import ROOT

import gemma  # noqa: E402

OUT = ROOT / "handbook" / "src" / "tests" / "golden.json"


def L(t):
    return t.detach().float().flatten().tolist()


def main():
    torch.manual_seed(0)
    x = torch.randn(3, 8) * 5
    w = torch.randn(8) * 0.5
    norm = gemma.GemmaRMSNorm(8, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(w)
        rms = norm(x)
        ln = F.layer_norm(x, (8,), eps=1e-6)

    # bf16: both the rounding and the bf16 result of Gemma's RMSNorm
    vals = torch.cat([torch.randn(200) * 50, torch.tensor([2048 ** 0.5, 1 / 3, 0.1, 65504.0, 1e-8, -0.0, 3.0e38])])
    xb = (torch.randn(2, 16) * 40).to(torch.bfloat16)
    wb = torch.randn(16)
    nb = gemma.GemmaRMSNorm(16)
    with torch.no_grad():
        nb.weight.copy_(wb)
        nb = nb.to(torch.bfloat16)
        rms_bf16 = nb(xb)

    g = torch.linspace(-6, 6, 49)
    doc = {
        "x": L(x), "x_shape": [3, 8], "w": L(w),
        "rmsnorm": L(rms), "layernorm": L(ln),
        "gelu_tanh": L(F.gelu(g, approximate="tanh")), "gelu_erf": L(F.gelu(g)), "silu": L(F.silu(g)), "grid": L(g),
        "softmax": L(torch.softmax(x, dim=-1)),
        "bf16_in": L(vals), "bf16_out": L(vals.to(torch.bfloat16)),
        # nb.to(bfloat16) also converts the weight to bf16: the converted values go to the JS side
        "rms_bf16_x": L(xb), "rms_bf16_w": L(nb.weight), "rms_bf16_out": L(rms_bf16),
    }
    OUT.write_text(json.dumps(doc))
    print("wrote:", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()

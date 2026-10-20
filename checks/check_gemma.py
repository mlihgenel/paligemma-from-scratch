"""Gemma parçaları HF ile aynı çıktıyı veriyor mu?

Parçalar küçükten büyüğe test edilir. src/gemma.py'de henüz yazılmamış sınıflar atlanır,
yani her yeni sınıfı yazdıkça script'i yeniden çalıştırabilirsin:

    GemmaRMSNorm       -> HF GemmaRMSNorm (float32 ve bfloat16 girdiyle)
    GemmaMLP           -> HF GemmaMLP
    GemmaDecoderLayer  -> HF GemmaDecoderLayer (attention yerine iki tarafta da aynı geçici modül)

Çalıştır (proje kökünden):
    .venv/bin/python checks/check_gemma.py

Hiçbir şey indirilmez: HF modülleri küçük bir config ve rastgele ağırlıklarla kurulur, ağırlıklar
bizim modüllere kopyalanır ve aynı girdiyle karşılaştırılır. Gerçek ağırlıklarla test tam model
kurulunca (Colab) yapılacak.
"""
import sys
from pathlib import Path

import torch
from transformers import GemmaConfig as HFGemmaConfig
from torch import nn
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer as HFGemmaDecoderLayer
from transformers.models.gemma.modeling_gemma import GemmaMLP as HFGemmaMLP
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm as HFGemmaRMSNorm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import gemma  # noqa: E402

# Küçük config: gerçek Gemma 2B'de hidden 2048, intermediate 16384. Mantık aynı, yerelde hızlı
SMALL = dict(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=1,
    head_dim=16,
)
EPS = 1e-6
B, T = 2, 10
ATOL = 1e-5


def compare(name, ours, theirs, exact=False):
    same_shape = ours.shape == theirs.shape
    same_dtype = ours.dtype == theirs.dtype
    max_diff = (ours.float() - theirs.float()).abs().max().item() if same_shape else float("nan")
    if exact:
        ok = same_shape and same_dtype and torch.equal(ours, theirs)
    else:
        ok = same_shape and same_dtype and torch.allclose(ours, theirs, atol=ATOL)
    print(f"[{'OK ' if ok else 'HATA'}] {name}: bizim {tuple(ours.shape)} {ours.dtype}, "
          f"HF {tuple(theirs.shape)} {theirs.dtype}, max fark {max_diff:.2e}")
    return ok


def copy_weights(name, ours, hf):
    """Anahtarları karşılaştır, tutuyorsa HF ağırlıklarını bizim modüle kopyala."""
    ours_keys, hf_keys = set(ours.state_dict()), set(hf.state_dict())
    if ours_keys != hf_keys:
        print(f"[HATA] {name}: state_dict anahtarları tutmuyor")
        print(f"   bizde olup HF'de olmayan: {sorted(ours_keys - hf_keys)}")
        print(f"   HF'de olup bizde olmayan: {sorted(hf_keys - ours_keys)}")
        return False
    ours.load_state_dict(hf.state_dict())
    return True


def check_rmsnorm():
    hf = HFGemmaRMSNorm(SMALL["hidden_size"], eps=EPS).eval()
    # Ağırlık sıfırla başlıyor. Sıfır kalırsa ölçek her iki tarafta da 1 olur ve weight'i yanlış
    # kullanan (ya da hiç kullanmayan) bir kod da testi geçer. Bu yüzden rastgele dolduruyoruz.
    with torch.no_grad():
        hf.weight.normal_()
    ours = gemma.GemmaRMSNorm(SMALL["hidden_size"], eps=EPS).eval()
    if not copy_weights("GemmaRMSNorm", ours, hf):
        return False

    # Gemma'da embedding'ler sqrt(hidden_size) ile çarpıldığı için girdiler büyük olabiliyor
    x = torch.randn(B, T, SMALL["hidden_size"]) * 40
    with torch.no_grad():
        ok32 = compare("GemmaRMSNorm float32", ours(x), hf(x))

        # Model bf16'da çalışacak. İşlem sırası HF ile aynıysa sonuç bit bit aynı olmalı.
        # Llama sırasıyla (önce bf16'ya dön, sonra çarp) yazılmışsa burada küçük bir fark çıkar.
        ours.to(torch.bfloat16), hf.to(torch.bfloat16)
        x16 = x.to(torch.bfloat16)
        ok16 = compare("GemmaRMSNorm bfloat16 (birebir)", ours(x16), hf(x16), exact=True)
    return ok32 and ok16


def check_mlp():
    hf_config = HFGemmaConfig(**SMALL, hidden_act="gelu_pytorch_tanh")
    hf = HFGemmaMLP(hf_config).eval()
    ours = gemma.GemmaMLP(gemma.GemmaConfig(**SMALL)).eval()
    if not copy_weights("GemmaMLP", ours, hf):
        return False

    x = torch.randn(B, T, SMALL["hidden_size"])
    with torch.no_grad():
        return compare("GemmaMLP", ours(x), hf(x))


class StubAttention(nn.Module):
    def __init__(self, hidden_size, hf_style):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hf_style = hf_style

    def forward(self, hidden_states, **kwargs):
        out = self.proj(hidden_states)
        return (out, None) if self.hf_style else out


def check_decoder_layer():
    hf_config = HFGemmaConfig(**SMALL, hidden_act="gelu_pytorch_tanh")
    hf = HFGemmaDecoderLayer(hf_config, layer_idx=0).eval()
    ours = gemma.GemmaDecoderLayer(gemma.GemmaConfig(**SMALL), layer_idx=0).eval()

    # Stub'ı self_attn adına koyuyoruz. Bizim katmanda attention başka bir adla duruyorsa stub hiç
    # çağrılmaz ve sebebi belirsiz bir sayısal fark çıkar. Önce adı kontrol et
    if not hasattr(ours, "self_attn"):
        others = [n for n, _ in ours.named_children() if n not in ("mlp", "input_layernorm", "post_attention_layernorm")]
        print(f"[HATA] GemmaDecoderLayer: 'self_attn' adında bir alt modül yok (bulunan: {others}). "
              f"HF anahtarları layers.N.self_attn.q_proj.weight şeklinde")
        return False

    # Gerçek attention'ı iki tarafta da stub ile değiştir (bizimki gerçek GemmaAttention olsa bile,
    # çünkü burada test edilen şey katmanın iskeleti. Attention'ın kendi testi aşama 5'te eklenecek)
    hf.self_attn = StubAttention(SMALL["hidden_size"], hf_style=True)
    ours.self_attn = StubAttention(SMALL["hidden_size"], hf_style=False)

    # Norm ağırlıklarını rastgele yap. Sıfır kalırsa iki norm aynı davranır ve
    # input_layernorm ile post_attention_layernorm'u karıştıran bir kod da testi geçer
    with torch.no_grad():
        hf.input_layernorm.weight.normal_()
        hf.post_attention_layernorm.weight.normal_()
    if not copy_weights("GemmaDecoderLayer", ours, hf):
        return False

    x = torch.randn(B, T, SMALL["hidden_size"]) * 40
    with torch.no_grad():
        return compare("GemmaDecoderLayer (stub attention)", ours(x), hf(x))


def main():
    torch.manual_seed(0)
    checks = [
        ("GemmaRMSNorm", check_rmsnorm),
        ("GemmaMLP", check_mlp),
        ("GemmaDecoderLayer", check_decoder_layer),
    ]

    results = []
    for cls_name, fn in checks:
        if not hasattr(gemma, cls_name):
            print(f"[ -- ] {cls_name}: gemma.py'de henüz yok, atlandı")
            continue
        try:
            results.append(fn())
        except Exception as e:
            print(f"[HATA] {cls_name}: {type(e).__name__}: {e}")
            results.append(False)

    ok = bool(results) and all(results)
    print(f"\nSonuç: {'yazılan parçalar HF ile aynı ✅' if ok else 'fark var ❌'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

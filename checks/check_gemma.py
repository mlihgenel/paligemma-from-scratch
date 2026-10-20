"""Gemma parçaları HF ile aynı çıktıyı veriyor mu?

Parçalar küçükten büyüğe test edilir. src/gemma.py'de henüz yazılmamış sınıflar atlanır,
yani her yeni sınıfı yazdıkça script'i yeniden çalıştırabilirsin:

    GemmaRMSNorm  -> HF GemmaRMSNorm (float32 ve bfloat16 girdiyle)
    GemmaMLP      -> HF GemmaMLP

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


def main():
    torch.manual_seed(0)
    checks = [("GemmaRMSNorm", check_rmsnorm), ("GemmaMLP", check_mlp)]

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

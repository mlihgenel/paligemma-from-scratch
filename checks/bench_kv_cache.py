"""KV cache ne kadar hızlandırıyor? Cache'li ve cache'siz üretimde token başına süre.

Model gerçek PaliGemma 3B'nin boyutlarında (SigLIP 1152, Gemma 2048, 8 Q head + 1 KV head, head_dim 256,
256 görüntü token'ı) ama 8 GB M1'e sığsın diye daha az katmanlı ve küçük sözlüklü, ağırlıklar rastgele.
Süre ağırlıkların değerine bağlı değil, sadece şekillere bağlı. Üretilen token'lar anlamsız, bakılmıyor.

İki ölçüm:
    1) Tek adımın süresi, dizi uzunluğu L iken:
       cache yok -> model(bütün L token + görüntü)   cache -> model(1 token, L uzunluğunda cache)
    2) generate ile uçtan uca N token, iki yolda da

Çalıştır (proje kökünden):
    .venv/bin/python checks/bench_kv_cache.py
    .venv/bin/python checks/bench_kv_cache.py --device cpu --text-layers 2
"""
import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import generate  # noqa: E402
import paligemma  # noqa: E402

VOCAB = 32768
IMAGE_TOKEN = VOCAB - 1
N_PROMPT = 10                     # <bos> + komut
N_IMAGE = 256                     # 224/14 = 16x16


def build_model(vision_layers, text_layers, device):
    config = paligemma.PaliGemmaConfig(
        vision_config=paligemma.SiglipVisionConfig(hidden_size=1152, intermediate_size=4304,
                                                   num_hidden_layers=vision_layers, num_attention_heads=16,
                                                   patch_size=14),
        text_config=paligemma.GemmaConfig(vocab_size=VOCAB, hidden_size=2048, intermediate_size=16384,
                                          num_hidden_layers=text_layers, num_attention_heads=8,
                                          num_key_value_heads=1, pad_token_id=0),
        image_token_index=IMAGE_TOKEN, vocab_size=VOCAB,
    )
    return paligemma.PaliGemmaForConditionalGeneration(config).eval().to(device)


def sync(device):
    # GPU'da (mps/cuda) çağrılar asenkron: senkronize etmeden ölçülen süre sadece işi kuyruğa koyma süresi olur
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, device, repeat=3):
    """fn'yi bir kez ısınma için, sonra repeat kez çalıştırır, en kısa süreyi (saniye) döndürür."""
    fn()
    sync(device)
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        sync(device)
        best = min(best, time.perf_counter() - start)
    return best


@torch.no_grad()
def bench_step(model, input_ids, pixel_values, device, n_answer):
    """Prefix + n_answer cevap token'ı varken bir sonraki adımın süresi, iki yolda."""
    prefix_len = input_ids.shape[1]
    answer = torch.randint(1, IMAGE_TOKEN, (1, n_answer), device=device)
    full_ids = torch.cat([input_ids, answer], dim=1)                    # [1, L]
    L = full_ids.shape[1]

    # Cache yok: aşama 6'daki gibi bütün dizi ve SigLIP baştan
    no_cache = timed(lambda: model(full_ids, pixel_values, torch.ones_like(full_ids), prefix_len=prefix_len), device)

    # Cache: önce L - 1 token'lık cache'i kur (ölçüme dahil değil), sonra sadece son token'ı ver.
    # Her ölçümde cache yeniden kuruluyor, çünkü adım cache'e bir token ekliyor
    def cached_step():
        cache = paligemma.KVCache()
        model(input_ids, pixel_values, torch.ones_like(input_ids), kv_cache=cache)
        if L - 1 > prefix_len:
            rest = full_ids[:, prefix_len:L - 1]
            model(rest, None, torch.ones_like(rest), kv_cache=cache)
        return cache

    best = float("inf")
    for i in range(4):                                                   # ilki ısınma
        cache = cached_step()
        sync(device)
        start = time.perf_counter()
        model(full_ids[:, -1:], None, torch.ones(1, 1, dtype=torch.long, device=device), kv_cache=cache)
        sync(device)
        if i > 0:
            best = min(best, time.perf_counter() - start)
    return L, no_cache, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--vision-layers", type=int, default=6, help="gerçek modelde 27")
    parser.add_argument("--text-layers", type=int, default=4, help="gerçek modelde 18")
    parser.add_argument("--new-tokens", type=int, default=32)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)

    model = build_model(args.vision_layers, args.text_layers, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"cihaz {device}, SigLIP {args.vision_layers} katman, Gemma {args.text_layers} katman, {n_params:.0f}M parametre")

    input_ids = torch.cat([torch.full((1, N_IMAGE), IMAGE_TOKEN), torch.randint(1, IMAGE_TOKEN, (1, N_PROMPT))], dim=1)
    input_ids = input_ids.to(device)                                     # [1, 266]
    pixel_values = torch.randn(1, 3, 224, 224, device=device)

    print("\n1) Tek adım (ms), dizi uzunluğu L = 266 prefix + cevap")
    print(f"{'cevap':>6} {'L':>5} {'cache yok':>10} {'cache':>8} {'hızlanma':>9}")
    for n_answer in (1, 16, 64, 128):
        L, no_cache, cached = bench_step(model, input_ids, pixel_values, device, n_answer)
        print(f"{n_answer:>6} {L:>5} {no_cache * 1e3:>10.1f} {cached * 1e3:>8.1f} {no_cache / cached:>8.1f}x")

    print(f"\n2) generate ile {args.new_tokens} token (eos yok)")
    results = {}
    for use_cache in (False, True):
        run = lambda: generate.generate(model, input_ids, pixel_values, torch.ones_like(input_ids),
                                        args.new_tokens, eos_token_id=-1, use_cache=use_cache)
        results[use_cache] = (timed(run, device, repeat=1), run())
    for use_cache, (seconds, _) in results.items():
        print(f"   {'cache' if use_cache else 'cache yok':>9}: {seconds:.2f} s toplam, "
              f"{seconds / args.new_tokens * 1e3:.1f} ms/token")
    same = torch.equal(results[True][1], results[False][1])
    print(f"   hızlanma {results[False][0] / results[True][0]:.1f}x, "
          f"token'lar {'aynı' if same else 'FARKLI'}")


if __name__ == "__main__":
    main()

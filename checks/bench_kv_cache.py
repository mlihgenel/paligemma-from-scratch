"""How much does the KV cache speed things up? Time per token with and without the cache.

The model has the dimensions of the real PaliGemma 3B (SigLIP 1152, Gemma 2048, 8 Q heads + 1 KV head,
head_dim 256, 256 image tokens) but fewer layers and a smaller vocabulary so it fits in 8 GB, with random
weights. Timing depends only on the shapes, not on the weight values. The generated tokens are meaningless.

Two measurements:
    1) Time of a single step at sequence length L:
       no cache -> model(all L tokens + image)   cache -> model(1 token, cache of length L)
    2) N tokens end to end with generate, both ways

Run (from the project root):
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
    # On a GPU (mps/cuda) calls are asynchronous: without a sync you only time how long it takes to queue the work
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, device, repeat=3):
    """Runs fn once to warm up, then repeat times, and returns the shortest time (seconds)."""
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
    """Time of the next step with the prefix + n_answer answer tokens, both ways."""
    prefix_len = input_ids.shape[1]
    answer = torch.randint(1, IMAGE_TOKEN, (1, n_answer), device=device)
    full_ids = torch.cat([input_ids, answer], dim=1)                    # [1, L]
    L = full_ids.shape[1]

    # No cache: the whole sequence and SigLIP from scratch
    no_cache = timed(lambda: model(full_ids, pixel_values, torch.ones_like(full_ids), prefix_len=prefix_len), device)

    # Cache: first build a cache of L - 1 tokens (not timed), then feed only the last token.
    # The cache is rebuilt for every measurement because a step appends a token to it
    def cached_step():
        cache = paligemma.KVCache()
        model(input_ids, pixel_values, torch.ones_like(input_ids), kv_cache=cache)
        if L - 1 > prefix_len:
            rest = full_ids[:, prefix_len:L - 1]
            model(rest, None, torch.ones_like(rest), kv_cache=cache)
        return cache

    best = float("inf")
    for i in range(4):                                                   # the first is a warm-up
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
    parser.add_argument("--vision-layers", type=int, default=6, help="27 in the real model")
    parser.add_argument("--text-layers", type=int, default=4, help="18 in the real model")
    parser.add_argument("--new-tokens", type=int, default=32)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)

    model = build_model(args.vision_layers, args.text_layers, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"device {device}, SigLIP {args.vision_layers} layers, Gemma {args.text_layers} layers, {n_params:.0f}M parameters")

    input_ids = torch.cat([torch.full((1, N_IMAGE), IMAGE_TOKEN), torch.randint(1, IMAGE_TOKEN, (1, N_PROMPT))], dim=1)
    input_ids = input_ids.to(device)                                     # [1, 266]
    pixel_values = torch.randn(1, 3, 224, 224, device=device)

    print("\n1) Single step (ms), sequence length L = 266 prefix + answer")
    print(f"{'answer':>6} {'L':>5} {'no cache':>10} {'cache':>8} {'speedup':>9}")
    for n_answer in (1, 16, 64, 128):
        L, no_cache, cached = bench_step(model, input_ids, pixel_values, device, n_answer)
        print(f"{n_answer:>6} {L:>5} {no_cache * 1e3:>10.1f} {cached * 1e3:>8.1f} {no_cache / cached:>8.1f}x")

    print(f"\n2) {args.new_tokens} tokens with generate (no eos)")
    results = {}
    for use_cache in (False, True):
        run = lambda: generate.generate(model, input_ids, pixel_values, torch.ones_like(input_ids),
                                        args.new_tokens, eos_token_id=-1, use_cache=use_cache)
        results[use_cache] = (timed(run, device, repeat=1), run())
    for use_cache, (seconds, _) in results.items():
        print(f"   {'cache' if use_cache else 'no cache':>9}: {seconds:.2f} s total, "
              f"{seconds / args.new_tokens * 1e3:.1f} ms/token")
    same = torch.equal(results[True][1], results[False][1])
    print(f"   speedup {results[False][0] / results[True][0]:.1f}x, "
          f"tokens {'identical' if same else 'DIFFERENT'}")


if __name__ == "__main__":
    main()

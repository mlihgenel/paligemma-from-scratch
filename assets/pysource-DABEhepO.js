const y=`"""How much does the KV cache speed things up? Time per token with and without the cache.

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

    print("\\n1) Single step (ms), sequence length L = 266 prefix + answer")
    print(f"{'answer':>6} {'L':>5} {'no cache':>10} {'cache':>8} {'speedup':>9}")
    for n_answer in (1, 16, 64, 128):
        L, no_cache, cached = bench_step(model, input_ids, pixel_values, device, n_answer)
        print(f"{n_answer:>6} {L:>5} {no_cache * 1e3:>10.1f} {cached * 1e3:>8.1f} {no_cache / cached:>8.1f}x")

    print(f"\\n2) {args.new_tokens} tokens with generate (no eos)")
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
`,b=`"""Does our SiglipVisionEmbeddings give the same output as HF?

Run (from the project root):
    .venv/bin/python checks/check_embeddings.py

The first run downloads google/siglip-base-patch16-224 (~800 MB, no login needed).
"""
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipVisionModel

# Add src/ to the import path so that \`from siglip import ...\` works
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from siglip import SiglipVisionConfig, SiglipVisionEmbeddings  # noqa: E402

MODEL_ID = "google/siglip-base-patch16-224"
IMAGE = ROOT / "assets/sample_sign.jpg"


def compare(name, ours, theirs):
    """Print the shape, the largest absolute difference and the allclose result."""
    same_shape = ours.shape == theirs.shape
    max_diff = (ours - theirs).abs().max().item() if same_shape else float("nan")
    ok = same_shape and torch.allclose(ours, theirs, atol=1e-5)
    print(f"[{'OK ' if ok else 'FAIL'}] {name}: ours {tuple(ours.shape)}, HF {tuple(theirs.shape)}, max diff {max_diff:.2e}")
    return ok


def main():
    torch.manual_seed(0)

    # 1) Load the HF model. We only need its embeddings module.
    #    transformers 5 layout: SiglipVisionModel -> embeddings, encoder, post_layernorm, head
    #    (4.x versions had an extra \`vision_model\` level in between)
    #    The checkpoint is the full image + text model, so loading lists the text_model.* weights
    #    as "UNEXPECTED". That is expected: we only take the image side.
    hf_model = SiglipVisionModel.from_pretrained(MODEL_ID).eval()
    hf_emb = hf_model.embeddings

    # 2) Build our module with the numbers from the HF config.
    #    Reading them from the config removes the risk of typing a number wrong.
    hf_cfg = hf_model.config
    config = SiglipVisionConfig(
        num_channels=hf_cfg.num_channels,
        hidden_size=hf_cfg.hidden_size,
        image_size=hf_cfg.image_size,
        patch_size=hf_cfg.patch_size,
    )
    ours = SiglipVisionEmbeddings(config).eval()

    # 3) Copy the weights. If our module names match HF, the keys match one to one.
    #    strict=True (the default) raises here on missing or extra keys,
    #    so a misnamed module (e.g. patch_embeddings) is caught on this line.
    print("HF state_dict keys:", list(hf_emb.state_dict().keys()))
    ours.load_state_dict(hf_emb.state_dict())

    # 4a) Compare on random input: this only tests that the architecture is the same.
    x = torch.randn(2, 3, config.image_size, config.image_size)
    with torch.no_grad():
        ok1 = compare("random input", ours(x), hf_emb(x))

    # 4b) Compare on a real image. pixel_values come from the HF image processor
    #     (our own preprocessing is in src/processing.py and tested by check_processing.py).
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)  # image side only, no tokenizer needed
    image = Image.open(IMAGE).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    print(f"pixel_values: shape {tuple(pixel_values.shape)}, range [{pixel_values.min():.2f}, {pixel_values.max():.2f}]")
    with torch.no_grad():
        ok2 = compare(f"real image ({IMAGE.name})", ours(pixel_values), hf_emb(pixel_values))

    print("\\nResult:", "embeddings match HF ✅" if ok1 and ok2 else "mismatch ❌")
    sys.exit(0 if ok1 and ok2 else 1)


if __name__ == "__main__":
    main()
`,v=`"""Do the SigLIP encoder parts give the same output as HF?

Parts are tested from small to large:

    SiglipAttention          -> HF encoder.layers[0].self_attn
    SiglipMLP                -> HF encoder.layers[0].mlp
    SiglipEncoderLayer       -> HF encoder.layers[0]
    SiglipEncoder            -> HF encoder (all layers)
    SiglipVisionTransformer  -> the whole HF model (embeddings + encoder + post_layernorm)

Run (from the project root):
    .venv/bin/python checks/check_encoder.py                                    # base: 768 dims, 12 layers
    .venv/bin/python checks/check_encoder.py google/siglip-so400m-patch14-224  # PaliGemma's encoder
"""
import dataclasses
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipVisionModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import siglip  # noqa: E402

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "google/siglip-base-patch16-224"
IMAGE = ROOT / "assets/sample_sign.jpg"
# Tolerances: float32 rounding differences pile up over the layers, and values in the deep
# encoder layers grow to ~200. So besides the absolute difference we also allow a relative one (rtol).
ATOL, RTOL = 1e-4, 1e-4


def first(out):
    """HF attention returns (output, attention weights). Ours may return a single tensor."""
    return out[0] if isinstance(out, tuple) else out


def compare(name, ours, theirs):
    same_shape = ours.shape == theirs.shape
    max_diff = (ours - theirs).abs().max().item() if same_shape else float("nan")
    rel_diff = max_diff / theirs.abs().max().item() if same_shape else float("nan")
    ok = same_shape and torch.allclose(ours, theirs, atol=ATOL, rtol=RTOL)
    print(f"[{'OK ' if ok else 'FAIL'}] {name}: ours {tuple(ours.shape)}, HF {tuple(theirs.shape)}, "
          f"max diff {max_diff:.2e} (relative {rel_diff:.1e})")
    return ok


def build_config(hf_cfg):
    """Fill the fields of our config dataclass from the HF config."""
    ours_fields = {f.name for f in dataclasses.fields(siglip.SiglipVisionConfig)}
    wanted = ["num_channels", "hidden_size", "image_size", "patch_size", "intermediate_size",
              "num_hidden_layers", "num_attention_heads", "layer_norm_eps", "attention_dropout"]
    missing = [k for k in wanted if k not in ours_fields]
    if missing:
        print("WARNING: fields missing from our config:", missing)
    return siglip.SiglipVisionConfig(**{k: getattr(hf_cfg, k) for k in wanted if k in ours_fields})


def load(ours, hf_module, name):
    """Copy the HF weights into our module and show any key mismatch readably."""
    hf_sd = {k: v for k, v in hf_module.state_dict().items() if not k.startswith("head.")}  # PaliGemma does not use the head
    ours_keys, hf_keys = set(ours.state_dict()), set(hf_sd)
    if ours_keys != hf_keys:
        print(f"[FAIL] {name}: state_dict keys do not match")
        print("   in ours but not in HF:", sorted(ours_keys - hf_keys)[:8])
        print("   in HF but not in ours:", sorted(hf_keys - ours_keys)[:8])
        return False
    ours.load_state_dict(hf_sd)
    return True


def main():
    torch.manual_seed(0)
    print("Model:", MODEL_ID)
    # attn_implementation="eager": HF computes attention with an explicit matmul + softmax, like ours.
    # The default "sdpa" (PyTorch's fast kernel) is the same math in a different order.
    # transformers 5 layout: embeddings, encoder, post_layernorm, head
    hf = SiglipVisionModel.from_pretrained(MODEL_ID, attn_implementation="eager").eval()
    config = build_config(hf.config)
    print("Our config:", config, "\\n")

    # Embeddings of a real image are the input to the encoder parts,
    # so the test uses values the model actually sees instead of random ones.
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    pixel_values = processor(images=Image.open(IMAGE).convert("RGB"), return_tensors="pt").pixel_values
    with torch.no_grad():
        x = hf.embeddings(pixel_values)  # [1, num_patches, hidden_size]
    hf_layer0 = hf.encoder.layers[0]

    # (our class name, HF counterpart, functions run in the test)
    tests = [
        ("SiglipAttention", hf_layer0.self_attn,
         lambda m: first(m(x)), lambda h: first(h(x))),
        ("SiglipMLP", hf_layer0.mlp,
         lambda m: m(x), lambda h: h(x)),
        ("SiglipEncoderLayer", hf_layer0,
         lambda m: first(m(x)), lambda h: first(h(x, None))),
        ("SiglipEncoder", hf.encoder,
         lambda m: first(m(x)), lambda h: h(x).last_hidden_state),
        ("SiglipVisionTransformer", hf,
         lambda m: first(m(pixel_values)), lambda h: h(pixel_values).last_hidden_state),
    ]

    results = []
    for cls_name, hf_module, run_ours, run_hf in tests:
        cls = getattr(siglip, cls_name)
        ours = cls(config).eval()
        if not load(ours, hf_module, cls_name):
            results.append(False)
            continue
        with torch.no_grad():
            results.append(compare(cls_name, run_ours(ours), run_hf(hf_module)))

    print(f"\\nResult: {sum(results)}/{len(results)} tests passed", "✅" if all(results) else "❌")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
`,x=`"""Do the Gemma parts give the same output as HF?

Parts are tested from small to large:

    GemmaRMSNorm       -> HF GemmaRMSNorm (float32 and bfloat16 input)
    GemmaMLP           -> HF GemmaMLP
    GemmaRotaryEmbedding -> HF GemmaRotaryEmbedding + apply_rotary_pos_emb (cos/sin and rotated Q/K)
    repeat_kv          -> HF repeat_kv (MQA and GQA with n_kv > 1)
    KVCache            -> HF DynamicCache (prefill + token-by-token appends, length)
    GemmaAttention     -> HF GemmaAttention (eager, with and without a causal mask; cached prefill + token by token)
    GemmaDecoderLayer  -> HF GemmaDecoderLayer (attention + MLP + two norms, causal mask)
    GemmaModel         -> HF GemmaModel (embedding scale + RoPE + causal mask + all layers + final norm;
                          cached prefill + token by token, position and mask from the cache length)
    GemmaForCausalLM   -> HF GemmaForCausalLM (logits and weight tying)

Run (from the project root):
    .venv/bin/python checks/check_gemma.py

Nothing is downloaded: the HF modules are built from a small config with random weights, the weights
are copied into our modules and both are compared on the same input. The test with the real weights
is in the Colab notebook.
"""
import sys
from pathlib import Path

import torch
from transformers import DynamicCache
from transformers import GemmaConfig as HFGemmaConfig
from transformers import GemmaForCausalLM as HFGemmaForCausalLM
from transformers.models.gemma.modeling_gemma import GemmaAttention as HFGemmaAttention
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer as HFGemmaDecoderLayer
from transformers.models.gemma.modeling_gemma import GemmaMLP as HFGemmaMLP
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm as HFGemmaRMSNorm
from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding as HFGemmaRotaryEmbedding
from transformers.models.gemma.modeling_gemma import apply_rotary_pos_emb as hf_apply_rotary_pos_emb
from transformers.models.gemma.modeling_gemma import repeat_kv as hf_repeat_kv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import gemma  # noqa: E402

# Small config: the real Gemma 2B has hidden 2048, intermediate 16384. Same logic, fast locally
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
    print(f"[{'OK ' if ok else 'FAIL'}] {name}: ours {tuple(ours.shape)} {ours.dtype}, "
          f"HF {tuple(theirs.shape)} {theirs.dtype}, max diff {max_diff:.2e}")
    return ok


def copy_weights(name, ours, hf):
    """Compare the keys and, if they match, copy the HF weights into our module."""
    ours_keys, hf_keys = set(ours.state_dict()), set(hf.state_dict())
    if ours_keys != hf_keys:
        print(f"[FAIL] {name}: state_dict keys do not match")
        print(f"   in ours but not in HF: {sorted(ours_keys - hf_keys)}")
        print(f"   in HF but not in ours: {sorted(hf_keys - ours_keys)}")
        return False
    ours.load_state_dict(hf.state_dict())
    return True


def check_rmsnorm():
    hf = HFGemmaRMSNorm(SMALL["hidden_size"], eps=EPS).eval()
    # The weight starts at zero. Left at zero, the scale is 1 on both sides and code that uses
    # the weight wrongly (or not at all) would still pass. So we fill it with random values.
    with torch.no_grad():
        hf.weight.normal_()
    ours = gemma.GemmaRMSNorm(SMALL["hidden_size"], eps=EPS).eval()
    if not copy_weights("GemmaRMSNorm", ours, hf):
        return False

    # Inputs can be large because Gemma multiplies embeddings by sqrt(hidden_size)
    x = torch.randn(B, T, SMALL["hidden_size"]) * 40
    with torch.no_grad():
        ok32 = compare("GemmaRMSNorm float32", ours(x), hf(x))

        # The model runs in bf16. With the same order of operations as HF the result must be bit-exact.
        # Written in Llama order (cast to bf16 first, then multiply) a small difference shows up here.
        ours.to(torch.bfloat16), hf.to(torch.bfloat16)
        x16 = x.to(torch.bfloat16)
        ok16 = compare("GemmaRMSNorm bfloat16 (bit-exact)", ours(x16), hf(x16), exact=True)
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


def check_rope():
    hf_config = HFGemmaConfig(**SMALL, hidden_act="gelu_pytorch_tanh")
    hf = HFGemmaRotaryEmbedding(hf_config)
    ours = gemma.GemmaRotaryEmbedding(SMALL["head_dim"], base=hf_config.rope_parameters["rope_theta"])
    # A buffer, so no copy_weights: we compute the frequencies ourselves and they must match
    ok_freq = compare("RoPE inv_freq", ours.inv_freq, hf.inv_freq)

    # Positions need not start at 0 (with a KV cache a generated token sits at position 260 or later)
    position_ids = torch.arange(T).unsqueeze(0).expand(B, -1) + torch.tensor([[0], [300]])
    q = torch.randn(B, SMALL["num_attention_heads"], T, SMALL["head_dim"])
    k = torch.randn(B, SMALL["num_key_value_heads"], T, SMALL["head_dim"])
    cos, sin = ours(q, position_ids)
    hf_cos, hf_sin = hf(q, position_ids)
    ok_cs = compare("RoPE cos", cos, hf_cos) & compare("RoPE sin", sin, hf_sin)

    q_ours, k_ours = gemma.apply_rotary_pos_emb(q, k, cos, sin)
    q_hf, k_hf = hf_apply_rotary_pos_emb(q, k, hf_cos, hf_sin)
    ok_qk = compare("RoPE rotated Q", q_ours, q_hf) & compare("RoPE rotated K", k_ours, k_hf)

    # Rotation must not change a vector's length, only its direction
    norm_kept = torch.allclose(q_ours.norm(dim=-1), q.norm(dim=-1), atol=ATOL)
    print(f"[{'OK ' if norm_kept else 'FAIL'}] RoPE keeps vector lengths")
    return bool(ok_freq and ok_cs and ok_qk and norm_kept)


def check_repeat_kv():
    n_heads = SMALL["num_attention_heads"]
    ok = True
    # 1: MQA like Gemma 2B. 2: GQA (n_kv > 1, where the grouping order shows). 4: n_rep = 1, nothing should change
    for n_kv in (1, 2, 4):
        n_rep = n_heads // n_kv
        x = torch.randn(B, n_kv, T, SMALL["head_dim"])
        ours, theirs = gemma.repeat_kv(x, n_rep), hf_repeat_kv(x, n_rep)
        ok &= compare(f"repeat_kv n_kv={n_kv}, n_rep={n_rep}", ours, theirs, exact=True)

        # Q head h must use KV head h // n_rep (groups are made of neighboring heads)
        grouped = all(torch.equal(ours[:, h], x[:, h // n_rep]) for h in range(n_heads))
        print(f"[{'OK ' if grouped else 'FAIL'}] repeat_kv n_kv={n_kv}: Q head h -> KV head h // n_rep")
        ok &= grouped
    return bool(ok)


def check_kv_cache():
    n_kv, hd, n_layers = SMALL["num_key_value_heads"], SMALL["head_dim"], SMALL["num_hidden_layers"]
    ours, theirs = gemma.KVCache(), DynamicCache()
    ok = ours.get_seq_length() == 0
    print(f"[{'OK ' if ok else 'FAIL'}] KVCache length is 0 when empty")
    # Prefill first (T tokens), then 3 token-by-token steps. Every step updates all layers
    for step, t_new in enumerate((T, 1, 1, 1)):
        for layer in range(n_layers):
            k, v = torch.randn(B, n_kv, t_new, hd), torch.randn(B, n_kv, t_new, hd)
            ok_k, ok_v = ours.update(k, v, layer)
            hf_k, hf_v = theirs.update(k, v, layer)
            if layer == n_layers - 1:
                ok &= compare(f"KVCache step {step} (T_new={t_new}) K, layer {layer}", ok_k, hf_k, exact=True)
                ok &= compare(f"KVCache step {step} (T_new={t_new}) V, layer {layer}", ok_v, hf_v, exact=True)
            else:
                ok &= torch.equal(ok_k, hf_k) and torch.equal(ok_v, hf_v)
    same_len = ours.get_seq_length() == theirs.get_seq_length() == T + 3
    print(f"[{'OK ' if same_len else 'FAIL'}] KVCache length: ours {ours.get_seq_length()}, "
          f"HF {theirs.get_seq_length()}, expected {T + 3}")
    return bool(ok and same_len)


def check_attention():
    hf_config = HFGemmaConfig(**SMALL, hidden_act="gelu_pytorch_tanh")
    hf_config._attn_implementation = "eager"      # not the sdpa/flash kernels, but the plain math
    hf = HFGemmaAttention(hf_config, layer_idx=0).eval()
    ours = gemma.GemmaAttention(gemma.GemmaConfig(**SMALL), layer_idx=0).eval()
    if not copy_weights("GemmaAttention", ours, hf):
        return False

    hf_rope = HFGemmaRotaryEmbedding(hf_config)
    x = torch.randn(B, T, SMALL["hidden_size"])
    position_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
    cos, sin = hf_rope(x, position_ids)

    # Causal mask: token i may only look at j <= i. Additive form: allowed 0, blocked -inf
    mask = torch.full((T, T), float("-inf")).triu(diagonal=1)[None, None].expand(B, 1, T, T)

    with torch.no_grad():
        ok_mask = compare("GemmaAttention (causal mask)",
                          ours(x, (cos, sin), mask), hf(x, (cos, sin), mask)[0])
        ok_nomask = compare("GemmaAttention (no mask)", ours(x, (cos, sin)), hf(x, (cos, sin), None)[0])

        # Does the causal mask work: changing a future token must not affect earlier outputs
        x2 = x.clone()
        x2[:, -1] += 10.0
        out1, out2 = ours(x, (cos, sin), mask), ours(x2, (cos, sin), mask)
        causal = torch.allclose(out1[:, :-1], out2[:, :-1], atol=ATOL) and not torch.allclose(out1[:, -1], out2[:, -1])
        print(f"[{'OK ' if causal else 'FAIL'}] GemmaAttention: changing the last token leaves earlier outputs unchanged")

        ok_cache = True
        if hasattr(gemma, "KVCache"):
            # With a cache: the first T-3 tokens are the prefill (causal mask), the last 3 go one by one. A single token's
            # mask is all zeros: it may look at the whole past, and the future is not in the cache anyway
            ours_cache, hf_cache = gemma.KVCache(), DynamicCache()
            P = T - 3
            outs = []
            for start, end in [(0, P)] + [(t, t + 1) for t in range(P, T)]:
                step_mask = mask[:, :, start:end, :end]                 # [B, 1, T_new, T_all]
                cs = (cos[:, start:end], sin[:, start:end])
                o = ours(x[:, start:end], cs, step_mask, kv_cache=ours_cache)
                h = hf(x[:, start:end], cs, step_mask, past_key_values=hf_cache)[0]
                ok_cache &= compare(f"GemmaAttention cached, tokens {start}..{end - 1}", o, h)
                outs.append(o)
            # The real claim: the cached steps joined together equal the full sequence processed at once
            ok_cache &= compare("GemmaAttention cached steps = uncached full sequence",
                                torch.cat(outs, dim=1), ours(x, (cos, sin), mask))
    return bool(ok_mask and ok_nomask and causal and ok_cache)


def hf_eager_config():
    cfg = HFGemmaConfig(**SMALL, hidden_act="gelu_pytorch_tanh")
    cfg._attn_implementation = "eager"      # not the sdpa/flash kernels, but the same math we wrote
    return cfg


def check_decoder_layer():
    hf_config = hf_eager_config()
    hf = HFGemmaDecoderLayer(hf_config, layer_idx=0).eval()
    ours = gemma.GemmaDecoderLayer(gemma.GemmaConfig(**SMALL), layer_idx=0).eval()

    # Randomize the norm weights. Left at zero, both norms behave the same and code that
    # swaps input_layernorm and post_attention_layernorm would still pass
    with torch.no_grad():
        hf.input_layernorm.weight.normal_()
        hf.post_attention_layernorm.weight.normal_()
    if not copy_weights("GemmaDecoderLayer", ours, hf):
        return False

    x = torch.randn(B, T, SMALL["hidden_size"]) * 40
    position_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
    cos, sin = HFGemmaRotaryEmbedding(hf_config)(x, position_ids)
    mask = gemma.make_causal_mask(T, x.dtype, x.device)
    with torch.no_grad():
        return compare("GemmaDecoderLayer", ours(x, (cos, sin), mask),
                       hf(x, attention_mask=mask, position_embeddings=(cos, sin)))


def randomize_norms(module):
    """Randomize all RMSNorm weights (left at zero, they all behave the same)."""
    with torch.no_grad():
        for n, p in module.named_parameters():
            if "norm" in n:
                p.normal_()


def scaled_embeds(ours_model):
    """Embeddings for the HF model with the scale already applied.

    transformers 5.x applies the sqrt(hidden_size) scale inside the embedding table
    (GemmaTextScaledWordEmbedding) and does not scale inputs_embeds passed in directly. Our
    GemmaModel applies the scale at the start of forward, on inputs_embeds (the older HF approach).
    To compare like with like: our model gets the raw embeddings, HF gets them multiplied by sqrt(d).
    """
    input_ids = torch.randint(0, SMALL["vocab_size"], (B, T))
    raw = ours_model.embed_tokens(input_ids)                       # [B, T, D], unscaled
    return raw, raw * torch.tensor(SMALL["hidden_size"] ** 0.5, dtype=raw.dtype)


def check_model():
    hf_config = hf_eager_config()
    hf = HFGemmaForCausalLM(hf_config).model.eval()   # the GemmaModel part
    ours = gemma.GemmaModel(gemma.GemmaConfig(**SMALL)).eval()
    randomize_norms(hf)
    if not copy_weights("GemmaModel", ours, hf):
        return False

    with torch.no_grad():
        raw, scaled = scaled_embeds(ours)
        ok = compare("GemmaModel", ours(raw), hf(inputs_embeds=scaled).last_hidden_state)

        if hasattr(gemma, "KVCache"):
            # No position or mask passed: both must follow from the cache length. HF does the same internally
            ours_cache, hf_cache = gemma.KVCache(), DynamicCache()
            P = T - 3
            outs = []
            for start, end in [(0, P)] + [(t, t + 1) for t in range(P, T)]:
                o = ours(raw[:, start:end], kv_cache=ours_cache)
                h = hf(inputs_embeds=scaled[:, start:end], past_key_values=hf_cache, use_cache=True).last_hidden_state
                ok &= compare(f"GemmaModel cached, tokens {start}..{end - 1}", o, h)
                outs.append(o)
            ok &= compare("GemmaModel cached steps = uncached full sequence", torch.cat(outs, dim=1), ours(raw))

            # Several new tokens with a past (past_len > 0, T > 1): the mask must be rectangular
            rect = gemma.make_causal_mask(3, torch.float32, torch.device("cpu"), past_len=4)[0, 0]
            expected = gemma.make_causal_mask(7, torch.float32, torch.device("cpu"))[0, 0, 4:]
            ok &= compare("make_causal_mask past_len=4, T=3 = last 3 rows of the square mask", rect, expected, exact=True)
        return bool(ok)


def check_causal_lm():
    hf_config = hf_eager_config()
    hf = HFGemmaForCausalLM(hf_config).eval()
    ours = gemma.GemmaForCausalLM(gemma.GemmaConfig(**SMALL)).eval()
    ours.tie_weights()   # in our code PaliGemma calls tying, here we do it by hand
    randomize_norms(hf)
    if not copy_weights("GemmaForCausalLM", ours, hf):
        return False

    # load_state_dict copies values but keeps the tying: both names must still be the same Parameter object
    tied = ours.lm_head.weight is ours.model.embed_tokens.weight
    print(f"[{'OK ' if tied else 'FAIL'}] weight tying: lm_head.weight is {'the same' if tied else 'a DIFFERENT'} "
          f"Parameter object as embed_tokens.weight")

    with torch.no_grad():
        raw, scaled = scaled_embeds(ours.model)
        ok = compare("GemmaForCausalLM logits", ours(raw), hf(inputs_embeds=scaled).logits)
    return tied and ok


def main():
    torch.manual_seed(0)
    checks = [
        ("GemmaRMSNorm", check_rmsnorm),
        ("GemmaMLP", check_mlp),
        ("GemmaRotaryEmbedding", check_rope),
        ("repeat_kv", check_repeat_kv),
        ("KVCache", check_kv_cache),
        ("GemmaAttention", check_attention),
        ("GemmaDecoderLayer", check_decoder_layer),
        ("GemmaModel", check_model),
        ("GemmaForCausalLM", check_causal_lm),
    ]

    results = []
    for cls_name, fn in checks:
        if not hasattr(gemma, cls_name):
            print(f"[ -- ] {cls_name}: not in gemma.py, skipped")
            continue
        try:
            results.append(fn())
        except Exception as e:
            print(f"[FAIL] {cls_name}: {type(e).__name__}: {e}")
            results.append(False)

    ok = bool(results) and all(results)
    print(f"\\nResult: {'all parts match HF ✅' if ok else 'mismatch ❌'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
`,w=`"""Do the PaliGemma parts work correctly?

    make_prefix_lm_mask -> against a hand-written expected pattern
    load_hf_weights, convert_hf_key -> are all HF weights loaded into our model under the right names
    PaliGemmaForConditionalGeneration -> HF PaliGemmaForConditionalGeneration (small random config, logits)
    (bf16)             -> the same model in bf16 is bit-exact with HF (unscaled image merge)
    (KV cache)         -> cached prefill + token-by-token logits = uncached full sequence (prefix + suffix)
    generate (src/generate.py) -> HF generate(do_sample=False): same tokens, including stopping at eos;
                          cached and uncached (use_cache=False) generation also match each other

Run (from the project root):
    .venv/bin/python checks/check_paligemma.py
"""
import sys
from pathlib import Path

import torch
from transformers import GemmaConfig as HFGemmaConfig
from transformers import PaliGemmaConfig as HFPaliGemmaConfig
from transformers import PaliGemmaForConditionalGeneration as HFPaliGemma
from transformers import SiglipVisionConfig as HFSiglipVisionConfig

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import generate  # noqa: E402
import paligemma  # noqa: E402


def compare(name, ours, theirs):
    same_shape = ours.shape == theirs.shape
    max_diff = (ours - theirs).abs().max().item() if same_shape else float("nan")
    ok = same_shape and torch.allclose(ours, theirs, atol=ATOL)
    print(f"[{'OK ' if ok else 'FAIL'}] {name}: ours {tuple(ours.shape)}, HF {tuple(theirs.shape)}, max diff {max_diff:.2e}")
    return ok


def check_prefix_lm_mask():
    # 5 tokens: the first 3 are prefix (image + prompt), the last 2 suffix (generated answer)
    total_len, prefix_len = 5, 3
    mask = paligemma.make_prefix_lm_mask(total_len, prefix_len, torch.float32, torch.device("cpu"))
    allowed = (mask[0, 0] == 0)

    # 1 = may look. Prefix rows see the whole prefix but not the suffix.
    # Suffix rows see the whole prefix + the tokens before them and themselves
    expected = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
    ], dtype=torch.bool)

    print("allowed positions (1 = may look):")
    for row in allowed.int().tolist():
        print("  ", row)

    ok_pattern = torch.equal(allowed, expected)
    print(f"[{'OK ' if ok_pattern else 'FAIL'}] prefix-LM mask: bidirectional prefix, causal suffix")

    # Blocked positions must be the smallest number, allowed ones exactly 0 (added to the scores in attention)
    ok_values = bool((mask[0, 0][~expected] == torch.finfo(torch.float32).min).all()
                     and (mask[0, 0][expected] == 0).all())
    print(f"[{'OK ' if ok_values else 'FAIL'}] mask values: allowed 0, blocked finfo.min")

    # prefix_len = 1 must equal the causal mask, prefix_len = total_len full attention
    from gemma import make_causal_mask
    ok_causal = torch.equal(paligemma.make_prefix_lm_mask(total_len, 1, torch.float32, torch.device("cpu")),
                            make_causal_mask(total_len, torch.float32, torch.device("cpu")))
    full = paligemma.make_prefix_lm_mask(total_len, total_len, torch.float32, torch.device("cpu"))
    ok_full = bool((full == 0).all())
    print(f"[{'OK ' if ok_causal else 'FAIL'}] prefix_len=1: same as the causal mask")
    print(f"[{'OK ' if ok_full else 'FAIL'}] prefix_len=total_len: full attention (nothing blocked)")
    return ok_pattern and ok_values and ok_causal and ok_full


# Small config: the real model has vision 1152/27 layers, text 2048/18 layers. Same logic, fast locally
VISION = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
              patch_size=14, image_size=28)                      # 28/14 = 2x2 = 4 image tokens
TEXT = dict(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=1, head_dim=16, pad_token_id=0)
IMAGE_TOKEN = 257
B, N_TEXT = 2, 6
ATOL = 1e-4


def build_models(hidden_size=TEXT["hidden_size"], bf16=False):
    text = dict(TEXT, hidden_size=hidden_size)
    hf_config = HFPaliGemmaConfig(
        vision_config=HFSiglipVisionConfig(**VISION, projection_dim=hidden_size),
        text_config=HFGemmaConfig(**text, hidden_act="gelu_pytorch_tanh"),
        image_token_index=IMAGE_TOKEN, projection_dim=hidden_size,
        hidden_size=hidden_size, vocab_size=text["vocab_size"],
    )
    hf_config._attn_implementation = "eager"
    hf_config.text_config._attn_implementation = "eager"
    hf_config.vision_config._attn_implementation = "eager"
    hf = HFPaliGemma(hf_config).eval()
    # The init weights are tiny (std 0.02): logits come out near zero and even a wrong mask or
    # position would pass. So everything gets larger random values (norm weights included)
    with torch.no_grad():
        for p in hf.parameters():
            p.normal_(0, 0.2)

    ours_config = paligemma.PaliGemmaConfig(
        vision_config=paligemma.SiglipVisionConfig(**VISION),
        text_config=paligemma.GemmaConfig(**text),
        image_token_index=IMAGE_TOKEN, projection_dim=hidden_size,
        hidden_size=hidden_size, vocab_size=text["vocab_size"],
    )
    if not bf16:
        return hf, paligemma.PaliGemmaForConditionalGeneration(ours_config).eval()

    # bf16: as if loaded with HF from_pretrained(dtype=bf16). A plain .to(bf16) would also round the RoPE
    # frequencies (a buffer) to bf16, while from_pretrained keeps them float32: put them back
    inv_freq = hf.model.language_model.rotary_emb.inv_freq.clone()
    hf = hf.to(torch.bfloat16)
    hf.model.language_model.rotary_emb.inv_freq = inv_freq
    # Ours is built with bf16 as the default dtype (as in Colab): inv_freq stays float32 because it is created explicitly so
    torch.set_default_dtype(torch.bfloat16)
    try:
        ours = paligemma.PaliGemmaForConditionalGeneration(ours_config).eval()
    finally:
        torch.set_default_dtype(torch.float32)
    return hf, ours


def build_loaded_pair(**kwargs):
    """The HF model + our model with its weights copied by the loader in src/paligemma.py."""
    hf, ours = build_models(**kwargs)
    paligemma.load_hf_weights(ours, hf.state_dict())
    return hf, ours


def check_loader():
    hf, ours = build_models()
    paligemma.load_hf_weights(ours, hf.state_dict())

    # Is every weight the same as in HF (except head.*), and is weight tying in place
    ours_sd = ours.state_dict()
    same = all(torch.equal(ours_sd[paligemma.convert_hf_key(k)], v)
               for k, v in hf.state_dict().items() if paligemma.convert_hf_key(k) is not None)
    tied = ours.language_model.lm_head.weight is ours.language_model.model.embed_tokens.weight
    print(f"[{'OK ' if same else 'FAIL'}] load_hf_weights: all weights match HF")
    print(f"[{'OK ' if tied else 'FAIL'}] load_hf_weights: lm_head.weight is the same Parameter as the embedding")

    # Names in the old (4.x) layout must also map to ours, and the head must be skipped
    old_layout = {"vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight": "vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight",
                  "language_model.model.layers.0.self_attn.q_proj.weight": "language_model.model.layers.0.self_attn.q_proj.weight",
                  "language_model.lm_head.weight": "language_model.lm_head.weight",
                  "vision_tower.vision_model.head.probe": None,
                  "model.vision_tower.head.probe": None}
    old_ok = all(paligemma.convert_hf_key(k) == v for k, v in old_layout.items())
    print(f"[{'OK ' if old_ok else 'FAIL'}] convert_hf_key: old layout passes through, head is skipped")

    # A missing key must not pass silently
    broken = {k: v for k, v in hf.state_dict().items() if "post_layernorm" not in k}
    try:
        paligemma.load_hf_weights(ours, broken)
        raised = False
    except KeyError:
        raised = True
    print(f"[{'OK ' if raised else 'FAIL'}] load_hf_weights: raises on a missing weight")
    return bool(same and tied and old_ok and raised)


def make_inputs():
    # [<image> x4, text...]. Token ids differ from 0 (pad) and from the image token
    input_ids = torch.cat([torch.full((B, 4), IMAGE_TOKEN), torch.randint(5, 250, (B, N_TEXT))], dim=1)
    pixel_values = torch.randn(B, 3, VISION["image_size"], VISION["image_size"])
    return input_ids, pixel_values, torch.ones_like(input_ids)


def check_model():
    hf, ours = build_loaded_pair()
    if hf is None:
        return False

    input_ids, pixel_values, attention_mask = make_inputs()
    T = input_ids.shape[1]

    ok = True
    with torch.no_grad():
        # 1) The whole input is prefix (no answer): token_type_ids = 0 makes HF treat the prefix as bidirectional
        hf_logits = hf(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask,
                       token_type_ids=torch.zeros(B, T, dtype=torch.long)).logits
        ok &= compare("PaliGemma logit'ler (hepsi prefix)", ours(input_ids, pixel_values, attention_mask), hf_logits)

        # 2) Son 3 token suffix (cevap): HF'de token_type_ids = 1
        prefix_len = T - 3
        token_type_ids = torch.cat([torch.zeros(B, prefix_len), torch.ones(B, 3)], dim=1).long()
        hf_logits_suffix = hf(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask,
                              token_type_ids=token_type_ids).logits
        ok &= compare("PaliGemma logit'ler (prefix + suffix)",
                      ours(input_ids, pixel_values, attention_mask, prefix_len=prefix_len), hf_logits_suffix)

        # 3) Without token_type_ids HF makes the prefix causal too. Our prefix-LM result must differ from that
        # (this also shows the test is sensitive: a wrong mask would give a wrong result)
        hf_causal = hf(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask).logits
        gap = (hf_causal - hf_logits).abs().max().item()
        differs = gap > 1e-2
        print(f"[{'OK ' if differs else 'FAIL'}] HF without token_type_ids (causal) differs from prefix-LM logits: max diff {gap:.2e}")
        ok &= differs

        # 4) Same result whether positions start at 0 or 1 (RoPE only depends on position differences)
        lm = ours.language_model
        embeds = torch.randn(B, T, TEXT["hidden_size"])
        mask = paligemma.make_prefix_lm_mask(T, prefix_len, embeds.dtype, embeds.device)
        pos0 = torch.arange(T).unsqueeze(0).expand(B, -1)
        ok &= compare("logits with positions from 0 vs from 1",
                      lm(embeds, mask, pos0), lm(embeds, mask, pos0 + 1))
    return bool(ok)


def check_kv_cache():
    _, ours = build_loaded_pair()
    input_ids, pixel_values, attention_mask = make_inputs()
    T = input_ids.shape[1]
    prefix_len = T - 3            # image + part of the prompt is prefix, the last 3 tokens go one by one like an "answer"

    ok = True
    with torch.no_grad():
        # Referans: cache'siz, tam dizi tek seferde
        full = ours(input_ids, pixel_values, attention_mask, prefix_len=prefix_len)          # [B, T, vocab]

        # Prefill: only the prefix, with the image. Then the suffix tokens one at a time, without the image
        cache = paligemma.KVCache()
        steps = [ours(input_ids[:, :prefix_len], pixel_values, attention_mask[:, :prefix_len], kv_cache=cache)]
        for t in range(prefix_len, T):
            steps.append(ours(input_ids[:, t:t + 1], None, attention_mask[:, t:t + 1], kv_cache=cache))
        ok &= compare("PaliGemma cached (prefill + 3 steps) = uncached full sequence", torch.cat(steps, dim=1), full)

        n_layers = TEXT["num_hidden_layers"]
        ok_len = len(cache.key_cache) == n_layers and cache.get_seq_length() == T
        print(f"[{'OK ' if ok_len else 'FAIL'}] cache: {len(cache.key_cache)} layers (expected {n_layers}), "
              f"length {cache.get_seq_length()} (expected {T})")
        ok &= ok_len
    return bool(ok)


def check_bf16_exact():
    # The real model runs in bf16. hidden_size=48: sqrt(48) is not a power of 2. With sqrt(64) = 8 the
    # "divide the image by sqrt(d), then multiply everything" approach would not round and this test would miss it
    hf, ours = build_loaded_pair(hidden_size=48, bf16=True)
    input_ids, pixel_values, attention_mask = make_inputs()
    pixel_values = pixel_values.to(torch.bfloat16)
    with torch.no_grad():
        hf_logits = hf(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask,
                       token_type_ids=torch.zeros_like(input_ids)).logits.float()
        our_logits = ours(input_ids, pixel_values, attention_mask).float()
    max_diff = (our_logits - hf_logits).abs().max().item()
    ok = torch.equal(our_logits, hf_logits)
    print(f"[{'OK ' if ok else 'FAIL'}] bf16, hidden_size=48: logits bit-exact with HF (max diff {max_diff:.2e})")
    return ok


def check_generate():
    hf, ours = build_loaded_pair()
    if hf is None:
        return False
    input_ids, pixel_values, attention_mask = make_inputs()
    T = input_ids.shape[1]
    steps = 8

    def hf_generate(eos):
        # token_type_ids = 0: bidirectional prefix. During generation HF itself adds 1 for new tokens (suffix)
        out = hf.generate(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask,
                          token_type_ids=torch.zeros_like(input_ids), max_new_tokens=steps, do_sample=False,
                          eos_token_id=eos, pad_token_id=TEXT["pad_token_id"])
        return out[:, T:]                                    # HF also returns the prompt

    def ours_generate(eos, use_cache):
        return generate.generate(ours, input_ids, pixel_values, attention_mask, steps,
                                 eos_token_id=eos, use_cache=use_cache)

    ok = True
    # 1) no eos: every sample generates max_new_tokens tokens
    # 2) eos: declare the first sample's 3rd token as eos. That sample must stop there and pad, the other must go on
    eos = int(ours_generate(-1, use_cache=True)[0, 2])
    for label, our_eos, hf_eos in [("no eos", -1, None), (f"eos={eos}", eos, eos)]:
        hf_tokens = hf_generate(eos=hf_eos)
        for use_cache in (True, False):
            ours_tokens = ours_generate(our_eos, use_cache)
            same = torch.equal(ours_tokens, hf_tokens)
            print(f"[{'OK ' if same else 'FAIL'}] greedy generation ({'cache' if use_cache else 'no cache'}), "
                  f"{label}: first sample {ours_tokens[0].tolist()}")
            if not same:
                print(f"   HF: {hf_tokens[0].tolist()}")
            ok &= same
    return bool(ok)


def main():
    torch.manual_seed(0)
    checks = [
        ("make_prefix_lm_mask", check_prefix_lm_mask),
        ("load_hf_weights", check_loader),
        ("PaliGemmaForConditionalGeneration", check_model),
        ("PaliGemmaForConditionalGeneration", check_bf16_exact),
        ("KVCache", check_kv_cache),
        ("generate", check_generate),
    ]

    results = []
    for name, fn in checks:
        if not hasattr(paligemma, name) and not hasattr(generate, name):
            print(f"[ -- ] {name}: not in paligemma.py, skipped")
            continue
        try:
            results.append(fn())
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            results.append(False)

    ok = bool(results) and all(results)
    print(f"\\nResult: {'everything works as expected ✅' if ok else 'mismatch ❌'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
`,T=`"""Do our process_images and PaliGemmaProcessor give the same pixel_values and input_ids as HF?

Run (from the project root):
    .venv/bin/python checks/check_processing.py

The image side only needs the preprocessor_config.json of google/siglip-base-patch16-224 (no login).
The text side downloads the tokenizer files of google/paligemma-3b-pt-224 (~22 MB, gated: needs \`hf auth login\`).
PaliGemma's image preprocessing is the same: 224x224, bicubic, mean=std=0.5.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoImageProcessor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import processing  # noqa: E402

MODEL_ID = "google/siglip-base-patch16-224"
IMAGE = ROOT / "assets/sample_sign.jpg"

# Settings for our function (the same values as in the HF processor config)
SIZE = (224, 224)  # (height, width)
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


def test_images():
    """The sample photo plus two crops with other sizes and aspect ratios, so resizing is tested on more than one shape."""
    image = Image.open(IMAGE).convert("RGB")
    w, h = image.size
    wide = image.crop((0, h // 4, w, h // 4 + w // 2))                 # landscape, 2:1
    small = image.crop((w // 4, h // 4, w // 4 + 150, h // 4 + 150))   # small square, upscaled to 224
    return [("sample", image), ("wide crop", wide), ("small crop", small)]


def main():
    fn = getattr(processing, "process_images", None)
    if fn is None:
        print("[FAIL] no process_images function in processing.py")
        sys.exit(1)

    hf = AutoImageProcessor.from_pretrained(MODEL_ID)
    print(f"HF processor: {type(hf).__name__}, size={hf.size}, resample={hf.resample}, "
          f"mean={hf.image_mean}, std={hf.image_std}\\n")

    named = test_images()
    images = [img for _, img in named]

    ours = fn(
        images=images,
        size=SIZE,
        resample=Image.Resampling.BICUBIC,
        rescale_factor=1 / 255.0,
        image_mean=MEAN,
        image_std=STD,
    )
    ours = np.stack(ours)  # list -> [B, 3, H, W]
    theirs = hf(images=images, return_tensors="np").pixel_values

    ok_all = True
    for i, (name, img) in enumerate(named):
        o, t = ours[i], theirs[i]
        same_shape = o.shape == t.shape
        max_diff = float(np.abs(o - t).max()) if same_shape else float("nan")
        ok = same_shape and o.dtype == np.float32 and np.allclose(o, t, atol=1e-5)
        ok_all &= ok
        print(f"[{'OK ' if ok else 'FAIL'}] {name} ({img.size[0]}x{img.size[1]}): "
              f"ours {o.shape} {o.dtype}, range [{o.min():.2f}, {o.max():.2f}] | "
              f"HF {t.shape}, max diff {max_diff:.2e}")
        if same_shape and o.dtype != np.float32:
            print(f"   dtype should be float32, ours is {o.dtype}")

    print(f"\\nImage side: {'matches HF ✅' if ok_all else 'mismatch ❌'}\\n")

    ok_all &= check_processor(images[0])
    print(f"\\nResult: {'processing matches HF ✅' if ok_all else 'mismatch ❌'}")
    sys.exit(0 if ok_all else 1)


def check_processor(image):
    """Text + image: does our PaliGemmaProcessor give the same tensors as HF's PaliGemmaProcessor?

    google/paligemma-3b-pt-224 is a gated repo: needs \`hf auth login\`. Only the tokenizer and
    config files are downloaded (~22 MB), not the model weights.
    """
    from transformers import AutoProcessor, AutoTokenizer

    pg_id = "google/paligemma-3b-pt-224"
    hf_proc = AutoProcessor.from_pretrained(pg_id)
    tokenizer = AutoTokenizer.from_pretrained(pg_id)
    ours = processing.PaliGemmaProcessor(tokenizer, num_image_tokens=256, image_size=224)

    # Task prefixes the pt model was trained with. The Turkish prompts also exercise non-ASCII tokenization
    prompts = ["caption en", "answer tr Tabelada ne yazıyor?", "ocr", "detect sign"]
    ok_all = True
    for prompt in prompts:
        o = ours(text=[prompt], images=[image])
        t = hf_proc(text="<image>" + prompt, images=image, return_tensors="pt")

        problems = []
        for key in ["input_ids", "attention_mask", "pixel_values"]:
            if key not in o:
                problems.append(f"{key} is not returned")
                continue
            a, b = o[key], t[key]
            if tuple(a.shape) != tuple(b.shape):
                problems.append(f"{key} shape {tuple(a.shape)}, HF {tuple(b.shape)}")
            elif key == "pixel_values":
                if not np.allclose(a.numpy(), b.numpy(), atol=1e-5):
                    problems.append(f"pixel_values differ (max diff {float((a - b).abs().max()):.2e})")
            elif not (a == b).all():
                problems.append(f"{key} differs")

        ok = not problems
        ok_all &= ok
        ids = o["input_ids"][0].tolist() if "input_ids" in o else []
        print(f"[{'OK ' if ok else 'FAIL'}] {prompt!r}: {len(ids)} tokens")
        if ids:
            print(f"   last tokens (ours): {tokenizer.convert_ids_to_tokens(ids[254:])}")
        if not ok:
            hf_ids = t["input_ids"][0].tolist()
            print(f"   last tokens (HF)  : {tokenizer.convert_ids_to_tokens(hf_ids[254:])}")
            for p in problems:
                print(f"   {p}")
    return ok_all


if __name__ == "__main__":
    main()
`,z=`"""Does our PaliGemmaMultiModalProjector give the same output as HF?

Run (from the project root):
    .venv/bin/python checks/check_projector.py

Nothing is downloaded: the HF projector is built from the PaliGemma 3B config with random weights,
the weights are copied into our module and both are compared on the same input. The test with
the real weights is in the Colab notebook.
"""
import sys
from pathlib import Path

import torch
from transformers import PaliGemmaConfig
from transformers.models.paligemma.modeling_paligemma import PaliGemmaMultiModalProjector as HFProjector

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import paligemma  # noqa: E402
from siglip import SiglipVisionConfig  # noqa: E402

# paligemma-3b-pt-224 values: so400m (1152) -> Gemma 2B (2048), 256 image tokens
VISION_HIDDEN = 1152
PROJECTION_DIM = 2048
NUM_TOKENS = 256


def main():
    torch.manual_seed(0)

    if not hasattr(paligemma, "PaliGemmaMultiModalProjector"):
        print("[FAIL] no PaliGemmaMultiModalProjector class in paligemma.py")
        sys.exit(1)

    hf_config = PaliGemmaConfig(vision_config={"hidden_size": VISION_HIDDEN, "projection_dim": PROJECTION_DIM})
    hf = HFProjector(hf_config).eval()

    config = SiglipVisionConfig(hidden_size=VISION_HIDDEN, projection_dim=PROJECTION_DIM)
    ours = paligemma.PaliGemmaMultiModalProjector(config).eval()

    # Same keys? (in the real checkpoint: multi_modal_projector.linear.weight / .bias)
    ours_keys, hf_keys = set(ours.state_dict()), set(hf.state_dict())
    if ours_keys != hf_keys:
        print("[FAIL] state_dict keys do not match")
        print(f"   in ours but not in HF: {sorted(ours_keys - hf_keys)}")
        print(f"   in HF but not in ours: {sorted(hf_keys - ours_keys)}")
        sys.exit(1)
    ours.load_state_dict(hf.state_dict())

    x = torch.randn(1, NUM_TOKENS, VISION_HIDDEN)  # stands in for the SigLIP output
    with torch.no_grad():
        o, t = ours(x), hf(x)

    same_shape = o.shape == t.shape
    max_diff = (o - t).abs().max().item() if same_shape else float("nan")
    ok = same_shape and torch.allclose(o, t, atol=1e-5)
    print(f"keys: {sorted(ours_keys)}")
    print(f"[{'OK ' if ok else 'FAIL'}] projector: input {tuple(x.shape)} -> ours {tuple(o.shape)}, "
          f"HF {tuple(t.shape)}, max diff {max_diff:.2e}")
    print(f"\\nResult: {'projector matches HF ✅' if ok else 'mismatch ❌'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
`,G=`from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class GemmaConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int = 256
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    pad_token_id: int | None = None
    num_image_tokens: int = 256
    
# Unlike LayerNorm: no mean subtraction and no bias, only the vector's magnitude (RMS) is fixed
class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Starts at zero: the scale is (1 + weight), so it is exactly 1 at init
        self.weight = nn.Parameter(torch.zeros(dim))

    # x / sqrt(mean(x²) + eps). Each token is normalized by its own vector: [B, T, D] -> [B, T, 1] -> broadcast
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The mean of squares loses precision in bf16, so compute in float32
        output = self._norm(x.float())
        # Llama casts back to dtype and then multiplies, Gemma multiplies in float32 and then casts back
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)
    
# GeGLU: two parallel paths instead of the single path of a classic MLP. gate carries "how much passes", up carries "what passes".
# Each token is processed on its own, nothing mixes across tokens (that is attention's job)
class GemmaMLP(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # All three without bias. Gemma 2B: 2048 -> 16384 -> 2048, ~100M parameters per layer
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)                    # [B, T, D] -> [B, T, I]
        up = self.up_proj(x)                        # [B, T, D] -> [B, T, I]
        # Activation only on gate. Gemma was trained with the tanh approximation, exact gelu breaks allclose
        h = F.gelu(gate, approximate="tanh") * up   # [B, T, I], elementwise product
        return self.down_proj(h)                    # [B, T, I] -> [B, T, D]


# RoPE: position is not added to the embedding, Q and K vectors are rotated by position instead.
# A head_dim vector is split into head_dim/2 two-dimensional pairs, each pair rotates at its own speed:
# pair i at position m rotates by m * theta_i. theta_i = base^(-2i/head_dim): first pairs fast, last pairs slow
class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        # [head_dim/2] frequencies. Not learned, fixed: persistent=False keeps them out of the state_dict
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    # Only builds the cos/sin table, does not touch Q/K. Only dtype and device are taken from x
    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Angles in float32: for large positions like 8000 the angle comes out wrong in bf16
        # [B, T, 1] * [head_dim/2] -> [B, T, head_dim/2], freqs[b, m, i] = m * theta_i
        freqs = position_ids[..., None].float() * self.inv_freq.to(x.device)
        # Same angle twice: rotate_half pairs (i, i + head_dim/2), so the second half needs it too
        emb = torch.cat((freqs, freqs), dim=-1)                   # [B, T, head_dim]
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


# The 2D rotation from the paper: (a, b) -> (a cos - b sin, b cos + a sin).
# Pairs are not neighbors (0,1), (2,3)... but first half with second half: (x_0, x_{d/2}), (x_1, x_{d/2+1})...
# rotate_half(x) = [-x2, x1] -> x * cos + rotate_half(x) * sin applies this rotation to all pairs at once
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,    # [B, n_heads, T, head_dim]
    k: torch.Tensor,    # [B, n_kv_heads, T, head_dim]
    cos: torch.Tensor,  # [B, T, head_dim]
    sin: torch.Tensor,  # [B, T, head_dim]
) -> tuple[torch.Tensor, torch.Tensor]:
    # Add a head axis: [B, 1, T, head_dim], every head rotates by the same angle at the same position
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# GQA/MQA: K and V have fewer heads than Q (1 in Gemma 2B). To compute scores, each KV head is repeated
# once per Q head that uses it (n_rep = n_heads / n_kv_heads).
# The stored K/V (the cache) stays small, the repeat only happens at compute time
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    B, n_kv_heads, T, head_dim = x.shape
    if n_rep == 1:
        return x
    # [B, n_kv, T, hd] -> [B, n_kv, 1, T, hd] -> [B, n_kv, n_rep, T, hd]: expand copies no data, it is just a view
    x = x[:, :, None, :, :].expand(B, n_kv_heads, n_rep, T, head_dim)
    # Merge: the first n_rep heads copy the first KV head, the next n_rep the second... (Q heads are grouped in the same order)
    # With n_kv = 1 (MQA) reshape returns a view too. With n_kv > 1 the memory is not contiguous, so it copies
    return x.reshape(B, n_kv_heads * n_rep, T, head_dim)           # [B, n_heads, T, hd]


# During generation each step computes K/V only for the new token, the past tokens' K/V are kept here.
# Q is not kept: past tokens already asked their question and got their answer. The new token's Q,
# however, has to look at all past K's. One K and one V tensor per layer (every layer has its own K/V)
class KVCache:
    def __init__(self):
        self.key_cache: list[torch.Tensor] = []      # per layer [B, n_kv, T_past, hd]
        self.value_cache: list[torch.Tensor] = []

    # How many tokens have been seen. All layers have the same length, so layer 0 is enough. 0 if empty (before prefill)
    def get_seq_length(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        return self.key_cache[0].shape[-2]

    # Append the new K/V (with RoPE applied) to the past and return the concatenation: attention uses it.
    # At prefill T_new = prompt length, during generation T_new = 1
    def update(
        self,
        key_states: torch.Tensor,      # [B, n_kv, T_new, hd]
        value_states: torch.Tensor,    # [B, n_kv, T_new, hd]
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # First call for this layer (prefill): there is no past, store as is
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # Concatenate on the sequence axis (-2): [B, n_kv, T_past, hd] + [B, n_kv, T_new, hd] -> [B, n_kv, T_past + T_new, hd]
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


# Additive mask: allowed positions 0, blocked positions the smallest number. Token i may only look at j <= i.
# finfo.min instead of -inf: a fully blocked row gives NaN in softmax with -inf, equal weights with finfo.min.
# past_len: number of tokens already in the cache. Rows are the T new tokens, columns past + new (past_len + T).
# New token i sits at position past_len + i: it sees all of the past and the new tokens up to itself
def make_causal_mask(T: int, dtype: torch.dtype, device: torch.device, past_len: int = 0) -> torch.Tensor:
    mask = torch.full((T, past_len + T), torch.finfo(dtype).min, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=past_len + 1)   # j > past_len + i blocked. past_len = 0 gives the square mask
    return mask[None, None]                           # [1, 1, T, past_len + T]: broadcast over batch and heads


# Each token compares its Q with all K's (scores) and takes a weighted sum of the V's.
# Q has 8 heads, K/V 1 head (MQA). With a kv_cache, the past tokens' K/V come from it:
# Q only for the new tokens (T_new), K/V for all past + new tokens (T_all = T_past + T_new)
class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # head_dim is read from the config, not hidden_size / n_heads (they differ in Gemma 2)
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads      # n_rep: how many Q heads share one KV head
        self.scaling = self.head_dim ** -0.5

        # Q gets the full number of heads, K/V fewer. The saving: k_proj and v_proj are small
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        # Its input need not equal hidden_size: num_heads * head_dim (8 * 256 != 2304 in Gemma 2)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,                         # [B, T, D], T = 1 when generating with a cache
        position_embeddings: tuple[torch.Tensor, torch.Tensor],   # (cos, sin), each [B, T, head_dim]: positions of the new tokens
        attention_mask: torch.Tensor | None = None,          # [B, 1, T, T_all], additive mask (allowed 0, blocked -inf)
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        B, T, _ = hidden_states.shape

        # Project, split the last dim into heads, move the head axis forward (heads work independently)
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)      # [B, n_heads, T, hd]
        k = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)   # [B, n_kv, T, hd]
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)   # [B, n_kv, T, hd]

        # RoPE only on Q and K, so position enters the scores; the content carried by V is left alone
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Add the new K/V to the cache and get it back joined with the past. Placement matters: AFTER RoPE (each token is
        # rotated once at its own position and stored), BEFORE repeat_kv (the cache holds n_kv heads, not n_heads)
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, self.layer_idx)      # [B, n_kv, T_all, hd]

        # Temporarily expand K/V to Q's head count (only at compute time)
        k = repeat_kv(k, self.num_kv_groups)                  # [B, n_heads, T_all, hd]
        v = repeat_kv(v, self.num_kv_groups)                  # [B, n_heads, T_all, hd]

        # Scores: [B, n_heads, T, hd] @ [B, n_heads, hd, T_all] -> [B, n_heads, T, T_all]; scores[i, j] = how much new token i attends to token j
        # Without a cache T_all = T and the matrix is square. When generating T = 1: one row, the new token looks at the whole past
        # Divide by sqrt(head_dim): otherwise scores grow with hd and softmax collapses onto a single token
        scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            scores = scores + attention_mask                  # blocked positions -inf -> 0 after softmax
        # Softmax in float32: rounding error accumulates in bf16; the output is cast back to the original dtype
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)    # [B, n_heads, T, T_all], rows sum to 1

        # Weighted sum of V: [B, n_heads, T, T_all] @ [B, n_heads, T_all, hd] -> [B, n_heads, T, hd]
        out = torch.matmul(weights, v)
        # Put the heads back side by side: [B, n_heads, T, hd] -> [B, T, n_heads, hd] -> [B, T, n_heads * hd]
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out)                               # [B, T, n_heads * hd] -> [B, T, D]


# Pre-norm layer: x = x + attn(norm(x)), then x = x + mlp(norm(x)). The residual stream itself is never normalized
class GemmaDecoderLayer(nn.Module): 
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__() 
        self.config = config 
        self.hidden_size = config.hidden_size 
        
        self.self_attn = GemmaAttention(config, layer_idx)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        # Despite the name, this is the norm before the MLP (a name inherited from Llama). In Gemma 2 the same name is applied to the attention output
        self.post_attention_layernorm = GemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps) 
        
    def forward(
        self,
        hidden_states: torch.Tensor,                              # [B, T, D]
        position_embeddings: tuple[torch.Tensor, torch.Tensor],   # (cos, sin), each [B, T, head_dim]
        attention_mask: torch.Tensor | None = None,               # [B or 1, 1, T, T_all], additive mask
        kv_cache: KVCache | None = None,                          # only attention uses it, the MLP does not mix tokens
    ) -> torch.Tensor:
        # Attention block. The residual is saved BEFORE the norm, otherwise the stream itself would get normalized
        residual = hidden_states                                  # [B, T, D]
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask, kv_cache)
        hidden_states = residual + hidden_states 
        
        # MLP block, same pattern
        residual = hidden_states 
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states 
        
        return hidden_states                                      # [B, T, D]
        
class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        # A plain list does not register parameters, and Sequential cannot pass mask/position_ids
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # The residual stream is never normalized across the layers, only once at the end before lm_head
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # cos/sin are the same in every layer (same head_dim and base), so they are computed once here and passed down
        self.rotary_emb = GemmaRotaryEmbedding(config.head_dim, base=config.rope_theta)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        inputs_embeds: torch.Tensor,                           # [B, T, D], T = 1 when generating with a cache
        attention_mask: torch.Tensor | None = None,            # [B or 1, 1, T, T_all], additive mask. Causal if None
        position_ids: torch.Tensor | None = None,              # [B, T]. past_len..past_len+T-1 if None
        scale_embeddings: bool = True,                         # False: input is already at the residual stream's scale (PaliGemma scales it itself)
        kv_cache: KVCache | None = None,                       # if given, layers read the past K/V from it and append the new ones
    ) -> torch.Tensor:                                         # [B, T, D]
        B, T, _ = inputs_embeds.shape
        # Read the past length BEFORE the layers update the cache: it grows by T once layer 0 appends
        past_len = kv_cache.get_seq_length() if kv_cache is not None else 0
        if position_ids is None:
            # New tokens continue where the past left off. Starting from 0 would rotate RoPE at the wrong position
            position_ids = torch.arange(past_len, past_len + T, device=inputs_embeds.device).unsqueeze(0).expand(B, -1)
        if attention_mask is None:
            attention_mask = make_causal_mask(T, inputs_embeds.dtype, inputs_embeds.device, past_len)

        hidden_states = inputs_embeds
        # Bring text embeddings to the residual stream's scale (~1 per element). The scale is in hidden_states' dtype:
        # in bf16 sqrt(2048) rounds to 45.25, and HF does the same. In PaliGemma this is done by paligemma.py
        # (text only: image features already leave the projector at this scale), so it must not be applied again here
        if scale_embeddings:
            normalizer = torch.tensor(self.config.hidden_size ** 0.5, dtype=hidden_states.dtype)
            hidden_states = hidden_states * normalizer

        position_embeddings = self.rotary_emb(hidden_states, position_ids)   # (cos, sin), [B, T, head_dim]
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_embeddings, attention_mask, kv_cache)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class GemmaForCausalLM(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    # The LM head is not a separate matrix: it is the embedding table itself (the same Parameter object)
    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        scale_embeddings: bool = True,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(inputs_embeds, attention_mask, position_ids, scale_embeddings, kv_cache)  # [B, T, hidden_size]
        logits = self.lm_head(hidden_states)       # [B, T, vocab_size]
        return logits.float()
    
`,F=`import torch
from paligemma import PaliGemmaForConditionalGeneration
from gemma import KVCache


# Greedy decoding: each step picks the most likely token, no randomness.
# use_cache=True: image + prompt are processed once (prefill) and their K/V go into the cache. Every later step
# feeds only the last generated token, and SigLIP never runs again. The cost per step is nearly constant.
# use_cache=False: each step recomputes the whole sequence, so the cost grows with its length.
# Both must produce the same tokens; False is kept for comparison and timing
@torch.no_grad()
def generate(
    model: PaliGemmaForConditionalGeneration,
    input_ids: torch.Tensor,        # [B, T] image tokens + <bos> + prompt
    pixel_values: torch.Tensor,     # [B, 3, 224, 224]
    attention_mask: torch.Tensor,   # [B, T]
    max_new_tokens: int,
    eos_token_id: int,
    use_cache: bool = True,
) -> torch.Tensor:                  # [B, <= max_new_tokens], generated tokens (prompt not included)
    model.eval()
    B, prefix_len = input_ids.shape      # the prompt length stays fixed: the mask keeps the prefix bidirectional
    pad_token_id = model.pad_token_id
    kv_cache = KVCache() if use_cache else None

    generated = input_ids.new_empty((B, 0))                          # [B, 0], tokens are appended here
    done = torch.zeros(B, dtype=torch.bool, device=input_ids.device) # samples that produced eos

    # The first call is the same in both modes: the whole image + prompt, all prefix. With a cache this is the prefill that fills it
    logits = model(input_ids, pixel_values, attention_mask, kv_cache=kv_cache)   # [B, prefix_len, vocab]

    for step in range(max_new_tokens):
        # Only the last position's logits give the next token. Earlier positions are for tokens we already know
        next_token = logits[:, -1, :].argmax(dim=-1)                 # [B]
        # Pad finished samples after eos, otherwise the model keeps extending a finished sentence
        next_token = torch.where(done, torch.full_like(next_token, pad_token_id), next_token)
        generated = torch.cat([generated, next_token[:, None]], dim=1)

        done = done | (next_token == eos_token_id)
        # No forward after the last token: its logits would never be used
        if done.all() or step == max_new_tokens - 1:
            break

        if use_cache:
            # Only the new token, no image. The past K/V are in the cache; position and mask follow from the cache length
            logits = model(
                next_token[:, None],                                 # [B, 1]
                None,
                torch.ones(B, 1, dtype=attention_mask.dtype, device=input_ids.device),
                kv_cache=kv_cache,
            )                                                        # [B, 1, vocab]
        else:
            logits = model(
                torch.cat([input_ids, generated], dim=1),            # [B, prefix_len + n]
                pixel_values,
                torch.ones(B, prefix_len + generated.shape[1], dtype=attention_mask.dtype, device=input_ids.device),
                prefix_len=prefix_len,
            )                                                        # [B, prefix_len + n, vocab]

    return generated
`,L=`from dataclasses import dataclass, field
import torch
from torch import nn
from siglip import SiglipVisionConfig, SiglipVisionModel
from gemma import GemmaConfig, GemmaForCausalLM, KVCache, make_causal_mask


# Values from the google/paligemma-3b-pt-224 config.json
@dataclass
class PaliGemmaConfig:
    vision_config: SiglipVisionConfig = field(default_factory=lambda: SiglipVisionConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        patch_size=14,
    ))
    text_config: GemmaConfig = field(default_factory=lambda: GemmaConfig(
        vocab_size=257216,
        hidden_size=2048,
        intermediate_size=16384,
        num_hidden_layers=18,
        num_attention_heads=8,
        num_key_value_heads=1,
        pad_token_id=0,
    ))
    ignore_index: int = -100
    image_token_index: int = 257152
    vocab_size: int = 257216
    projection_dim: int = 2048
    hidden_size: int = 2048
    pad_token_id: int | None = 0

    def __post_init__(self) -> None:
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = self.projection_dim


class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.projection_dim, bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear(image_features)  # [B, 256, 1152] -> [B, 256, 2048]
        return hidden_states


# Prefix-LM mask. Prefix = image tokens + prompt, suffix = the answer the model generates.
# The prefix has full attention within itself (every prefix token sees the whole prefix, including what comes "after": image
# patches and prompt look at each other in both directions). The suffix is causal: each token sees the whole prefix and the tokens before it.
# Rule: token i may look at token j <=> j < prefix_len or j <= i
def make_prefix_lm_mask(
    total_len: int,
    prefix_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:                                         # [1, 1, total_len, total_len], additive mask
    # Start from the causal mask (j > i blocked), then open the prefix columns (j < prefix_len)
    mask = torch.full((total_len, total_len), torch.finfo(dtype).min, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    mask[:, :prefix_len] = 0
    return mask[None, None]


# HF parameter names change between versions, ours are fixed. transformers 5 layout: model.vision_tower.*,
# model.language_model.* (the GemmaModel itself), lm_head.*. The old layout (4.x) already matches ours and is left alone.
# The weights are identical, only the names differ. Returning None skips that weight
def convert_hf_key(key: str) -> str | None:
    # PaliGemma does not use SigLIP's pooling head (probe, attention, mlp): only the encoder output is taken
    if key.startswith(("model.vision_tower.head.", "vision_tower.vision_model.head.")):
        return None
    for old, new in [("model.vision_tower.", "vision_tower.vision_model."),
                     ("model.multi_modal_projector.", "multi_modal_projector."),
                     ("model.language_model.", "language_model.model."),
                     ("lm_head.", "language_model.lm_head.")]:
        if key.startswith(old):
            return new + key[len(old):]
    return key


# Loads an HF state_dict into our model. If the keys do not match one to one it raises instead of silently skipping:
# a missing weight stays random, and the model runs but talks nonsense
def load_hf_weights(model: "PaliGemmaForConditionalGeneration", hf_state_dict: dict[str, torch.Tensor]) -> None:
    converted = {}
    for key, value in hf_state_dict.items():
        new_key = convert_hf_key(key)
        if new_key is not None:
            converted[new_key] = value

    ours_keys, hf_keys = set(model.state_dict()), set(converted)
    if ours_keys != hf_keys:
        raise KeyError(f"state_dict keys do not match. In ours but not in HF: {sorted(ours_keys - hf_keys)[:5]}, "
                       f"in HF but not in ours: {sorted(hf_keys - ours_keys)[:5]}")
    model.load_state_dict(converted)
    # load_state_dict copies values; it does not guarantee lm_head and the embedding are the same Parameter
    model.tie_weights()


class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config.vision_config)
        self.vocab_size = config.vocab_size
        self.language_model = GemmaForCausalLM(config.text_config)
        self.pad_token_id = config.pad_token_id if config.pad_token_id is not None else -1
        self.tie_weights()

    def tie_weights(self) -> None:
        self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
        self,
        image_features: torch.Tensor,   # [B, 256, 2048]
        inputs_embeds: torch.Tensor,    # [B, T, 2048]
        input_ids: torch.Tensor,        # [B, T]
    ) -> torch.Tensor:                  # [B, T, 2048]
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        # Text embeddings are multiplied by sqrt(hidden_size) (Gemma's embedding scale), image features are left unscaled.
        # Dividing image features by sqrt(d) and letting Gemma multiply everything is the same math, but in bf16 it
        # adds two roundings (sqrt(2048) = 45.25 is not a power of 2), which gave a ~0.2 mean logit difference from HF.
        # The scale is in inputs_embeds' dtype: the same operation as HF's scaled embedding table
        normalizer = torch.tensor(self.config.hidden_size ** 0.5, dtype=inputs_embeds.dtype)
        scaled_text_embeds = inputs_embeds * normalizer

        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=dtype, device=device)

        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)  # [B, T]
        image_mask = input_ids == self.config.image_token_index                                      # [B, T]
        pad_mask = input_ids == self.pad_token_id                                                    # [B, T]

        # [B, T] -> [B, T, embed_dim]
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        final_embedding = torch.where(text_mask_expanded, scaled_text_embeds, final_embedding)
        # torch.where does not work here: image features are [B, 256, D], the sequence is [B, T, D]. masked_scatter fills them in order
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, image_features.to(dtype))
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        return final_embedding

    def forward(
        self,
        input_ids: torch.Tensor,                # [B, T]. In a cached generation step only the new token: [B, 1]
        pixel_values: torch.Tensor | None,      # [B, 3, 224, 224]. None in a generation step: the image is already in the cache
        attention_mask: torch.Tensor,           # [B, T]
        prefix_len: int | None = None,          # how many leading tokens are prefix (image + prompt). The whole input if None
        kv_cache: KVCache | None = None,        # empty: prefill (gets filled), non-empty: generation step
    ) -> torch.Tensor:                          # [B, T, vocab_size]
        assert torch.all(attention_mask == 1), "padding is not supported"
        B, T = input_ids.shape

        # Generation step: the cache already holds image + prompt + the answer so far. New tokens are text only,
        # so SigLIP and the projector do not run again (they ran once at prefill, their result is in the cache as K/V)
        past_len = kv_cache.get_seq_length() if kv_cache is not None else 0
        if past_len > 0:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)        # [B, T, 2048]
            # Same as what prefill does to text: the scale is in inputs_embeds' dtype, as in _merge...
            inputs_embeds = inputs_embeds * torch.tensor(self.config.hidden_size ** 0.5, dtype=inputs_embeds.dtype)
            # Positions continue from prefill: prefill used 1..past_len, new tokens start at past_len+1
            position_ids = torch.arange(past_len + 1, past_len + T + 1, device=input_ids.device).unsqueeze(0).expand(B, -1)
            # New tokens are suffix (answer): they see the whole past (prefix + earlier answer) and the new tokens before them.
            # With T = 1 it is [1, 1, 1, past_len + 1] and all zeros. prefix_len is not needed, the whole prefix is in the past
            mask = make_causal_mask(T, inputs_embeds.dtype, inputs_embeds.device, past_len)
            return self.language_model(inputs_embeds, attention_mask=mask, position_ids=position_ids,
                                       scale_embeddings=False, kv_cache=kv_cache)
        # Without a cache, generation appends the answer tokens to the input and calls forward again: the prefix is then
        # the length of the original prompt, not T. On the first call there is no answer, so the whole input is prefix
        if prefix_len is None:
            prefix_len = T

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)       # [B, T, 2048]
        image_features = self.vision_tower(pixel_values.to(inputs_embeds.dtype))    # [B, 256, 1152]
        image_features = self.multi_modal_projector(image_features)                 # [B, 256, 2048]
        inputs_embeds = self._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids)

        # Positions start at 1 (as in HF PaliGemma). RoPE only depends on position differences, so starting at 0 would
        # give the same result mathematically, but we match HF to stay bit-exact
        position_ids = torch.arange(1, T + 1, device=input_ids.device).unsqueeze(0).expand(B, -1)   # [B, T]
        mask = make_prefix_lm_mask(T, prefix_len, inputs_embeds.dtype, inputs_embeds.device)         # [1, 1, T, T]

        # Text is already scaled here (_merge...), so GemmaModel must not multiply again.
        # With a cache this is the prefill: every layer writes the image + prompt K/V into it
        logits = self.language_model(inputs_embeds, attention_mask=mask, position_ids=position_ids,
                                     scale_embeddings=False, kv_cache=kv_cache)
        return logits
`,M=`from typing import Dict, Iterable, List, Optional, Union, Tuple
from PIL import Image
import numpy as np 
import torch 

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

def add_image_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token): 
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\\n"

def resize(
    image: Image, 
    size: Tuple[int, int], 
    resample: Image.Resampling = None,
    reducing_gap: Optional[int] = None
) -> Image: 
    height, width = size 
    resize_image = image.resize((width, height), resample=resample, reducing_gap=reducing_gap)
    
    return resize_image

def rescale(image: np.ndarray, scale: float, dtype: np.dtype = np.float32) -> np.ndarray: 
    rescaled_image = image * scale 
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image 

def normalize(image: np.ndarray, mean: Union[float, Iterable[float]], std: Union[Iterable[float]]) -> np.ndarray: 
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    normalized_image = (image - mean) / std 
    return normalized_image 

def process_images(
    images: List[Image.Image], 
    size: Tuple[int, int] = None,
    resample: Image.Resampling = None, 
    rescale_factor: float = None, 
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None
) -> List[np.ndarray]: 
    height, width = size[0], size[1]
    images = [resize(image=image, size=[height, width], resample=resample) for image in images]
    images = [np.array(image) for image in images]
    images = [rescale(image, scale=rescale_factor) for image in images]
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    images = [image.transpose(2, 0, 1) for image in images]
    return images 

class PaliGemmaProcessor: 
    
    IMAGE_TOKEN = "<image>"
    
    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        self.image_seq_length = num_image_tokens
        self.image_size = image_size 
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        self.tokenizer = tokenizer
        
    def __call__(self, 
                text: List[str], 
                images: List[Image.Image], 
                padding: str = "longest",
                truncation: bool = True,
                add_special_tokens: bool = False 
                ) -> dict:
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts."
        
        pixel_values = process_images(
            images, 
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD
        )   
        
        
        pixel_values = np.stack(pixel_values, axis=0)
        pixel_values = torch.tensor(pixel_values)

        input_strings = [
            add_image_to_prompt(
                prefix_prompt=prompt, 
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN
            )
            for prompt in text
        ]
        
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
            add_special_tokens=add_special_tokens
        )
        
        return_data = {"pixel_values": pixel_values, **inputs}
        return return_data`,B=`from dataclasses import dataclass
import torch
from torch import nn 
import torch.nn.functional as F

@dataclass
class SiglipVisionConfig: 
    num_channels: int = 3 
    hidden_size: int = 768 
    image_size: int = 224 
    patch_size: int = 16 
    intermediate_size: int = 3072 
    num_hidden_layers: int = 12 
    num_attention_heads: int = 12 
    layer_norm_eps: float = 1e-6 
    attention_dropout: float = 0.0 
    projection_dim: int = 2048
    
class SiglipVisionEmbeddings(nn.Module): 
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config 
        
        self.num_channels = config.num_channels
        self.hidden_size = config.hidden_size
        self.image_size = config.image_size 
        self.patch_size = config.patch_size
        
        self.patch_embedding = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid"
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2 
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(
            num_embeddings=self.num_positions, 
            embedding_dim=self.hidden_size
        )
        self.register_buffer(
            name="position_ids", 
            tensor=torch.arange(self.num_positions).expand(1, -1),
            persistent=False 
        )
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor: 
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(start_dim=2, end_dim=3)
        embeddings = embeddings.transpose(1, 2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings
        
class SiglipAttention(nn.Module): 
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config 
        
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout 
        
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: 
        B, T, C = hidden_states.shape 
        k_states = self.k_proj(hidden_states) # [B, 196, 768]
        q_states = self.q_proj(hidden_states) # [B, 196, 768]
        v_states = self.v_proj(hidden_states) # [B, 196, 768] 
        
        k_states = k_states.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # [B, 12, 196, 64]
        q_states = q_states.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # [B, 12, 196, 64]
        v_states = v_states.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # [B, 12, 196, 64]
        
        wei = (q_states @ k_states.transpose(-2, -1)) * self.scale # [B, 12, 196, 196]
        wei = F.softmax(wei, dim=-1)
        wei = F.dropout(wei, p=self.dropout, training=self.training)
        
        out = wei @ v_states  # [B, 12, 196, 64]
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(B, T, C)
        out = self.out_proj(out)
        return out 
        
        
class SiglipMLP(nn.Module): 
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: 
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        return hidden_states
    
class SiglipEncoderLayer(nn.Module): 
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config 
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = SiglipAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: 
        residual = hidden_states 
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states 
        
        residual = hidden_states 
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states 
        
class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config 
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: 
        for layer in self.layers: 
            hidden_states = layer(hidden_states)
        return hidden_states 
    
class SiglipVisionTransformer(nn.Module): 
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config 
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor: 
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state

class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision_model(pixel_values)  # [B, num_patches, hidden_size]
        `,p=Object.assign({"../../../checks/bench_kv_cache.py":y,"../../../checks/check_embeddings.py":b,"../../../checks/check_encoder.py":v,"../../../checks/check_gemma.py":x,"../../../checks/check_paligemma.py":w,"../../../checks/check_processing.py":T,"../../../checks/check_projector.py":z,"../../../src/gemma.py":G,"../../../src/generate.py":F,"../../../src/paligemma.py":L,"../../../src/processing.py":M,"../../../src/siglip.py":B}),u=Object.fromEntries(Object.entries(p).map(([e,t])=>[e.split("/").pop(),t])),P=Object.fromEntries(Object.keys(p).map(e=>{const t=e.split("/");return[t.pop(),t.pop()]})),m=e=>e.length-e.trimStart().length;function A(e,t,d=!0){const _=u[e];if(!_)return{file:e,start:0,end:0,lines:[`# ${e} not found`],found:!1};const n=_.split(`
`),g=t.split(".");let c=0,r=n.length,l=-1,s=-1;for(const h of g){const k=new RegExp(`^\\s*(class|def)\\s+${h}\\b`);s=-1;for(let a=c;a<r;a++)if(k.test(n[a])&&(l<0||m(n[a])>l)){s=a;break}if(s<0)return{file:e,start:0,end:0,lines:[`# ${t} not found in ${e}`],found:!1};l=m(n[s]);let i=s+1,f=!/:\s*(#.*)?$/.test(n[s]);for(;i<r;){const a=n[i];if(f){/:\s*(#.*)?$/.test(a)&&(f=!1),i++;continue}if(a.trim()===""||m(a)>l){i++;continue}break}for(;i>s+1&&n[i-1].trim()==="";)i--;c=s,r=i}let o=s;if(d)for(;o>0&&n[o-1].trim().startsWith("#")&&m(n[o-1])===l;)o--;return{file:e,start:o+1,end:r,lines:n.slice(o,r).map(h=>h.replace(/\s+$/,"")),found:!0}}function N(e,t){const d=(u[e]??"").split(`
`).findIndex(_=>_.includes(t));return d<0?-1:d+1}export{P as D,u as S,A as f,N as l};

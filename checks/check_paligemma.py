"""Do the PaliGemma parts work correctly?

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
    print(f"\nResult: {'everything works as expected ✅' if ok else 'mismatch ❌'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

import torch
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

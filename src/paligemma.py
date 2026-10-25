from dataclasses import dataclass, field
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

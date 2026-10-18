from dataclasses import dataclass, field
import torch
from torch import nn
from siglip import SiglipVisionConfig, SiglipVisionModel
from gemma import GemmaConfig, GemmaForCausalLM


# Değerler google/paligemma-3b-pt-224 config.json'dan
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

        # Gemma ileride bütün embedding'leri sqrt(hidden_size) ile çarpacak, görsel özellikler bundan etkilenmesin
        scaled_image_features = image_features / (self.config.hidden_size ** 0.5)

        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=dtype, device=device)

        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)  # [B, T]
        image_mask = input_ids == self.config.image_token_index                                      # [B, T]
        pad_mask = input_ids == self.pad_token_id                                                    # [B, T]

        # [B, T] -> [B, T, embed_dim]
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
        # torch.where kullanılamaz: görsel özellikler [B, 256, D], dizi [B, T, D]. masked_scatter sırayla yerleştirir
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features.to(dtype))
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        return final_embedding

    def forward(
        self,
        input_ids: torch.Tensor,        # [B, T]
        pixel_values: torch.Tensor,     # [B, 3, 224, 224]
        attention_mask: torch.Tensor,   # [B, T]
    ) -> torch.Tensor:                  # [B, T, vocab_size]
        assert torch.all(attention_mask == 1), "padding desteklenmiyor"

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)       # [B, T, 2048]
        image_features = self.vision_tower(pixel_values.to(inputs_embeds.dtype))    # [B, 256, 1152]
        image_features = self.multi_modal_projector(image_features)                 # [B, 256, 2048]
        inputs_embeds = self._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids)

        logits = self.language_model(inputs_embeds)
        return logits

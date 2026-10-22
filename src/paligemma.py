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


# Prefix-LM maskesi. Prefix = görüntü token'ları + komut (prompt), suffix = modelin ürettiği cevap.
# Prefix kendi içinde tam attention (her prefix token'ı bütün prefix'i görüyor, "sonrası" da dahil: görüntü
# yamaları ve komut birbirine iki yönlü bakıyor). Suffix causal: her token prefix'in tamamını ve kendinden öncekileri görüyor.
# Kural: i. token j. token'a bakabilir <=> j < prefix_len veya j <= i
def make_prefix_lm_mask(
    total_len: int,
    prefix_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:                                         # [1, 1, total_len, total_len], toplanan maske
    # Önce nedensel maske (j > i yasak), sonra prefix sütunlarını (j < prefix_len) açıyoruz
    mask = torch.full((total_len, total_len), torch.finfo(dtype).min, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    mask[:, :prefix_len] = 0
    return mask[None, None]


# HF'nin parametre adları sürüme göre değişiyor, bizimki sabit. transformers 5 düzeni: model.vision_tower.*,
# model.language_model.* (GemmaModel doğrudan), lm_head.*. Eski düzen (4.x) zaten bizimkiyle aynı, dokunulmuyor.
# Ağırlıkların kendisi aynı, sadece adlar farklı. None dönerse o ağırlık atlanıyor
def convert_hf_key(key: str) -> str | None:
    # PaliGemma SigLIP'in pooling head'ini (probe, attention, mlp) kullanmıyor: sadece encoder'ın çıktısı alınıyor
    if key.startswith(("model.vision_tower.head.", "vision_tower.vision_model.head.")):
        return None
    for old, new in [("model.vision_tower.", "vision_tower.vision_model."),
                     ("model.multi_modal_projector.", "multi_modal_projector."),
                     ("model.language_model.", "language_model.model."),
                     ("lm_head.", "language_model.lm_head.")]:
        if key.startswith(old):
            return new + key[len(old):]
    return key


# HF state_dict'ini bizim modele yükler. Anahtarlar bire bir tutmazsa sessizce atlamak yerine hata verir:
# eksik bir ağırlık rastgele kalırsa model çalışır ama saçmalar
def load_hf_weights(model: "PaliGemmaForConditionalGeneration", hf_state_dict: dict[str, torch.Tensor]) -> None:
    converted = {}
    for key, value in hf_state_dict.items():
        new_key = convert_hf_key(key)
        if new_key is not None:
            converted[new_key] = value

    ours_keys, hf_keys = set(model.state_dict()), set(converted)
    if ours_keys != hf_keys:
        raise KeyError(f"state_dict anahtarları tutmuyor. Bizde olup HF'de olmayan: {sorted(ours_keys - hf_keys)[:5]}, "
                       f"HF'de olup bizde olmayan: {sorted(hf_keys - ours_keys)[:5]}")
    model.load_state_dict(converted)
    # load_state_dict değerleri kopyalar, lm_head ile embedding'in aynı Parameter olmasını garanti etmez
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

        # Metin embedding'leri sqrt(hidden_size) ile çarpılıyor (Gemma'nın embedding ölçeği), görsel özellikler ölçeksiz.
        # Eskiden görseller önce sqrt(d)'ye bölünüp Gemma hepsini birden çarpıyordu: matematiksel olarak aynı ama bf16'da
        # iki yuvarlama ekliyordu (sqrt(2048) = 45.25 2'nin kuvveti değil), HF'ye göre logit'lerde ~0.2 ortalama fark çıkıyordu.
        # Ölçek inputs_embeds'in dtype'ında: HF'nin ölçekli embedding tablosuyla aynı işlem
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
        # torch.where kullanılamaz: görsel özellikler [B, 256, D], dizi [B, T, D]. masked_scatter sırayla yerleştirir
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, image_features.to(dtype))
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        return final_embedding

    def forward(
        self,
        input_ids: torch.Tensor,        # [B, T]
        pixel_values: torch.Tensor,     # [B, 3, 224, 224]
        attention_mask: torch.Tensor,   # [B, T]
        prefix_len: int | None = None,  # ilk kaç token prefix (görüntü + komut). Verilmezse bütün girdi prefix
    ) -> torch.Tensor:                  # [B, T, vocab_size]
        assert torch.all(attention_mask == 1), "padding desteklenmiyor"
        B, T = input_ids.shape
        # Üretimde (aşama 6'nın sonu) girdiye cevap token'ları eklenip forward tekrar çağrılacak: prefix o zaman
        # ilk komutun uzunluğu, T değil. İlk çağrıda cevap yok, bütün girdi prefix
        if prefix_len is None:
            prefix_len = T

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)       # [B, T, 2048]
        image_features = self.vision_tower(pixel_values.to(inputs_embeds.dtype))    # [B, 256, 1152]
        image_features = self.multi_modal_projector(image_features)                 # [B, 256, 2048]
        inputs_embeds = self._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids)

        # Konumlar 1'den başlıyor (HF PaliGemma böyle). RoPE yalnızca konum farkına bağlı olduğu için 0'dan başlasa da
        # sonuç matematiksel olarak aynı olurdu, ama HF ile birebir kalmak için aynısını yapıyoruz
        position_ids = torch.arange(1, T + 1, device=input_ids.device).unsqueeze(0).expand(B, -1)   # [B, T]
        mask = make_prefix_lm_mask(T, prefix_len, inputs_embeds.dtype, inputs_embeds.device)         # [1, 1, T, T]

        # Metin burada zaten ölçeklendi (_merge...), GemmaModel tekrar çarpmasın
        logits = self.language_model(inputs_embeds, attention_mask=mask, position_ids=position_ids, scale_embeddings=False)
        return logits

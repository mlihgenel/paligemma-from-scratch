import torch
from paligemma import PaliGemmaForConditionalGeneration


# Greedy decode: her adımda en yüksek olasılıklı token seçiliyor, rastgelelik yok.
# KV cache olmadığı için her adımda bütün dizi (görüntü + komut + şimdiye kadar üretilen) baştan hesaplanıyor.
# Adım başına maliyet dizi uzunluğuyla büyüyor, aşama 7'de cache bunu düzeltecek
@torch.no_grad()
def generate(
    model: PaliGemmaForConditionalGeneration,
    input_ids: torch.Tensor,        # [B, T] görüntü token'ları + <bos> + komut
    pixel_values: torch.Tensor,     # [B, 3, 224, 224]
    attention_mask: torch.Tensor,   # [B, T]
    max_new_tokens: int,
    eos_token_id: int,
) -> torch.Tensor:                  # [B, <= max_new_tokens], üretilen token'lar (komut dahil değil)
    model.eval()
    B, prefix_len = input_ids.shape      # komutun uzunluğu sabit kalıyor: maske prefix'i hep iki yönlü tutacak
    pad_token_id = model.pad_token_id

    generated = input_ids.new_empty((B, 0))                          # [B, 0], buraya token eklenecek
    done = torch.zeros(B, dtype=torch.bool, device=input_ids.device) # eos üreten örnekler

    for _ in range(max_new_tokens):
        logits = model(
            torch.cat([input_ids, generated], dim=1),                # [B, prefix_len + n]
            pixel_values,
            torch.ones(B, prefix_len + generated.shape[1], dtype=attention_mask.dtype, device=input_ids.device),
            prefix_len=prefix_len,
        )                                                            # [B, prefix_len + n, vocab]

        # Sadece son konumun logit'i bir sonraki token'ı veriyor. Önceki konumlar zaten bilinen token'lar için
        next_token = logits[:, -1, :].argmax(dim=-1)                 # [B]
        # Bitmiş örneklerde eos'tan sonrasını pad ile doldur, yoksa model bitmiş cümleyi sürdürür
        next_token = torch.where(done, torch.full_like(next_token, pad_token_id), next_token)
        generated = torch.cat([generated, next_token[:, None]], dim=1)

        done = done | (next_token == eos_token_id)
        if done.all():
            break

    return generated

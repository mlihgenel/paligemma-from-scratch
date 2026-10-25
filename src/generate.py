import torch
from paligemma import PaliGemmaForConditionalGeneration
from gemma import KVCache


# Greedy decode: her adımda en yüksek olasılıklı token seçiliyor, rastgelelik yok.
# use_cache=True: görüntü + komut bir kez işleniyor (prefill), K/V'leri cache'e yazılıyor. Sonraki her adımda
# modele sadece son üretilen token giriyor, SigLIP bir daha çalışmıyor. Adım başına maliyet neredeyse sabit.
# use_cache=False: her adımda bütün dizi baştan hesaplanıyor, maliyet dizi uzunluğuyla büyüyor.
# İkisi aynı token'ları üretmeli, False karşılaştırma ve süre ölçümü için duruyor
@torch.no_grad()
def generate(
    model: PaliGemmaForConditionalGeneration,
    input_ids: torch.Tensor,        # [B, T] görüntü token'ları + <bos> + komut
    pixel_values: torch.Tensor,     # [B, 3, 224, 224]
    attention_mask: torch.Tensor,   # [B, T]
    max_new_tokens: int,
    eos_token_id: int,
    use_cache: bool = True,
) -> torch.Tensor:                  # [B, <= max_new_tokens], üretilen token'lar (komut dahil değil)
    model.eval()
    B, prefix_len = input_ids.shape      # komutun uzunluğu sabit kalıyor: maske prefix'i hep iki yönlü tutacak
    pad_token_id = model.pad_token_id
    kv_cache = KVCache() if use_cache else None

    generated = input_ids.new_empty((B, 0))                          # [B, 0], buraya token eklenecek
    done = torch.zeros(B, dtype=torch.bool, device=input_ids.device) # eos üreten örnekler

    # İlk çağrı iki durumda da aynı: görüntü + komutun tamamı, hepsi prefix. Cache varsa bu prefill, cache'i dolduruyor
    logits = model(input_ids, pixel_values, attention_mask, kv_cache=kv_cache)   # [B, prefix_len, vocab]

    for step in range(max_new_tokens):
        # Sadece son konumun logit'i bir sonraki token'ı veriyor. Önceki konumlar zaten bilinen token'lar için
        next_token = logits[:, -1, :].argmax(dim=-1)                 # [B]
        # Bitmiş örneklerde eos'tan sonrasını pad ile doldur, yoksa model bitmiş cümleyi sürdürür
        next_token = torch.where(done, torch.full_like(next_token, pad_token_id), next_token)
        generated = torch.cat([generated, next_token[:, None]], dim=1)

        done = done | (next_token == eos_token_id)
        # Son token'dan sonra bir forward daha gereksiz: logit'i kullanılmayacak
        if done.all() or step == max_new_tokens - 1:
            break

        if use_cache:
            # Sadece yeni token, görüntüsüz. Geçmişin K/V'si cache'te, konum ve maske cache uzunluğundan çıkıyor
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

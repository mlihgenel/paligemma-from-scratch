# PaliGemma, sıfırdan

Google DeepMind'ın görsel-dil modeli PaliGemma'nın mimarisini PyTorch ile sıfırdan yazıyorum: görüntü kodlayıcı (SigLIP), görüntüyü dil modeline bağlayan projeksiyon katmanı ve dil modeli (Gemma). Bu çalışma, sıfırdan GPT ve BPE tokenizer yazdığım [yz50](https://github.com/mlihgenel/yz50)'nin devamı. PaliGemma 2 makalesinin incelemesi ve Türkçe tabelalarla yaptığım deney [paper-to-practice](https://github.com/mlihgenel/paper-to-practice/tree/main/papers/01-paligemma2) reposunda.

**Yöntem:** Model eğitilmiyor. Mimariyi kendim yazıyorum, Google'ın yayımladığı ağırlıkları kendi modüllerime yüklüyorum ve her parçanın çıktısını Hugging Face `transformers` implementasyonuyla karşılaştırıyorum. Çıktılar aynıysa mimari doğru yazılmış demektir.

**Sıra:** Önce PaliGemma 1 mimarisi (SigLIP + linear projeksiyon + Gemma). Ardından KV cache ve Gemma 2'ye özgü kısımlar (soft-capping, sliding window, sandwich norm) eklenerek PaliGemma 2. PaliGemma 2 makalesine göre iki model arasındaki tek mimari fark dil modeli.

## İlerleme

| # | Aşama | Durum |
|---|---|---|
| 1 | SigLIP: yama embedding'leri | ✅ |
| 2 | SigLIP: attention, MLP, encoder (base → so400m) | 🔄 attention ✅ |
| 3 | Processor + projeksiyon | ⬜ |
| 4 | Gemma: RMSNorm, GeGLU, weight tying | ⬜ |
| 5 | Gemma attention: GQA + RoPE | ⬜ |
| 6 | Uçtan uca PaliGemma 1 (`paligemma-3b-pt-224`) | ⬜ |
| 7 | KV cache | ⬜ |
| 8 | Gemma 2 → PaliGemma 2 (`paligemma2-3b-mix-224`) | ⬜ |

## Dosyalar

| Dosya | İçerik |
|---|---|
| `src/siglip.py` | SigLIP görüntü kodlayıcı: config, embeddings, attention, … |
| `checks/check_embeddings.py` | Embeddings'i HF ile karşılaştırır |
| `checks/check_encoder.py` | Attention, MLP, encoder katmanı, encoder ve modelin tamamını HF ile karşılaştırır. Henüz yazılmamış parçaları atlar |

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python checks/check_encoder.py                                    # siglip-base-patch16-224
.venv/bin/python checks/check_encoder.py google/siglip-so400m-patch14-224  # PaliGemma'nın kodlayıcısı
```

## Kaynaklar

- Tunga Bayrak, [Writing a Transformer Model from SCRATCH with PYTORCH](https://www.youtube.com/watch?v=r8y7gXIpb4A): SigLIP
- Umar Jamil, [Coding a Multimodal (Vision) Language Model from scratch in PyTorch](https://www.youtube.com/watch?v=vAmKB7iPkWw) · [kod](https://github.com/hkproj/pytorch-paligemma): PaliGemma 1'in tamamı
- Umar Jamil, [LLaMA explained](https://www.youtube.com/watch?v=Mn_9W1nCFLo): RMSNorm, RoPE, KV cache, GQA
- Hugging Face `transformers`: [`siglip`](https://github.com/huggingface/transformers/tree/main/src/transformers/models/siglip), [`gemma`](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma), [`gemma2`](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma2), [`paligemma`](https://github.com/huggingface/transformers/tree/main/src/transformers/models/paligemma)
- Makaleler: [ViT](https://arxiv.org/abs/2010.11929), [SigLIP](https://arxiv.org/abs/2303.15343), [RoFormer](https://arxiv.org/abs/2104.09864), [Gemma](https://arxiv.org/abs/2403.08295), [Gemma 2](https://arxiv.org/abs/2408.00118), [PaliGemma](https://arxiv.org/abs/2407.07726), [PaliGemma 2](https://arxiv.org/abs/2412.03555)

Model ağırlıkları Google'a aittir ve [Gemma kullanım koşullarına](https://ai.google.dev/gemma/terms) tabidir. Ağırlıklar bu repoya dahil değildir, Hugging Face'ten indirilir.

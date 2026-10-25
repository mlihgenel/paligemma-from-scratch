// Stops of the end-to-end tour: each has a shape, dtype, class, code line and chapter link.
import { ReactNode } from "react";
import type { L } from "../../i18n";

export const N = 16;
export const NP = N * N;

// Real and small (check_gemma SMALL + a small siglip-base-like) dimensions
export const DIMS = {
  real: { V: 1152, D: 2048, I: 16384, Vc: 257216, P: 256, side: 16, patch: 14, enc: 27, dec: 18 },
  small: { V: 64, D: 64, I: 256, Vc: 128, P: 256, side: 16, patch: 14, enc: 2, dec: 2 },
};

export interface StageDef {
  key: string; title: L<string>; z: number;
  shape: (d: typeof DIMS.real, T: number) => number[]; ax: string; dtype: string;
  cls: string; code?: [string, string]; to: string; note: L<ReactNode>;
}

export const STAGES: StageDef[] = [
  { key: "raw", title: { en: "Raw photo", tr: "Ham fotoğraf" }, z: 0, shape: () => [0, 0, 3], ax: "H,W,C", dtype: "uint8", cls: "PIL.Image", code: ["processing.py", "images = [resize(image=image"], to: "/patches",
    note: { en: <>The photo at its own size. The model has not seen anything yet.</>, tr: <>Fotoğraf, kendi boyutunda. Model henüz hiçbir şey görmedi.</> } },
  { key: "pre", title: { en: "Preprocessing", tr: "Ön işleme" }, z: 0, shape: () => [1, 3, 224, 224], ax: "B,C,H,W", dtype: "float32", cls: "process_images", code: ["processing.py", "images = [normalize(image"], to: "/patches",
    note: { en: <>Bicubic resize to 224×224, ÷255, then (x − 0.5)/0.5 into the range [−1, 1], HWC → CHW.</>, tr: <>224×224'e bicubic resize, ÷255, sonra (x − 0.5)/0.5 ile [−1, 1] aralığı, HWC → CHW.</> } },
  { key: "patch", title: { en: "Patches: Conv2d (stride 14)", tr: "Yamalar: Conv2d (stride 14)" }, z: 0, shape: (d) => [1, d.V, d.side, d.side], ax: "B,D,H,W", dtype: "float32", cls: "SiglipVisionEmbeddings · Conv2d", code: ["siglip.py", "patch_embeds = self.patch_embedding(pixel_values)"], to: "/patches#conv",
    note: { en: <>16×16 = 256 non-overlapping patches. The same 14·14·3 → 1152 linear map is applied to each patch.</>, tr: <>16×16 = 256 üst üste binmeyen yama. Her yamaya aynı 14·14·3 → 1152 linear dönüşüm.</> } },
  { key: "tokens", title: { en: "flatten + transpose → token sequence", tr: "flatten + transpose → token dizisi" }, z: 0, shape: (d) => [1, d.P, d.V], ax: "B,T,D", dtype: "float32", cls: "SiglipVisionEmbeddings", code: ["siglip.py", "embeddings = embeddings.transpose(1, 2)"], to: "/patches#conv",
    note: { en: <>The patches are laid out row by row into a sequence of 256 tokens. The image is now a "sentence".</>, tr: <>Yamalar satır satır dizilip 256 token'lık bir diziye dönüşüyor. Artık görüntü bir "cümle".</> } },
  { key: "pos", title: { en: "+ position embedding", tr: "+ konum embedding'i" }, z: 0, shape: (d) => [1, d.P, d.V], ax: "B,T,D", dtype: "float32", cls: "nn.Embedding(256, 1152)", code: ["siglip.py", "embeddings = embeddings + self.position_embedding"], to: "/patches#conv",
    note: { en: <>A learned vector for its position is added to each token. Without it the model could not know where a patch is.</>, tr: <>Her token'a konumuna ait öğrenilmiş bir vektör ekleniyor. Yoksa model yamaların yerini bilemezdi.</> } },
  { key: "enc", title: { en: "SigLIP encoder ×27 + post_layernorm", tr: "SigLIP encoder ×27 + post_layernorm" }, z: -3.2, shape: (d) => [1, d.P, d.V], ax: "B,T,D", dtype: "float32", cls: "SiglipEncoder", code: ["siglip.py", "last_hidden_state = self.post_layernorm"], to: "/encoder",
    note: { en: <>Attention without a mask: every patch sees every patch. The shape never changes; the content gets richer with "what is this patch part of?".</>, tr: <>Maskesiz attention: her yama her yamayı görüyor. Şekil hiç değişmiyor, içerik "bu yama neyin parçası?" bilgisiyle zenginleşiyor.</> } },
  { key: "proj", title: { en: "Projector: 1152 → 2048", tr: "Projektör: 1152 → 2048" }, z: -5.6, shape: (d) => [1, d.P, d.D], ax: "B,T,D", dtype: "float32", cls: "PaliGemmaMultiModalProjector", code: ["paligemma.py", "hidden_states = self.linear(image_features)"], to: "/bridge",
    note: { en: <>A single Linear (with bias). The image tokens are moved to the language model's size.</>, tr: <>Tek bir Linear (bias'lı). Görsel token'lar dil modelinin boyutuna taşınıyor.</> } },
  { key: "merge", title: { en: "Text and image merge", tr: "Metin + görüntü birleşiyor" }, z: -7.6, shape: (d, T) => [1, T, d.D], ax: "B,T,D", dtype: "float32", cls: "masked_scatter", code: ["paligemma.py", "final_embedding = final_embedding.masked_scatter"], to: "/bridge#merge",
    note: { en: <>Prompt tokens come from the embedding table and are multiplied by √2048 (45.25) (green). The image features are written into the &lt;image&gt; positions in order, <b>unscaled</b>.</>, tr: <>Prompt token'ları embedding tablosundan geliyor ve √2048 (45.25) ile çarpılıyor (yeşil). &lt;image&gt; konumlarına görsel özellikler <b>ölçeksiz</b>, sırayla yazılıyor.</> } },
  { key: "dec", title: { en: "Gemma decoder ×18", tr: "Gemma decoder ×18" }, z: -10.6, shape: (d, T) => [1, T, d.D], ax: "B,T,D", dtype: "float32", cls: "GemmaDecoderLayer", code: ["gemma.py", "hidden_states = decoder_layer(hidden_states, position_embeddings"], to: "/gemma/residual",
    note: { en: <>Each layer adds two updates to the stream (attn + mlp). Attention is the only place where tokens see each other: the image and prompt see each other fully (prefix), the answer looks only at earlier tokens.</>, tr: <>Her katman akışa iki güncelleme ekliyor (attn + mlp). Token'ların birbirini gördüğü tek yer attention: görüntü ve komut birbirini tam görüyor (prefix), cevap sadece öncekilere bakıyor.</> } },
  { key: "norm", title: { en: "Final RMSNorm", tr: "Son RMSNorm" }, z: -13, shape: (d, T) => [1, T, d.D], ax: "B,T,D", dtype: "float32", cls: "GemmaRMSNorm", code: ["gemma.py", "hidden_states = self.norm(hidden_states)"], to: "/gemma/rmsnorm",
    note: { en: <>The stream, which grew over 18 layers, is normalized once before lm_head.</>, tr: <>18 katman boyunca büyüyen akış, lm_head'den önce bir kez normalize ediliyor.</> } },
  { key: "head", title: { en: "lm_head → logits", tr: "lm_head → logit'ler" }, z: -15, shape: (d, T) => [1, T, d.Vc], ax: "B,T,V", dtype: "float32", cls: "nn.Linear (= embed_tokens.weight)", code: ["gemma.py", "logits = self.lm_head(hidden_states)"], to: "/gemma/tying",
    note: { en: <>For every position, a score for each of the 257216 tokens in the vocabulary. Generation only uses the scores of the <b>last</b> position: argmax → the next token.</>, tr: <>Her konum için sözlükteki 257216 token'a bir skor. Üretimde sadece <b>son</b> konumun skorları kullanılıyor: argmax → bir sonraki token.</> } },
];

export const idx = (k: string) => STAGES.findIndex((s) => s.key === k);

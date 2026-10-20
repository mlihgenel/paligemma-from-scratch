// Oyun alanı durakları: her durakta şekil, dtype, sınıf, kod satırı ve bölüm linki.
import { ReactNode } from "react";

export const N = 16;
export const NP = N * N;

// Gerçek ve küçük (check_gemma SMALL + siglip-base benzeri küçük) boyutlar
export const DIMS = {
  real: { V: 1152, D: 2048, I: 16384, Vc: 257216, P: 256, side: 16, patch: 14, enc: 27, dec: 18 },
  small: { V: 64, D: 64, I: 256, Vc: 128, P: 256, side: 16, patch: 14, enc: 2, dec: 2 },
};

export interface StageDef {
  key: string; title: string; z: number;
  shape: (d: typeof DIMS.real, T: number) => number[]; ax: string; dtype: string;
  cls: string; code?: [string, string]; to: string; note: ReactNode;
}

export const STAGES: StageDef[] = [
  { key: "raw", title: "Ham fotoğraf", z: 0, shape: () => [0, 0, 3], ax: "H,W,C", dtype: "uint8", cls: "PIL.Image", code: ["processing.py", "images = [resize(image=image"], to: "/goruntu",
    note: <>Deneyindeki fotoğraf, kendi boyutunda. Model henüz hiçbir şey görmedi.</> },
  { key: "pre", title: "Ön işleme", z: 0, shape: () => [1, 3, 224, 224], ax: "B,C,H,W", dtype: "float32", cls: "process_images", code: ["processing.py", "images = [normalize(image"], to: "/goruntu",
    note: <>224×224'e bicubic resize, ÷255, sonra (x − 0.5)/0.5 ile [−1, 1] aralığı, HWC → CHW.</> },
  { key: "patch", title: "Yamalar: Conv2d (stride 14)", z: 0, shape: (d) => [1, d.V, d.side, d.side], ax: "B,D,H,W", dtype: "float32", cls: "SiglipVisionEmbeddings · Conv2d", code: ["siglip.py", "patch_embeds = self.patch_embedding(pixel_values)"], to: "/goruntu#yama",
    note: <>16×16 = 256 üst üste binmeyen yama. Her yamaya aynı 14·14·3 → 1152 linear dönüşüm.</> },
  { key: "tokens", title: "flatten + transpose → token dizisi", z: 0, shape: (d) => [1, d.P, d.V], ax: "B,T,D", dtype: "float32", cls: "SiglipVisionEmbeddings", code: ["siglip.py", "embeddings = embeddings.transpose(1, 2)"], to: "/goruntu#yama",
    note: <>Yamalar satır satır dizilip 256 token'lık bir diziye dönüşüyor. Artık görüntü bir "cümle".</> },
  { key: "pos", title: "+ konum embedding'i", z: 0, shape: (d) => [1, d.P, d.V], ax: "B,T,D", dtype: "float32", cls: "nn.Embedding(256, 1152)", code: ["siglip.py", "embeddings = embeddings + self.position_embedding"], to: "/goruntu#yama",
    note: <>Her token'a konumuna ait öğrenilmiş bir vektör ekleniyor. Yoksa model yamaların yerini bilemezdi.</> },
  { key: "enc", title: "SigLIP encoder ×27 + post_layernorm", z: -3.2, shape: (d) => [1, d.P, d.V], ax: "B,T,D", dtype: "float32", cls: "SiglipEncoder", code: ["siglip.py", "last_hidden_state = self.post_layernorm"], to: "/encoder",
    note: <>Maskesiz attention: her yama her yamayı görüyor. Şekil hiç değişmiyor, içerik "bu yama neyin parçası?" bilgisiyle zenginleşiyor.</> },
  { key: "proj", title: "Projektör: 1152 → 2048", z: -5.6, shape: (d) => [1, d.P, d.D], ax: "B,T,D", dtype: "float32", cls: "PaliGemmaMultiModalProjector", code: ["paligemma.py", "hidden_states = self.linear(image_features)"], to: "/kopru",
    note: <>Tek bir Linear (bias'lı). Görsel token'lar dil modelinin boyutuna taşınıyor.</> },
  { key: "div", title: "÷ √2048", z: -5.6, shape: (d) => [1, d.P, d.D], ax: "B,T,D", dtype: "float32", cls: "_merge_input_ids_with_image_features", code: ["paligemma.py", "scaled_image_features = image_features /"], to: "/gemma/sqrt-d",
    note: <>Gemma birazdan bütün diziyi ×√2048 yapacak. Görsel token'lar bundan etkilenmesin diye önce bölünüyor.</> },
  { key: "merge", title: "Metin + görüntü birleşiyor", z: -7.6, shape: (d, T) => [1, T, d.D], ax: "B,T,D", dtype: "float32", cls: "masked_scatter", code: ["paligemma.py", "final_embedding = final_embedding.masked_scatter"], to: "/hepsi#birlestirme",
    note: <>Prompt token'ları embedding tablosundan geliyor (yeşil), &lt;image&gt; konumlarına görsel özellikler sırayla yazılıyor.</> },
  { key: "scale", title: "× √2048", z: -7.6, shape: (d, T) => [1, T, d.D], ax: "B,T,D", dtype: "float32", cls: "GemmaModel.forward", code: ["gemma.py", "hidden_states = hidden_states * normalizer"], to: "/gemma/sqrt-d",
    note: <>Bütün dizi 45.25 ile çarpılıyor. Görsel token'lar için net etki ×1, metin token'ları 45 kat büyüyor.</> },
  { key: "dec", title: "Gemma decoder ×18", z: -10.6, shape: (d, T) => [1, T, d.D], ax: "B,T,D", dtype: "float32", cls: "GemmaDecoderLayer", code: ["gemma.py", "hidden_states = decoder_layer(hidden_states)"], to: "/gemma/akis",
    note: <>Her katman akışa iki güncelleme ekliyor (attn + mlp). Attention şimdilik Identity, aşama 5'te token'lar birbirini görecek.</> },
  { key: "norm", title: "Son RMSNorm", z: -13, shape: (d, T) => [1, T, d.D], ax: "B,T,D", dtype: "float32", cls: "GemmaRMSNorm", code: ["gemma.py", "hidden_states = self.norm(hidden_states)"], to: "/gemma/rmsnorm",
    note: <>18 katman boyunca büyüyen akış, lm_head'den önce bir kez normalize ediliyor.</> },
  { key: "head", title: "lm_head → logit'ler", z: -15, shape: (d, T) => [1, T, d.Vc], ax: "B,T,V", dtype: "float32", cls: "nn.Linear (= embed_tokens.weight)", code: ["gemma.py", "logits = self.lm_head(hidden_states)"], to: "/gemma/tying",
    note: <>Her konum için sözlükteki 257216 token'a bir skor. Üretimde sadece <b>son</b> konumun skorları kullanılıyor: argmax → bir sonraki token (aşama 6).</> },
];

export const idx = (k: string) => STAGES.findIndex((s) => s.key === k);

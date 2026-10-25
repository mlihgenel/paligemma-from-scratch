import { lazy, ComponentType, LazyExoticComponent } from "react";
import type { L } from "./i18n";

/** Boxes of the paper's architecture diagram (ArchDiagram). The boxes a page covers are highlighted. */
export type ArchKey = "image" | "siglip" | "proj" | "prompt" | "tokenizer" | "gemma" | "output";
/** The book's three parts: used for colors and for grouping the table of contents */
export type Part = "vision" | "bridge" | "lang";

export interface Page {
  path: string;
  num: string;
  title: L<string>;
  /** One line in the table of contents: "what is here" */
  blurb: L<string>;
  part: Part | null;
  arch: ArchKey[];
  Component: L<LazyExoticComponent<ComponentType<any>>>;
}

type Loader = () => Promise<{ default: ComponentType<any> }>;
const mdx = (en: Loader, tr: Loader) => ({ en: lazy(en), tr: lazy(tr) });

export const PART_TITLE: Record<Part, L<string>> = {
  vision: { en: "Vision side: SigLIP", tr: "Görüntü tarafı: SigLIP" },
  bridge: { en: "The bridge", tr: "Köprü" },
  lang: { en: "Language side: Gemma", tr: "Dil tarafı: Gemma" },
};

export const START_TITLE: L<string> = { en: "Start", tr: "Başlangıç" };

/** The part's color: used in CSS as var(--part) and var(--part-soft) */
export const partVars = (part: Part | null) => (part ? { ["--part" as any]: `var(--part-${part})`, ["--part-soft" as any]: `var(--part-${part}-soft)` } : {});

export const PAGES: Page[] = [
  { path: "/", num: "", part: null, arch: [],
    title: { en: "How does PaliGemma work?", tr: "PaliGemma nasıl çalışır?" },
    blurb: { en: "The whole architecture in one animation: from image to answer.", tr: "Bütün mimari tek bir animasyonda: görüntüden cevaba." },
    Component: mdx(() => import("./chapters/giris.en.mdx"), () => import("./chapters/giris.tr.mdx")) },
  { path: "/end-to-end", num: "1", part: null, arch: ["image", "siglip", "proj", "prompt", "tokenizer", "gemma", "output"],
    title: { en: "End to end: a photo's journey", tr: "Uçtan uca: bir fotoğrafın yolculuğu" },
    blurb: { en: "A real street sign passes through the model stop by stop.", tr: "Gerçek bir tabela fotoğrafı modelin içinden durak durak geçiyor." },
    Component: mdx(() => import("./chapters/uctan/uctan.en.mdx"), () => import("./chapters/uctan/uctan.tr.mdx")) },

  { path: "/big-picture", num: "2.1", part: "vision", arch: ["image", "siglip"],
    title: { en: "The big picture", tr: "Büyük resim" },
    blurb: { en: "Why does SigLIP produce an image representation aligned with language?", tr: "SigLIP neden dille hizalı bir görüntü temsili üretiyor?" },
    Component: mdx(() => import("./chapters/siglip/buyuk.en.mdx"), () => import("./chapters/siglip/buyuk.tr.mdx")) },
  { path: "/patches", num: "2.2", part: "vision", arch: ["image", "siglip"],
    title: { en: "From image to patches", tr: "Görüntüden yamaya" },
    blurb: { en: "Preprocessing, 14 × 14 patches and the position embedding.", tr: "Ön işleme, 14 × 14 yamalar ve konum embedding'i." },
    Component: mdx(() => import("./chapters/siglip/goruntu.en.mdx"), () => import("./chapters/siglip/goruntu.tr.mdx")) },
  { path: "/encoder", num: "2.3", part: "vision", arch: ["siglip"],
    title: { en: "The SigLIP encoder", tr: "SigLIP encoder" },
    blurb: { en: "27 layers: patches see each other through attention.", tr: "27 katman: yamalar attention ile birbirini görüyor." },
    Component: mdx(() => import("./chapters/siglip/encoder.en.mdx"), () => import("./chapters/siglip/encoder.tr.mdx")) },

  { path: "/bridge", num: "3", part: "bridge", arch: ["proj", "prompt", "tokenizer"],
    title: { en: "The bridge: how the image enters the language model", tr: "Köprü: görüntü dil modeline nasıl girer" },
    blurb: { en: "Processor, projector, merging and the prefix-LM mask.", tr: "Processor, projektör, birleştirme ve prefix-LM maskesi." },
    Component: mdx(() => import("./chapters/siglip/kopru.en.mdx"), () => import("./chapters/siglip/kopru.tr.mdx")) },

  { path: "/gemma", num: "4.1", part: "lang", arch: ["gemma", "output"],
    title: { en: "Map of Gemma", tr: "Gemma haritası" },
    blurb: { en: "The parts of the language model and their shapes at a glance.", tr: "Dil modelinin parçaları ve şekilleri, tek bakışta." },
    Component: mdx(() => import("./chapters/gemma/harita.en.mdx"), () => import("./chapters/gemma/harita.tr.mdx")) },
  { path: "/gemma/residual", num: "4.2", part: "lang", arch: ["gemma"],
    title: { en: "The residual stream: a highway", tr: "Residual akışı: otoyol" },
    blurb: { en: "Layers read from and write to the same stream.", tr: "Katmanlar aynı akışa okuyup yazıyor." },
    Component: mdx(() => import("./chapters/gemma/akis.en.mdx"), () => import("./chapters/gemma/akis.tr.mdx")) },
  { path: "/gemma/sqrt-d", num: "4.3", part: "lang", arch: ["proj", "gemma"],
    title: { en: "The √d scale", tr: "√d ölçeği" },
    blurb: { en: "Why are embeddings multiplied by 45.25, and why not the image tokens?", tr: "Embedding neden 45.25 ile çarpılıyor, görsel token'lar neden çarpılmıyor?" },
    Component: mdx(() => import("./chapters/gemma/sqrtd.en.mdx"), () => import("./chapters/gemma/sqrtd.tr.mdx")) },
  { path: "/gemma/rmsnorm", num: "4.4", part: "lang", arch: ["gemma"],
    title: { en: "RMSNorm", tr: "RMSNorm" },
    blurb: { en: "Drop the magnitude, keep the direction: (1 + w), keepdim, float32.", tr: "Büyüklüğü at, yönü koru: (1 + w), keepdim, float32." },
    Component: mdx(() => import("./chapters/gemma/rmsnorm.en.mdx"), () => import("./chapters/gemma/rmsnorm.tr.mdx")) },
  { path: "/gemma/geglu", num: "4.5", part: "lang", arch: ["gemma"],
    title: { en: "The GeGLU MLP", tr: "GeGLU MLP" },
    blurb: { en: "A gated MLP: gate, up, down.", tr: "Kapılı MLP: gate, up, down." },
    Component: mdx(() => import("./chapters/gemma/geglu.en.mdx"), () => import("./chapters/gemma/geglu.tr.mdx")) },
  { path: "/gemma/layer", num: "4.6", part: "lang", arch: ["gemma"],
    title: { en: "The decoder layer", tr: "Decoder katmanı" },
    blurb: { en: "Norm → attention → ⊕, norm → MLP → ⊕.", tr: "Norm → attention → ⊕, norm → MLP → ⊕." },
    Component: mdx(() => import("./chapters/gemma/katman.en.mdx"), () => import("./chapters/gemma/katman.tr.mdx")) },
  { path: "/gemma/model", num: "4.7", part: "lang", arch: ["gemma"],
    title: { en: "GemmaModel", tr: "GemmaModel" },
    blurb: { en: "18 layers stacked and a final norm.", tr: "18 katman üst üste ve son norm." },
    Component: mdx(() => import("./chapters/gemma/model.en.mdx"), () => import("./chapters/gemma/model.tr.mdx")) },
  { path: "/gemma/tying", num: "4.8", part: "lang", arch: ["gemma", "output"],
    title: { en: "lm_head and weight tying", tr: "lm_head ve weight tying" },
    blurb: { en: "One table at both ends: from vector to word.", tr: "Aynı tablo hem girişte hem çıkışta: vektörden kelimeye." },
    Component: mdx(() => import("./chapters/gemma/tying.en.mdx"), () => import("./chapters/gemma/tying.tr.mdx")) },
  { path: "/gemma/rope", num: "4.9", part: "lang", arch: ["gemma"],
    title: { en: "RoPE: position by rotation", tr: "RoPE: konum, dönerek" },
    blurb: { en: "Q and K rotate by position: the score depends only on the position difference.", tr: "Q ve K konumlarına göre dönüyor: skor yalnızca konum farkına bağlı." },
    Component: mdx(() => import("./chapters/gemma/rope.en.mdx"), () => import("./chapters/gemma/rope.tr.mdx")) },
  { path: "/gemma/mqa", num: "4.10", part: "lang", arch: ["gemma"],
    title: { en: "MQA and repeat_kv", tr: "MQA ve repeat_kv" },
    blurb: { en: "8 query heads, 1 KV head: the saving is in memory.", tr: "8 sorgu head'i, 1 KV head'i: tasarruf bellekten geliyor." },
    Component: mdx(() => import("./chapters/gemma/mqa.en.mdx"), () => import("./chapters/gemma/mqa.tr.mdx")) },
  { path: "/gemma/attention", num: "4.11", part: "lang", arch: ["gemma"],
    title: { en: "GemmaAttention", tr: "GemmaAttention" },
    blurb: { en: "Scores, the causal mask, softmax and how it plugs into the layer.", tr: "Skor, nedensel maske, softmax ve katmana bağlanış." },
    Component: mdx(() => import("./chapters/gemma/attention.en.mdx"), () => import("./chapters/gemma/attention.tr.mdx")) },
  { path: "/gemma/kv-cache", num: "4.12", part: "lang", arch: ["siglip", "gemma", "output"],
    title: { en: "KV cache: generating token by token", tr: "KV cache: token token üretim" },
    blurb: { en: "Prefill once, then one token per step: what is stored and why it is enough.", tr: "Bir kez prefill, sonra adım başına bir token: ne saklanıyor, neden yetiyor?" },
    Component: mdx(() => import("./chapters/gemma/kvcache.en.mdx"), () => import("./chapters/gemma/kvcache.tr.mdx")) },
];

/** Old addresses (including the earlier Turkish slugs): keep bookmarks working */
export const REDIRECTS: Record<string, string> = {
  "/uctan-uca": "/end-to-end",
  "/buyuk-resim": "/big-picture",
  "/goruntu": "/patches",
  "/kopru": "/bridge",
  "/gemma/akis": "/gemma/residual",
  "/gemma/katman": "/gemma/layer",
  "/oyun-alani": "/end-to-end",
  "/hepsi": "/bridge",
  "/ekler": "/",
  "/gemma/dogrulama": "/gemma",
  "/gemma/sina": "/gemma",
};

import { lazy, ComponentType, LazyExoticComponent } from "react";

/** Makaledeki mimari diyagramının kutuları (ArchDiagram). Sayfanın anlattığı kutular vurgulanır. */
export type ArchKey = "image" | "siglip" | "proj" | "prompt" | "tokenizer" | "gemma" | "output";
/** Kitabın üç kısmı: renk ve içindekiler gruplaması için */
export type Part = "vision" | "bridge" | "lang";

export interface Page {
  path: string;
  num: string;
  title: string;
  /** İçindekiler'de bir satırlık "burada ne var" */
  blurb: string;
  part: Part | null;
  arch: ArchKey[];
  Component: LazyExoticComponent<ComponentType<any>>;
}

const mdx = (f: () => Promise<{ default: ComponentType<any> }>) => lazy(f);

export const PART_TITLE: Record<Part, string> = {
  vision: "Görüntü tarafı: SigLIP",
  bridge: "Köprü",
  lang: "Dil tarafı: Gemma",
};

/** Kısmın rengi: CSS'te var(--part) ve var(--part-soft) olarak kullanılıyor */
export const partVars = (part: Part | null) => (part ? { ["--part" as any]: `var(--part-${part})`, ["--part-soft" as any]: `var(--part-${part}-soft)` } : {});

export const PAGES: Page[] = [
  { path: "/", num: "", title: "PaliGemma nasıl çalışır?", blurb: "Bütün mimari tek bir animasyonda: görüntüden cevaba.", part: null, arch: [], Component: mdx(() => import("./chapters/giris.mdx")) },
  { path: "/uctan-uca", num: "1", title: "Uçtan uca: bir fotoğrafın yolculuğu", blurb: "Gerçek bir tabela fotoğrafı modelin içinden durak durak geçiyor.", part: null, arch: ["image", "siglip", "proj", "prompt", "tokenizer", "gemma", "output"], Component: mdx(() => import("./chapters/uctan/uctan.mdx")) },

  { path: "/buyuk-resim", num: "2.1", title: "Büyük resim", blurb: "SigLIP neden dille hizalı bir görüntü temsili üretiyor?", part: "vision", arch: ["image", "siglip"], Component: mdx(() => import("./chapters/siglip/buyuk.mdx")) },
  { path: "/goruntu", num: "2.2", title: "Görüntüden yamaya", blurb: "Ön işleme, 14 × 14 yamalar ve konum embedding'i.", part: "vision", arch: ["image", "siglip"], Component: mdx(() => import("./chapters/siglip/goruntu.mdx")) },
  { path: "/encoder", num: "2.3", title: "SigLIP encoder", blurb: "27 katman: yamalar attention ile birbirini görüyor.", part: "vision", arch: ["siglip"], Component: mdx(() => import("./chapters/siglip/encoder.mdx")) },

  { path: "/kopru", num: "3", title: "Köprü: görüntü dil modeline nasıl girer", blurb: "Processor, projektör, birleştirme ve prefix-LM maskesi.", part: "bridge", arch: ["proj", "prompt", "tokenizer"], Component: mdx(() => import("./chapters/siglip/kopru.mdx")) },

  { path: "/gemma", num: "4.1", title: "Gemma haritası", blurb: "Dil modelinin parçaları ve şekilleri, tek bakışta.", part: "lang", arch: ["gemma", "output"], Component: mdx(() => import("./chapters/gemma/harita.mdx")) },
  { path: "/gemma/akis", num: "4.2", title: "Residual akışı: otoyol", blurb: "Katmanlar aynı akışa okuyup yazıyor.", part: "lang", arch: ["gemma"], Component: mdx(() => import("./chapters/gemma/akis.mdx")) },
  { path: "/gemma/sqrt-d", num: "4.3", title: "√d ölçeği", blurb: "Embedding neden 45.25 ile çarpılıyor, görsel token'lar neden bölünüyor?", part: "lang", arch: ["proj", "gemma"], Component: mdx(() => import("./chapters/gemma/sqrtd.mdx")) },
  { path: "/gemma/rmsnorm", num: "4.4", title: "RMSNorm", blurb: "Büyüklüğü at, yönü koru: (1 + w), keepdim, float32.", part: "lang", arch: ["gemma"], Component: mdx(() => import("./chapters/gemma/rmsnorm.mdx")) },
  { path: "/gemma/geglu", num: "4.5", title: "GeGLU MLP", blurb: "Kapılı MLP: gate, up, down.", part: "lang", arch: ["gemma"], Component: mdx(() => import("./chapters/gemma/geglu.mdx")) },
  { path: "/gemma/katman", num: "4.6", title: "Decoder katmanı", blurb: "Norm → attention → ⊕, norm → MLP → ⊕.", part: "lang", arch: ["gemma"], Component: mdx(() => import("./chapters/gemma/katman.mdx")) },
  { path: "/gemma/model", num: "4.7", title: "GemmaModel", blurb: "18 katman üst üste ve son norm.", part: "lang", arch: ["gemma"], Component: mdx(() => import("./chapters/gemma/model.mdx")) },
  { path: "/gemma/tying", num: "4.8", title: "lm_head ve weight tying", blurb: "Aynı tablo hem girişte hem çıkışta: vektörden kelimeye.", part: "lang", arch: ["gemma", "output"], Component: mdx(() => import("./chapters/gemma/tying.mdx")) },
];

/** Eski adresler: yer imleri kırılmasın */
export const REDIRECTS: Record<string, string> = {
  "/oyun-alani": "/uctan-uca",
  "/hepsi": "/kopru",
  "/ekler": "/",
  "/gemma/dogrulama": "/gemma",
  "/gemma/sina": "/gemma",
};

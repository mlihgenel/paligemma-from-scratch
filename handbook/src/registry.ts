import { lazy, ComponentType, LazyExoticComponent } from "react";
import type { StageKey } from "./components/PipelineMinimap";

export interface Page {
  path: string;
  num: string;
  title: string;
  chapter: string;
  active: StageKey[];
  Component?: LazyExoticComponent<ComponentType<any>>;
}

const mdx = (f: () => Promise<{ default: ComponentType<any> }>) => lazy(f);

export const PAGES: Page[] = [
  { path: "/", num: "", title: "Bu kitap nasıl okunur", chapter: "Giriş", active: [], Component: mdx(() => import("./chapters/giris.mdx")) },
  { path: "/oyun-alani", num: "0", title: "Oyun alanı: uçtan uca", chapter: "Giriş", active: ["image", "process", "patch", "encoder", "proj", "merge", "scale", "decoder", "norm", "head"], Component: mdx(() => import("./chapters/oyun/oyun.mdx")) },

  { path: "/buyuk-resim", num: "1", title: "Büyük resim", chapter: "Görüntü tarafı", active: ["image", "encoder"], Component: mdx(() => import("./chapters/siglip/buyuk.mdx")) },
  { path: "/goruntu", num: "2", title: "Görüntüden yamaya", chapter: "Görüntü tarafı", active: ["image", "process", "patch"], Component: mdx(() => import("./chapters/siglip/goruntu.mdx")) },
  { path: "/encoder", num: "3", title: "SigLIP encoder", chapter: "Görüntü tarafı", active: ["encoder"], Component: mdx(() => import("./chapters/siglip/encoder.mdx")) },
  { path: "/kopru", num: "4", title: "Köprü: processor ve projektör", chapter: "Görüntü tarafı", active: ["proj", "merge"], Component: mdx(() => import("./chapters/siglip/kopru.mdx")) },

  { path: "/gemma", num: "5.0", title: "Gemma haritası", chapter: "Gemma çekirdeği", active: ["scale", "decoder", "norm", "head"], Component: mdx(() => import("./chapters/gemma/harita.mdx")) },
  { path: "/gemma/akis", num: "5.1", title: "Residual akışı: otoyol", chapter: "Gemma çekirdeği", active: ["decoder"], Component: mdx(() => import("./chapters/gemma/akis.mdx")) },
  { path: "/gemma/sqrt-d", num: "5.2", title: "√d ölçeği", chapter: "Gemma çekirdeği", active: ["merge", "scale"], Component: mdx(() => import("./chapters/gemma/sqrtd.mdx")) },
  { path: "/gemma/rmsnorm", num: "5.3", title: "RMSNorm", chapter: "Gemma çekirdeği", active: ["decoder", "norm"], Component: mdx(() => import("./chapters/gemma/rmsnorm.mdx")) },
  { path: "/gemma/geglu", num: "5.4", title: "GeGLU MLP", chapter: "Gemma çekirdeği", active: ["decoder"], Component: mdx(() => import("./chapters/gemma/geglu.mdx")) },
  { path: "/gemma/katman", num: "5.5", title: "GemmaDecoderLayer", chapter: "Gemma çekirdeği", active: ["decoder"], Component: mdx(() => import("./chapters/gemma/katman.mdx")) },
  { path: "/gemma/model", num: "5.6", title: "GemmaModel", chapter: "Gemma çekirdeği", active: ["scale", "decoder", "norm"], Component: mdx(() => import("./chapters/gemma/model.mdx")) },
  { path: "/gemma/tying", num: "5.7", title: "lm_head ve weight tying", chapter: "Gemma çekirdeği", active: ["head"], Component: mdx(() => import("./chapters/gemma/tying.mdx")) },
  { path: "/gemma/dogrulama", num: "5.8", title: "Doğrulama nasıl çalışıyor", chapter: "Gemma çekirdeği", active: [], Component: mdx(() => import("./chapters/gemma/dogrulama.mdx")) },
  { path: "/gemma/sina", num: "5.9", title: "Kendini sına", chapter: "Gemma çekirdeği", active: [], Component: mdx(() => import("./chapters/gemma/sina.mdx")) },

  { path: "/hepsi", num: "6", title: "Hepsi bir arada", chapter: "Birleştirme", active: ["proj", "merge", "scale", "decoder", "norm", "head"], Component: mdx(() => import("./chapters/hepsi/hepsi.mdx")) },
  { path: "/ekler", num: "E", title: "Ekler: kopya kâğıdı, sözlük", chapter: "Birleştirme", active: [], Component: mdx(() => import("./chapters/ekler/ekler.mdx")) },
];

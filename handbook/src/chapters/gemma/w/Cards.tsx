// gemma.py satır kartları: ön yüz satır, arka yüz "olmasa ne bozulur"
import { useState } from "react";
import { lineOf, SOURCES } from "../../../lib/pysource";

const CARDS: { match: string; back: string }[] = [
  { match: "self.weight = nn.Parameter(torch.zeros(dim))", back: "ones ile başlatırsan ölçek 1+1 = 2 olur. Checkpoint yüklenince değerler üzerine yazıldığı için inference'ta fark etmez, ama sıfırdan eğitimde norm 2× büyütür. Asıl tehlike tersi: Llama'nın ones + x*w koduyla Gemma ağırlığı yüklemek." },
  { match: "x.pow(2).mean(-1, keepdim=True)", back: "keepdim olmazsa [B,T] çıkar. Broadcast T ile D'yi eşler: T≠D ise hata, T=D ise sessizce her token yanlış RMS'le bölünür." },
  { match: "output = self._norm(x.float())", back: ".float() olmazsa bf16'da kare toplamı 7 bit mantisle hesaplanır. 2048 karenin toplamı kaba yuvarlanır, HF ile bf16 testi birebir tutmaz." },
  { match: "output = output * (1.0 + self.weight.float())", back: "Önce type_as sonra çarpma (Llama sırası): iki bf16 yuvarlaması. Elemanların ~%33'ü 1 ulp farklı. exact=True testi düşer." },
  { match: "h = F.gelu(gate, approximate=\"tanh\") * up", back: "approximate olmazsa (erf GELU): ~5e-4'e kadar fark, allclose(1e-5) düşer. Aktivasyonu up'a uygularsan: tamamen farklı bir fonksiyon, test anında düşer." },
  { match: "residual = hidden_states", back: "Norm'dan sonra saklanırsa akış her alt blokta normalize edilir (fiilen post-norm). Oyuncak modelde embedding'e benzerlik 2 katmanda 0.2'ye çöker." },
  { match: "hidden_states = residual + hidden_states", back: "Toplama olmazsa katman çıktısı akışın yerine geçer: residual bağlantı yok, 18 katmanlık derin ağ gradyan yolu olmadan çalışır. Ağırlıklarla uyumsuz, çıktı anlamsız." },
  { match: "self.layers = nn.ModuleList(", back: "Düz liste: parametreler kaydedilmez, state_dict'te layers.* yok, load_state_dict yüklemez, .to(device) taşımaz. Hata vermeden rastgele katmanlarla çalışır." },
  { match: "normalizer = torch.tensor", back: "Python float ile çarparsan bf16'da elemanların ~%2.6'sı 1 ulp farklı (45.2548 ile 45.25). Hiç çarpmazsan embedding ilk katmanda boğulur." },
  { match: "hidden_states = self.norm(hidden_states)", back: "Son norm olmazsa lm_head büyümüş, ölçeksiz akışı görür. Logit'lerin ölçeği girdiye bağlı olur, ağırlıklarla uyumsuz." },
  { match: "self.lm_head.weight = self.model.embed_tokens.weight", back: "Bu satır olmazsa lm_head rastgele kalır (checkpoint'te lm_head anahtarı yok!). Model tamamen anlamsız token'lar üretir, hiçbir hata vermeden." },
  { match: "return logits.float()", back: "bf16 logit'ler 20 civarında 0.125 adımla yuvarlanır: olasılıklar %13'e kadar kayar, yakın adaylarda greedy seçim HF'den ayrışabilir." },
];

export function LineCards() {
  const [flipped, setFlipped] = useState<Set<number>>(new Set());
  const src = SOURCES["gemma.py"] ?? "";
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))", gap: 10 }}>
      {CARDS.map((c, i) => {
        const ln = lineOf("gemma.py", c.match);
        const line = ln > 0 ? src.split("\n")[ln - 1].trim() : null;
        const f = flipped.has(i);
        return (
          <button key={i} onClick={() => { const s = new Set(flipped); f ? s.delete(i) : s.add(i); setFlipped(s); }}
            style={{ all: "unset", cursor: "pointer", minHeight: 120, padding: 14, borderRadius: 10, border: `1px solid ${f ? "var(--bad)" : "var(--border)"}`, background: f ? "var(--bad-soft)" : "var(--bg-elev)", boxShadow: "var(--shadow)", display: "flex", flexDirection: "column", gap: 8 }}>
            {!f ? (
              <>
                <span className="small muted">gemma.py:{ln > 0 ? ln : "?"}</span>
                <code style={{ fontSize: 12, wordBreak: "break-word", whiteSpace: "pre-wrap" }}>{line ?? `(bulunamadı: ${c.match})`}</code>
                <span className="small" style={{ marginTop: "auto", color: "var(--accent)" }}>Bu satır olmasa / yanlış olsa? →</span>
              </>
            ) : (
              <span style={{ fontSize: 14, lineHeight: 1.55 }}>{c.back}</span>
            )}
          </button>
        );
      })}
    </div>
  );
}

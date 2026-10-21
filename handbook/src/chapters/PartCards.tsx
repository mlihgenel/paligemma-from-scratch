// Açılıştaki üç kısım kartı: registry'den okunur, renkler kısma göre.
import { Link } from "react-router-dom";
import { PAGES, PART_TITLE, Part, partVars } from "../registry";

const DESC: Record<Part, string> = {
  vision: "Fotoğraf 256 yamaya bölünüyor, her yama 1152 sayılık bir vektör oluyor.",
  bridge: "Görsel vektörler dil modelinin boyutuna taşınıp komutla tek diziye diziliyor.",
  lang: "Gemma diziyi okuyup cevabı token token yazıyor.",
};

export function PartCards() {
  return (
    <div className="part-cards">
      {(["vision", "bridge", "lang"] as Part[]).map((part, i) => {
        const pages = PAGES.filter((p) => p.part === part);
        return (
          <Link key={part} to={pages[0].path} className="part-card" style={partVars(part)}>
            <div className="k">Bölüm {i + 2}</div>
            <div className="h">{PART_TITLE[part]}</div>
            <div className="d">{DESC[part]}</div>
            {pages.length > 1 && (
              <ul>{pages.map((p) => <li key={p.path}><span className="mono">{p.num}</span>{p.title}</li>)}</ul>
            )}
          </Link>
        );
      })}
    </div>
  );
}

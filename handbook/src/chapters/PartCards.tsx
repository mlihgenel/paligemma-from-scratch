// The three part cards on the landing page: read from the registry, colored by part.
import { Link } from "react-router-dom";
import { PAGES, PART_TITLE, Part, partVars } from "../registry";
import { useLang, useT, type L } from "../i18n";

const DESC: Record<Part, L<string>> = {
  vision: { en: "The photo is split into 256 patches, and each patch becomes a vector of 1152 numbers.", tr: "Fotoğraf 256 yamaya bölünüyor, her yama 1152 sayılık bir vektör oluyor." },
  bridge: { en: "The image vectors are moved to the language model's size and lined up with the prompt in one sequence.", tr: "Görsel vektörler dil modelinin boyutuna taşınıp komutla tek diziye diziliyor." },
  lang: { en: "Gemma reads the sequence and writes the answer token by token.", tr: "Gemma diziyi okuyup cevabı token token yazıyor." },
};

export function PartCards() {
  const lang = useLang();
  const t = useT();
  return (
    <div className="part-cards">
      {(["vision", "bridge", "lang"] as Part[]).map((part, i) => {
        const pages = PAGES.filter((p) => p.part === part);
        return (
          <Link key={part} to={pages[0].path} className="part-card" style={partVars(part)}>
            <div className="k">{t("Part", "Kısım")} {i + 2}</div>
            <div className="h">{PART_TITLE[part][lang]}</div>
            <div className="d">{DESC[part][lang]}</div>
            {pages.length > 1 && (
              <ul>{pages.map((p) => <li key={p.path}><span className="mono">{p.num}</span>{p.title[lang]}</li>)}</ul>
            )}
          </Link>
        );
      })}
    </div>
  );
}

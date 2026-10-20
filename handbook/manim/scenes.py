"""PaliGemma el kitabı için Manim (Community) sahneleri.

Render (proje kökünden):
    bash handbook/manim/render.sh            # hepsi
    bash handbook/manim/render.sh RMSNormGeometry
Çıktı: handbook/public/videos/<Sahne>.mp4
"""
import numpy as np
from manim import *

# Kitabın açık tema tokenları (src/styles/tokens.css) ile aynı
BG = "#f4f2ed"
INK = "#1b1b1f"
C_T = "#2a78d6"     # akış / token (eksen T)
C_D = "#eb6834"     # gizli boyut (eksen D)
C_I = "#6250d6"     # MLP ara boyutu (eksen I)
C_V = "#1baf7a"     # sözlük (eksen V)
C_N = "#d4558a"     # norm (eksen hd rengi)
C_A = "#3d4bd1"     # vurgu (indigo)
C_F = "#9a968d"     # soluk
FONT = "Inter"

config.background_color = BG


def T(s, size=28, color=INK, **kw):
    return Text(s, font=FONT, font_size=size, color=color, **kw)


def bars(values, color=C_T, width=0.18, scale=0.5, gap=0.05):
    """Bir vektörü dikey çubuklarla göster (sıfır çizgisi ortada)."""
    g = VGroup()
    for i, v in enumerate(values):
        h = max(abs(v) * scale, 0.02)
        r = Rectangle(width=width, height=h, stroke_width=0, fill_opacity=0.95,
                      fill_color=color if v >= 0 else C_F)
        r.move_to([i * (width + gap), np.sign(v) * h / 2, 0])
        g.add(r)
    g.move_to(ORIGIN)
    return g


def rms(v):
    return np.sqrt(np.mean(np.square(v)))


class ResidualHighway(Scene):
    def construct(self):
        title = T("Residual akışı: dallar okur, akışa ekler", 32).to_edge(UP, buff=0.4)
        self.play(FadeIn(title))
        lane = Line(LEFT * 6.4, RIGHT * 6.4, color=C_T, stroke_width=14, stroke_opacity=0.35).shift(DOWN * 1.6)
        lane_lbl = T("akış (hidden_states)", 20, C_T).next_to(lane, DOWN, buff=0.25).align_to(lane, LEFT)
        self.play(Create(lane), FadeIn(lane_lbl))

        rng = np.random.default_rng(3)
        h = rng.normal(size=10) * 1.2
        vec = bars(h, C_T, scale=0.45).move_to(lane.get_left() + RIGHT * 0.9 + UP * 0.9)
        cap = T("embedding × √d", 20, C_F).next_to(vec, DOWN, buff=0.9)
        self.play(FadeIn(vec), FadeIn(cap))
        self.wait(0.5)
        self.play(FadeOut(cap))

        xs = [-3.2, 0.2, 3.6]
        names = ["1. attn", "1. mlp", "2. attn"]
        for k, (x, name) in enumerate(zip(xs, names)):
            # akış boyunca ilerle
            self.play(vec.animate.move_to([x - 1.4, vec.get_center()[1], 0]), run_time=0.8)
            norm_box = RoundedRectangle(width=1.2, height=0.7, corner_radius=0.12, color=C_N).move_to([x - 0.2, 1.7, 0])
            f_box = RoundedRectangle(width=1.2, height=0.7, corner_radius=0.12, color=C_D if "mlp" in name else C_F).move_to([x + 1.3, 1.7, 0])
            nl = T("norm", 20, C_N).move_to(norm_box)
            fl = T(name, 20, INK).move_to(f_box)
            up = CurvedArrow(vec.get_top() + UP * 0.05, norm_box.get_bottom(), angle=-PI / 4, color=C_F, stroke_width=3)
            self.play(Create(norm_box), Create(f_box), FadeIn(nl), FadeIn(fl), Create(up), run_time=0.7)

            # dal: norm'lu kopya (akış değişmeden duruyor)
            copy = vec.copy()
            self.play(copy.animate.scale(0.6).move_to(norm_box.get_center() + DOWN * 0.95), run_time=0.6)
            normed = bars(h / rms(h), C_N, scale=0.28).move_to(copy)
            self.play(Transform(copy, normed), run_time=0.6)
            note = T("ölçekten bağımsız: rms = 1", 18, C_N).next_to(norm_box, UP, buff=0.15)
            if k == 0:
                self.play(FadeIn(note))
            # f uygulanıyor
            u = rng.normal(size=10) * 0.8
            upd = bars(u, C_D if "mlp" in name else C_F, scale=0.45).scale(0.6).move_to(f_box.get_center() + DOWN * 0.95)
            self.play(ReplacementTransform(copy, upd), run_time=0.6)
            # akışa ekle
            plus = MathTex(r"\oplus", color=C_A).scale(1.2).move_to([x + 2.1, lane.get_center()[1], 0])
            down = CurvedArrow(f_box.get_bottom(), plus.get_top(), angle=-PI / 4, color=C_A, stroke_width=3)
            self.play(Create(down), FadeIn(plus), run_time=0.5)
            self.play(vec.animate.move_to([x + 2.1, vec.get_center()[1], 0]), upd.animate.move_to([x + 2.1, vec.get_center()[1] + 1.4, 0]), run_time=0.7)
            h = h + u
            new = bars(h, C_T, scale=0.45).move_to(vec)
            self.play(Transform(vec, new), FadeOut(upd), run_time=0.6)
            self.play(*[FadeOut(m) for m in (norm_box, f_box, nl, fl, up, down, plus)] + ([FadeOut(note)] if k == 0 else []), run_time=0.4)

        final = T("h ← h + f(norm(h))   ·   akışın kendisi hiç normalize edilmiyor", 22, C_A).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(final))
        self.wait(2)


class ScaleRoundTrip(Scene):
    def construct(self):
        title = T("İki yol, tek akış: √d ölçeği", 32).to_edge(UP, buff=0.4)
        self.play(FadeIn(title))
        rng = np.random.default_rng(1)

        # Metin yolu
        tl = T("metin token'ı", 22, C_V).move_to([-5.2, 1.5, 0])
        e = rng.normal(size=8) * 0.19
        eb = bars(e, C_V, scale=2.2).move_to([-2.6, 1.5, 0])
        el = T("embed_tokens satırı\nstd ≈ 0.19", 16, C_F).next_to(eb, DOWN, buff=0.25)
        self.play(FadeIn(tl), FadeIn(eb), FadeIn(el))

        # Görüntü yolu
        il = T("görsel token", 22, C_T).move_to([-5.2, -1.6, 0])
        v = rng.normal(size=8) * 1.6
        vb = bars(v, C_T, scale=0.26).move_to([-2.6, -1.6, 0])
        vl = T("projektör çıktısı", 16, C_F).next_to(vb, DOWN, buff=0.25)
        self.play(FadeIn(il), FadeIn(vb), FadeIn(vl))
        self.wait(0.3)

        # Görüntü: ÷√d (paligemma.py)
        div = MathTex(r"\div\sqrt{2048}", color=C_A).scale(0.8).move_to([-0.6, -1.0, 0])
        dl = T("paligemma.py", 16, C_F).next_to(div, UP, buff=0.12)
        vb2 = bars(v / 45.25, C_T, scale=0.26).move_to([0.9, -1.6, 0])
        self.play(FadeIn(div), FadeIn(dl), TransformFromCopy(vb, vb2), run_time=1)
        tiny = T("çok küçük", 16, C_F).next_to(vb2, DOWN, buff=0.25)
        self.play(FadeIn(tiny))

        # Birleştirme
        merge = RoundedRectangle(width=1.6, height=4.6, corner_radius=0.2, color=C_I).move_to([2.6, 0, 0])
        ml = T("birleştir (masked_scatter)", 16, C_I).next_to(merge, DOWN, buff=0.12)
        self.play(Create(merge), FadeIn(ml))
        self.play(eb.copy().animate.move_to([2.6, 1.3, 0]), vb2.animate.move_to([2.6, -1.3, 0]), FadeOut(tiny), run_time=1)

        # ×√d (gemma.py) hepsine
        mul = MathTex(r"\times\sqrt{2048}", color=C_A).scale(0.8).move_to([4.1, 0, 0])
        gl = T("gemma.py\nhepsine", 16, C_F).next_to(mul, DOWN, buff=0.12)
        self.play(FadeIn(mul), FadeIn(gl))
        out_t = bars(e * 45.25, C_V, scale=0.075).move_to([5.6, 1.5, 0])
        out_v = bars(v, C_T, scale=0.26).move_to([5.6, -1.6, 0])
        self.play(TransformFromCopy(eb, out_t), TransformFromCopy(vb2, out_v), run_time=1.2)
        eq = T("= projektör çıktısının kendisi", 16, C_T).next_to(out_v, DOWN, buff=0.25)
        big = T("45 kat büyüdü (ölçek 1/3,5)", 16, C_V).next_to(out_t, DOWN, buff=0.2)
        self.play(FadeIn(eq), FadeIn(big))
        self.play(Indicate(vb, color=C_A), Indicate(out_v, color=C_A))
        foot = T("görsel token: ÷√d · ×√d = 1   ·   metin token'ı: ×√d", 22, C_A).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(foot))
        self.wait(2)


class RMSNormGeometry(Scene):
    def construct(self):
        title = T("RMSNorm: her vektör √d yarıçaplı çembere", 32).to_edge(UP, buff=0.3)
        self.play(FadeIn(title))
        ax = Axes(x_range=[-4, 4, 1], y_range=[-3, 3, 1], x_length=7.2, y_length=5.4, axis_config={"color": C_F, "stroke_width": 2}).shift(DOWN * 0.3)
        self.play(Create(ax))
        rng = np.random.default_rng(5)
        P = rng.normal(size=(70, 2)) * [1.4, 1.0] + [0.6, 0.4]
        dots = VGroup(*[Dot(ax.c2p(*p), radius=0.05, color=C_F) for p in P])
        self.play(LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.02))
        lbl = T("d = 2: rastgele vektörler", 20, C_F).to_edge(DOWN, buff=0.75)
        self.play(FadeIn(lbl))

        R = np.sqrt(2)
        circ = Circle(radius=ax.x_length / 8 * R, color=C_T, stroke_width=2).move_to(ax.c2p(0, 0))
        f1 = MathTex(r"x \;/\; \sqrt{\tfrac{1}{d}\sum x_i^2}", color=C_T).scale(0.8).to_corner(UR).shift(DOWN * 0.6)
        rays = VGroup(*[DashedLine(ax.c2p(0, 0), d.get_center(), stroke_width=1, color=C_F, stroke_opacity=0.4) for d in dots])
        self.play(Create(rays), run_time=0.8)
        normed = [p / np.sqrt(np.mean(p ** 2)) for p in P]
        self.play(FadeIn(f1), Create(circ), *[d.animate.move_to(ax.c2p(*q)).set_color(C_T) for d, q in zip(dots, normed)], FadeOut(rays), run_time=2)
        self.play(Transform(lbl, T("yön korunuyor, büyüklük atılıyor · yarıçap √2", 20, C_T).to_edge(DOWN, buff=0.75)))
        self.wait(0.8)

        # (1 + w)
        w = np.array([0.7, -0.45])
        f2 = MathTex(r"\odot\,(1 + w)", color=C_N).scale(0.8).next_to(f1, DOWN)
        ell = Ellipse(width=ax.x_length / 8 * R * 2 * (1 + w[0]), height=ax.y_length / 6 * R * 2 * (1 + w[1]), color=C_N, stroke_width=2).move_to(ax.c2p(0, 0))
        self.play(FadeIn(f2), Transform(circ, ell), *[d.animate.move_to(ax.c2p(*(q * (1 + w)))).set_color(C_N) for d, q in zip(dots, normed)], run_time=1.8)
        self.play(Transform(lbl, T("(1+w): her eksen ayrı ölçek → elips", 20, C_N).to_edge(DOWN, buff=0.75)))
        self.wait(0.8)

        # LayerNorm karşılaştırması
        self.play(FadeOut(f1), FadeOut(f2), FadeOut(circ), *[d.animate.move_to(ax.c2p(*p)).set_color(C_F) for d, p in zip(dots, P)], run_time=1.2)
        line = ax.plot(lambda x: -x, x_range=[-3, 3], color=C_A, stroke_width=2)
        f3 = T("LayerNorm: önce ortalamayı çıkar → x₀ + x₁ = 0 doğrusu", 20, C_A).to_edge(DOWN, buff=0.75)
        self.play(Transform(lbl, f3), Create(line))
        proj = [p - p.mean() for p in P]
        self.play(*[d.animate.move_to(ax.c2p(*q)).set_color(C_A) for d, q in zip(dots, proj)], run_time=1.5)
        ln = [q / np.sqrt(np.mean(q ** 2) + 1e-9) for q in proj]
        self.play(*[d.animate.move_to(ax.c2p(*q)) for d, q in zip(dots, ln)], run_time=1.5)
        self.play(Transform(lbl, T("d = 2'de LayerNorm sadece iki nokta bırakıyor: (1, −1) ve (−1, 1)", 20, C_A).to_edge(DOWN, buff=0.75)))
        foot = T("RMSNorm ortalamayı korur · bir serbestlik derecesi daha fazla", 22, C_T).to_edge(DOWN, buff=0.2)
        self.play(FadeIn(foot))
        self.wait(2)


class GeGLUGating(Scene):
    def construct(self):
        title = T("GeGLU: kapı ⊙ değer", 32).to_edge(UP, buff=0.4)
        self.play(FadeIn(title))
        rng = np.random.default_rng(7)
        x = rng.normal(size=6)
        xb = bars(x, C_T, scale=0.4).move_to([-5.3, 0, 0])
        xl = T("norm(h)\n[2048]", 18, C_T).next_to(xb, DOWN, buff=0.3)
        self.play(FadeIn(xb), FadeIn(xl))

        g = rng.normal(size=12) * 1.4
        u = rng.normal(size=12) * 1.0
        gb = bars(g, C_N, scale=0.35, width=0.16).move_to([-2.0, 1.6, 0])
        ub = bars(u, C_D, scale=0.35, width=0.16).move_to([-2.0, -1.6, 0])
        gl = T("gate_proj → [16384]", 18, C_N).next_to(gb, UP, buff=0.2)
        ul = T("up_proj → [16384]", 18, C_D).next_to(ub, DOWN, buff=0.2)
        self.play(TransformFromCopy(xb, gb), TransformFromCopy(xb, ub), FadeIn(gl), FadeIn(ul), run_time=1.2)

        # GELU eğrisi
        ax = Axes(x_range=[-3, 3, 1], y_range=[-0.5, 3, 1], x_length=2.6, y_length=1.8, axis_config={"color": C_F, "stroke_width": 1.5}).move_to([1.2, 1.6, 0])
        gelu = lambda t: 0.5 * t * (1 + np.tanh(np.sqrt(2 / np.pi) * (t + 0.044715 * t ** 3)))
        curve = ax.plot(gelu, color=C_A)
        cl = T("gelu_tanh", 18, C_A).next_to(ax, UP, buff=0.1)
        self.play(Create(ax), Create(curve), FadeIn(cl))
        a = gelu(g)
        ab = bars(a, C_A, scale=0.35, width=0.16).move_to([3.9, 1.6, 0])
        self.play(TransformFromCopy(gb, ab), run_time=1)
        closed = [i for i, v in enumerate(a) if abs(v) < 0.12]
        marks = VGroup(*[SurroundingRectangle(ab[i], color=C_F, buff=0.03, stroke_width=2) for i in closed])
        self.play(Create(marks))
        kl = T("≈ 0: kapı kapalı", 16, C_F).next_to(ab, DOWN, buff=0.15)
        self.play(FadeIn(kl))

        # çarpım
        times = MathTex(r"\odot", color=INK).scale(1.2).move_to([3.9, 0, 0])
        ub2 = ub.copy()
        self.play(ub2.animate.move_to([3.9, -1.6, 0]), FadeIn(times))
        p = a * u
        pb = bars(p, C_I, scale=0.35, width=0.16).move_to([5.6, 0, 0])
        pl = T("h = gelu(gate) ⊙ up", 16, C_I).next_to(pb, DOWN, buff=0.3)
        self.play(TransformFromCopy(VGroup(ab, ub2), pb), FadeIn(pl), run_time=1.2)
        pmarks = VGroup(*[SurroundingRectangle(pb[i], color=C_F, buff=0.03, stroke_width=2) for i in closed])
        self.play(Create(pmarks))
        foot = T("kapı kapalıysa up'ın değeri ne olursa olsun geçmez → sonra down_proj ile [2048]", 20, C_A).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(foot))
        self.wait(2)


class WeightTying(Scene):
    def construct(self):
        title = T("Weight tying: tek matris, iki rol", 32).to_edge(UP, buff=0.4)
        self.play(FadeIn(title))
        V, D = 9, 6
        rng = np.random.default_rng(2)
        W = rng.normal(size=(V, D))
        cells = VGroup(*[VGroup(*[Square(0.42, stroke_width=1, stroke_color=BG, fill_opacity=0.35 + 0.5 * abs(np.tanh(W[r, c])), fill_color=C_V) for c in range(D)]).arrange(RIGHT, buff=0.02) for r in range(V)]).arrange(DOWN, buff=0.02)
        cells.move_to(ORIGIN)
        name = T("embed_tokens.weight  is  lm_head.weight", 20, C_V).next_to(cells, UP, buff=0.25)
        vl = T("V = 257216", 18, C_F).next_to(cells, LEFT, buff=0.3)
        dl = T("D = 2048", 18, C_F).next_to(cells, DOWN, buff=0.2)
        self.play(FadeIn(cells), FadeIn(name), FadeIn(vl), FadeIn(dl))

        # Giriş: lookup
        sub = T("Girişte: id → satır seç (hesap yok)", 22, C_T).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(sub))
        idt = T("id = 4", 22, C_A).move_to([-5, cells[4].get_center()[1], 0])
        self.play(FadeIn(idt), cells[4].animate.set_stroke(C_A, width=3))
        row = cells[4].copy()
        self.play(row.animate.next_to(cells, LEFT, buff=1.6).shift(UP * 2.2).set_fill(C_T, opacity=0.9), run_time=1.2)
        rl = T("vektör [2048]", 18, C_T).next_to(row, DOWN, buff=0.15)
        self.play(FadeIn(rl))
        self.wait(0.8)
        self.play(FadeOut(row), FadeOut(rl), FadeOut(idt), cells[4].animate.set_stroke(BG, width=1), FadeOut(sub))

        # Çıkış: h @ W.T
        sub2 = T("Çıkışta: norm(h) her satırla nokta çarpımı → logit", 22, C_D).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(sub2))
        h = rng.normal(size=D)
        hb = VGroup(*[Square(0.42, stroke_width=1, stroke_color=BG, fill_opacity=0.9, fill_color=C_D) for _ in range(D)]).arrange(RIGHT, buff=0.02)
        hb.next_to(cells, UP, buff=0.75)
        hl = T("norm(h)", 18, C_D).next_to(hb, LEFT, buff=0.2)
        self.play(FadeOut(name), FadeIn(hb), FadeIn(hl))
        logits = W @ h
        col = VGroup()
        for r in range(V):
            L = logits[r]
            bar = Rectangle(width=max(0.05, (L + 4) * 0.25), height=0.38, stroke_width=0, fill_opacity=0.9, fill_color=C_A if r == int(np.argmax(logits)) else C_I)
            bar.next_to(cells[r], RIGHT, buff=0.5).align_to(cells[r], DOWN)
            bar.shift(RIGHT * bar.width / 2 - RIGHT * 0)
            col.add(bar)
        for r in range(V):
            ghost = hb.copy()
            self.play(ghost.animate.move_to(cells[r]).set_opacity(0.0), GrowFromEdge(col[r], LEFT), run_time=0.28)
        lgl = T("logits [257216]", 18, C_I).next_to(col, UP, buff=0.2)
        best = T("en benzer satır → en yüksek skor", 18, C_A).next_to(col, DOWN, buff=0.2)
        self.play(FadeIn(lgl), FadeIn(best))
        self.wait(2)


def grid_cell(color, size=0.5, opacity=0.9):
    return Square(size, stroke_width=1.5, stroke_color=BG, fill_color=color, fill_opacity=opacity)


class ConvAsPatchLinear(Scene):
    """Conv2d(kernel = stride) = görüntüyü yamalara bölüp her yamaya aynı linear katmanı uygulamak"""
    def construct(self):
        title = T("Conv2d, kernel = stride: her yamaya aynı linear", 30).to_edge(UP, buff=0.4)
        self.play(FadeIn(title))
        rng = np.random.default_rng(2)
        n, k = 6, 3
        vals = rng.uniform(0.15, 0.95, size=(n, n))
        img = VGroup(*[grid_cell(interpolate_color(ManimColor("#f4f2ed"), ManimColor(C_T), vals[r, c]), 0.55) for r in range(n) for c in range(n)])
        img.arrange_in_grid(rows=n, cols=n, buff=0.02).shift(LEFT * 4 + DOWN * 0.2)
        il = T("görüntü 6 × 6 (gerçekte 224 × 224 × 3)", 18, C_F).next_to(img, DOWN, buff=0.25)
        self.play(FadeIn(img), FadeIn(il))
        filt = rng.normal(size=k * k)
        out = VGroup(*[grid_cell(C_F, 0.75, 0.15) for _ in range(4)]).arrange_in_grid(rows=2, cols=2, buff=0.06).shift(RIGHT * 4.6 + DOWN * 0.2)
        ol = T("çıktı 2 × 2 (gerçekte 16 × 16)", 18, C_F).next_to(out, DOWN, buff=0.25)
        self.play(FadeIn(out), FadeIn(ol))
        wl = T("aynı filtre w (9 sayı)", 18, C_D).move_to([0.6, 2.4, 0])
        wvec = VGroup(*[grid_cell(interpolate_color(ManimColor("#f4f2ed"), ManimColor(C_D), min(1, abs(v) / 2)), 0.3) for v in filt]).arrange(RIGHT, buff=0.02).next_to(wl, DOWN, buff=0.15)
        self.play(FadeIn(wl), FadeIn(wvec))
        for pi, (pr, pc) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            idxs = [(pr * k + r) * n + (pc * k + c) for r in range(k) for c in range(k)]
            box = SurroundingRectangle(VGroup(*[img[i] for i in idxs]), color=C_A, buff=0.03, stroke_width=4)
            self.play(Create(box), run_time=0.4)
            flat = VGroup(*[img[i].copy() for i in idxs])
            self.play(flat.animate.arrange(RIGHT, buff=0.02).scale(0.55).next_to(wvec, DOWN, buff=0.35), run_time=0.7)
            dot = MathTex(r"\cdot", color=INK).next_to(flat, RIGHT, buff=0.15)
            v = float((vals.reshape(-1)[idxs] * filt).sum())
            res = T(f"= {v:+.2f}", 20, C_A).next_to(dot, RIGHT, buff=0.12)
            self.play(FadeIn(dot), FadeIn(res), run_time=0.4)
            tgt = out[pi]
            self.play(res.animate.move_to(tgt).scale(0.8), tgt.animate.set_fill(C_A, opacity=0.25), FadeOut(flat), FadeOut(dot), FadeOut(box), run_time=0.7)
        foot = T("yamalar üst üste binmiyor · her yama: 588 sayı · w ile çarpım → 1 kanal (1152 filtre → 1152 kanal)", 19, C_A).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(foot))
        self.wait(2)


class FlattenTranspose(Scene):
    """[C, H, W] → flatten(1) → [C, H·W] → transpose → [H·W, C], eleman takibiyle"""
    def construct(self):
        title = T("flatten(2) + transpose(1, 2): yamalar token'a dönüşüyor", 30).to_edge(UP, buff=0.4)
        self.play(FadeIn(title))
        C, H, W = 3, 2, 2
        cols = [C_T, C_D, C_V]
        cells = {}
        maps = VGroup()
        for c in range(C):
            g = VGroup()
            for r in range(H):
                for w in range(W):
                    sq = grid_cell(cols[c], 0.6)
                    lab = T(f"{r * W + w}", 16, "#ffffff").move_to(sq)
                    cell = VGroup(sq, lab)
                    cells[(c, r, w)] = cell
                    g.add(cell)
            g.arrange_in_grid(rows=H, cols=W, buff=0.04)
            maps.add(g)
        maps.arrange(RIGHT, buff=0.6).shift(UP * 0.6)
        l1 = T("[C, H, W] = [3, 2, 2] · her kanal bir harita, sayılar yama konumu", 19, C_F).next_to(maps, DOWN, buff=0.4)
        self.play(FadeIn(maps), FadeIn(l1))
        self.wait(0.6)
        # flatten: her kanal bir satır
        for c in range(C):
            for r in range(H):
                for w in range(W):
                    pass
        rows = []
        anims = []
        for c in range(C):
            for p in range(H * W):
                r, w = divmod(p, W)
                tgt = np.array([-1.0 + p * 0.66, 1.0 - c * 0.66, 0])
                anims.append(cells[(c, r, w)].animate.move_to(tgt))
        l2 = T("flatten(2) → [C, H·W] = [3, 4] · satır = kanal", 19, C_F).move_to(l1)
        self.play(*anims, Transform(l1, l2), run_time=1.6)
        self.wait(0.6)
        anims = []
        for c in range(C):
            for p in range(H * W):
                r, w = divmod(p, W)
                tgt = np.array([-0.66 + c * 0.66, 1.4 - p * 0.66, 0])
                anims.append(cells[(c, r, w)].animate.move_to(tgt))
        l3 = T("transpose(1, 2) → [H·W, C] = [4, 3] · satır = yama = token", 19, C_A).move_to(l1).shift(DOWN * 0.6)
        self.play(*anims, Transform(l1, l3), run_time=1.6)
        brace = Brace(VGroup(*[cells[(c, 0, 0)] for c in range(C)]), RIGHT, color=C_A)
        bl = T("token 0 = yama 0'ın C sayısı", 18, C_A).next_to(brace, RIGHT)
        self.play(GrowFromCenter(brace), FadeIn(bl))
        self.wait(2)


class MultiHeadReshape(Scene):
    """[T, D] → view [T, H, hd] → transpose [H, T, hd]: her head kendi bloğunda"""
    def construct(self):
        title = T("Multi-head: view ve transpose", 30).to_edge(UP, buff=0.4)
        self.play(FadeIn(title))
        Tn, H, hd = 3, 2, 3
        hc = [C_T, C_D]
        cells = {}
        for t in range(Tn):
            for d in range(H * hd):
                sq = grid_cell(hc[d // hd], 0.55)
                lab = T(f"t{t}", 13, "#ffffff").move_to(sq)
                cells[(t, d)] = VGroup(sq, lab).move_to([-1.5 + d * 0.6, 1.2 - t * 0.6, 0])
        self.play(*[FadeIn(c) for c in cells.values()])
        l = T("q_proj(x): [T, D] = [3, 6]", 20, C_F).move_to([0, -1.2, 0])
        self.play(FadeIn(l))
        self.wait(0.5)
        # view: D'yi H × hd olarak ayır (araya boşluk)
        anims = [c.animate.move_to([-1.8 + (d // hd) * 2.1 + (d % hd) * 0.6, 1.2 - t * 0.6, 0]) for (t, d), c in cells.items()]
        l2 = T("view(B, T, H, hd): [3, 2, 3] · bellek aynı, sadece okunuş", 20, C_F).move_to(l)
        self.play(*anims, Transform(l, l2), run_time=1.4)
        self.wait(0.5)
        # transpose: head'ler ayrı bloklara
        anims = [c.animate.move_to([-3.2 + (d // hd) * 4.0 + (d % hd) * 0.6, 1.2 - t * 0.6, 0]) for (t, d), c in cells.items()]
        l3 = T("transpose(1, 2): [H, T, hd] = [2, 3, 3] · her head kendi T × hd matrisi", 20, C_A).move_to(l)
        self.play(*anims, Transform(l, l3), run_time=1.4)
        h1 = T("head 0", 18, C_T).move_to([-2.6, 2.0, 0]); h2 = T("head 1", 18, C_D).move_to([1.4, 2.0, 0])
        self.play(FadeIn(h1), FadeIn(h2))
        foot = T("q @ kᵀ her head'de ayrı: [H, T, T] skor · sonra transpose ve reshape ile geri [T, D]", 19, C_A).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(foot))
        self.wait(2)

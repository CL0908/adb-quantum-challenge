"""Three short loops for the home page's "Three models, one inside the other" block (Manim Community Edition).
Every number on screen is read from the committed site data (public/data/*.json); the dots, rings and world ticks are
schematic and labelled as such. 4:3 at 720x540, sized for the panel they sit in. Render from frontend/:
  manim -qh animations/nest_scenes.py Rdpo SmallWorld Twin
then python animations/encode.py writes public/anim/<name>.mp4/.webm and a poster .webp."""
import json
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, UR, Circle, Create, DashedLine, Dot, FadeIn, FadeOut, LaggedStart, Line,
                   Rectangle, Scene, Text, Transform, ValueTracker, VGroup, always_redraw, config, smooth)

DATA = Path(__file__).resolve().parents[1] / "public" / "data"
NEST = json.loads((DATA / "nest.json").read_text(encoding="utf-8"))
FALLBACK = json.loads((DATA / "fallback.json").read_text(encoding="utf-8"))["adb-base|joint_exhaustion|16384"]

config.background_color = "#FFFFFF"
config.pixel_width, config.pixel_height, config.frame_rate = 720, 540, 30
config.frame_height = 8.0
config.frame_width = 8.0 * 720 / 540   # 10.67 x 8 units
INK, MUTED, HAIR = "#111413", "#6A6E69", "#D9D5CC"
NAVY, TERRA, TEAL, GREY = "#1F3557", "#C8553A", "#0E8074", "#9AA0A6"
FONT = "Inter"


def T(s, size=28, color=INK, weight="NORMAL"):
    # rendered large and scaled down: Pango drops word gaps at small sizes
    return Text(s, font=FONT, font_size=size * 4, color=color, weight=weight).scale(0.25)


def sci(x):
    e = int(np.floor(np.log10(x)))
    return f"{x / 10 ** e:.2f}×10" + str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))


def note(scene):
    scene.add(T("schematic · numbers from the committed runs", 20, MUTED).to_edge(DOWN, buff=0.22))


class Rdpo(Scene):
    """Plain Monte Carlo misses the rare joint loss; RDPO tilts paths toward it and reweights them."""
    def construct(self):
        note(self)
        rng = np.random.default_rng(7)
        S = 3.3
        box = Rectangle(width=S, height=S, stroke_color=HAIR, stroke_width=2).move_to(LEFT * 2.4 + UP * 0.55)
        corner = Rectangle(width=0.5, height=0.5, stroke_width=0, fill_color=TERRA, fill_opacity=0.3)
        corner.move_to(box.get_corner(UR) + np.array([-0.25, -0.25, 0]))
        xl = T("Kyrgyz loss", 22, MUTED).next_to(box, DOWN, buff=0.12)
        yl = T("Tajik loss", 22, MUTED).rotate(np.pi / 2).next_to(box, LEFT, buff=0.12)
        cl = T("both exhausted", 22, TERRA).next_to(corner, RIGHT, buff=0.12).align_to(corner, UP)
        title = T("Plain Monte Carlo", 34, INK, "MEDIUM").to_edge(UP, buff=0.4)
        self.add(box, corner, xl, yl, cl)
        self.play(FadeIn(title), run_time=0.4)

        def place(u, v):
            return box.get_corner(DOWN + LEFT) + np.array([u * S, v * S, 0])
        pts = rng.uniform(0.03, 0.84, size=(220, 2))  # the corner holds about 1 in 31,600 of the mass: none land there
        grey = VGroup(*[Dot(place(u, v), radius=0.035, color=GREY) for u, v in pts])
        self.play(LaggedStart(*[FadeIn(d) for d in grey], lag_ratio=0.012), run_time=1.6)
        side = RIGHT * 2.75 + UP * 0.55
        mc = VGroup(T("16,384 paths", 28), T(f"{FALLBACK['mc']['hits']} in the corner", 28, TERRA),
                    T("estimate 0", 28, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.28).move_to(side)
        self.play(FadeIn(mc), run_time=0.5)
        self.wait(1.0)

        title2 = T("RDPO, a learned sampler", 34, TEAL, "MEDIUM").move_to(title)
        self.play(FadeOut(grey), FadeOut(mc), Transform(title, title2), run_time=0.6)
        u, v = 1 - rng.beta(1.3, 6.0, 150), 1 - rng.beta(1.3, 6.0, 150)
        teal = VGroup(*[Dot(place(a, b), radius=0.04, color=TEAL) for a, b in zip(u, v)])
        self.play(LaggedStart(*[FadeIn(d) for d in teal], lag_ratio=0.01), run_time=1.6)
        self.play(*[d.animate.scale(0.45) for d in teal], run_time=0.6)
        w = T("each path reweighted: unbiased", 24, MUTED).next_to(xl, DOWN, buff=0.18)
        self.play(FadeIn(w), run_time=0.4)
        est = ValueTracker(FALLBACK["estimate"] * 2.4)
        ex = T("exact", 24, MUTED)
        exv = T(sci(FALLBACK["exact"]), 30, INK)
        rl = T("RDPO", 24, TEAL)
        grp = VGroup(ex, exv, rl).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to(side + UP * 0.35)
        live = always_redraw(lambda: T(sci(est.get_value()), 30, TEAL, "MEDIUM").next_to(rl, DOWN, buff=0.16, aligned_edge=LEFT))
        self.play(FadeIn(grp), FadeIn(live), run_time=0.4)
        self.play(est.animate.set_value(FALLBACK["estimate"]), run_time=1.8, rate_func=smooth)
        self.wait(1.4)
        self.play(*[FadeOut(m) for m in (teal, grp, live, w, title)], run_time=0.5)


class SmallWorld(Scene):
    """Remove the payout caps, solve the pool exactly in one pass; alone it overstates the tail, cap tables fix it."""
    def construct(self):
        note(self)
        c = LEFT * 3.1 + UP * 0.4
        R, n = 1.45, 28
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
        pos = [c + R * np.array([np.cos(a), np.sin(a), 0]) for a in ang]
        nodes = VGroup(*[Dot(p, radius=0.07, color=NAVY) for p in pos])
        ring = Circle(radius=R, stroke_color=HAIR, stroke_width=2).move_to(c)
        caps = VGroup(*[Line(p + np.array([-0.09, 0.14, 0]), p + np.array([0.09, 0.14, 0]), color=TERRA, stroke_width=3) for p in pos])
        title = T("28-country ring pool", 32, INK, "MEDIUM").to_edge(UP, buff=0.4)
        self.play(Create(ring), LaggedStart(*[FadeIn(d) for d in nodes], lag_ratio=0.03), FadeIn(title), run_time=1.1)
        self.play(LaggedStart(*[FadeIn(k) for k in caps], lag_ratio=0.02), run_time=0.5)
        self.play(FadeOut(caps), Transform(title, T("small world: caps removed", 32, INK, "MEDIUM").move_to(title)), run_time=0.7)
        self.play(Transform(title, T("solved exactly, one pass", 32, TEAL, "MEDIUM").move_to(title)),
                  *[d.animate.move_to(c + (p - c) * 0.5) for d, p in zip(nodes, pos)], ring.animate.scale(0.5), run_time=0.8)
        self.play(*[d.animate.move_to(c) for d in nodes], ring.animate.scale(0.15), run_time=0.6)
        self.play(FadeOut(ring), FadeOut(nodes), run_time=0.3)

        # each answer divided by the exact value, log axis 0.5x to 100x
        bx, by, H = -0.6, -2.4, 4.6
        y = lambda r: by + H * (np.log10(r) - np.log10(0.5)) / (np.log10(100) - np.log10(0.5))
        axis = Line([bx, by, 0], [bx, by + H, 0], color=HAIR)
        ticks = VGroup(*[VGroup(Line([bx - 0.1, y(r), 0], [bx, y(r), 0], color=MUTED), T(f"{r:g}×", 22, MUTED).move_to([bx - 0.5, y(r), 0])) for r in (1, 10, 100)])
        exact = DashedLine([bx, y(1), 0], [5.0, y(1), 0], color=INK, dash_length=0.1)
        el = T("exact", 22, INK).next_to(exact.get_end(), UP, buff=0.08).shift(LEFT * 0.35)
        self.play(Create(axis), FadeIn(ticks), Create(exact), FadeIn(el), run_time=0.6)
        g, gt = NEST["answer_over_truth"]["guide"], NEST["answer_over_truth"]["guide_tables"]
        b1 = Rectangle(width=1.1, height=y(g) - by, stroke_width=0, fill_color=TERRA, fill_opacity=0.9).move_to([bx + 1.1, by + (y(g) - by) / 2, 0])
        l1 = VGroup(T("alone", 22, TERRA), T(f"{g:.0f}×", 32, TERRA, "MEDIUM")).arrange(DOWN, buff=0.06).next_to(b1, UP, buff=0.1)
        self.play(FadeIn(b1, shift=UP * 0.3), FadeIn(l1), run_time=0.7)
        self.wait(0.7)
        b2 = Rectangle(width=1.1, height=y(gt) - by, stroke_width=0, fill_color=TEAL, fill_opacity=0.9).move_to([bx + 3.2, by + (y(gt) - by) / 2, 0])
        l2 = VGroup(T("+ cap tables", 22, TEAL), T(f"{gt:.3f}×", 32, TEAL, "MEDIUM")).arrange(DOWN, buff=0.06).next_to(b2, UP, buff=0.45)
        self.play(FadeIn(b2, shift=UP * 0.2), FadeIn(l2), run_time=0.7)
        self.wait(1.6)
        self.play(*[FadeOut(m) for m in (axis, ticks, exact, el, b1, l1, b2, l2, title)], run_time=0.5)


class Twin(Scene):
    """Many possible worlds instead of one; counting them raises the pool's tail probability."""
    def construct(self):
        note(self)
        tw = NEST["twin"]["exceedance"]
        x0, x1, lo, hi, yb = -4.8, 4.8, 0.0, 0.04, -1.5
        X = lambda p: x0 + (x1 - x0) * (p - lo) / (hi - lo)
        axis = Line([x0, yb, 0], [x1, yb, 0], color=HAIR, stroke_width=2)
        ticks = VGroup(*[VGroup(Line([X(p), yb, 0], [X(p), yb - 0.12, 0], color=MUTED), T(f"{p * 100:g}%", 22, MUTED).move_to([X(p), yb - 0.4, 0])) for p in (0, 0.01, 0.02, 0.03, 0.04)])
        cap = T("chance the pool pays over US$80m", 24, MUTED).move_to([0, yb - 0.95, 0])
        title = T("one world, inputs fixed", 32, NAVY, "MEDIUM").to_edge(UP, buff=0.4)
        self.play(Create(axis), FadeIn(ticks), FadeIn(cap), FadeIn(title), run_time=0.8)
        fx = X(tw["fixed"])
        fixed = Line([fx, yb, 0], [fx, yb + 3.0, 0], color=NAVY, stroke_width=5)
        fl = T(f"{tw['fixed'] * 100:.2f}%", 30, NAVY, "MEDIUM").next_to(fixed, UP, buff=0.1).shift(LEFT * 0.45)
        self.play(Create(fixed), FadeIn(fl), run_time=0.7)
        self.wait(0.6)
        self.play(Transform(title, T("the twin: many possible worlds", 32, TEAL, "MEDIUM").move_to(title)), run_time=0.5)
        rng = np.random.default_rng(3)
        s = rng.lognormal(0, 0.55, 90)  # illustrative right-skewed spread whose mean is the committed value
        p = s / s.mean() * tw["uncertain"]
        p = p[(p > lo) & (p < hi)]
        worlds = VGroup(*[Line([X(q), yb, 0], [X(q), yb + 1.0 + 0.6 * rng.random(), 0], color=TEAL, stroke_width=2, stroke_opacity=0.5) for q in p])
        self.play(LaggedStart(*[Create(w) for w in worlds], lag_ratio=0.02), run_time=1.8)
        mean = ValueTracker(tw["fixed"])
        m = always_redraw(lambda: Line([X(mean.get_value()), yb, 0], [X(mean.get_value()), yb + 3.0, 0], color=TEAL, stroke_width=5))
        # the moving label appears once it has cleared the fixed one
        ml = always_redraw(lambda: T(f"{mean.get_value() * 100:.2f}%", 30, TEAL, "MEDIUM").next_to(m, UP, buff=0.1).shift(RIGHT * 0.45)
                           .set_opacity(min(1.0, max(0.0, (mean.get_value() - tw["fixed"]) / 0.0025))))
        self.add(m, ml)
        self.play(mean.animate.set_value(tw["uncertain"]), run_time=1.6, rate_func=smooth)
        gain = T(f"averaged over the worlds  +{tw['rel'] * 100:.0f}%", 28, INK).next_to(title, DOWN, buff=0.35)
        self.play(FadeIn(gain), run_time=0.4)
        self.wait(1.8)
        self.play(*[FadeOut(o) for o in (worlds, m, ml, fixed, fl, gain, title, axis, ticks, cap)], run_time=0.5)

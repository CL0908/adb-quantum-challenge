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

        title2 = T("RDPO: sends paths where it matters", 34, TEAL, "MEDIUM").move_to(title)
        self.play(FadeOut(grey), FadeOut(mc), Transform(title, title2), run_time=0.6)
        u, v = 1 - rng.beta(1.3, 6.0, 150), 1 - rng.beta(1.3, 6.0, 150)
        teal = VGroup(*[Dot(place(a, b), radius=0.04, color=TEAL) for a, b in zip(u, v)])
        self.play(LaggedStart(*[FadeIn(d) for d in teal], lag_ratio=0.01), run_time=1.6)
        self.play(*[d.animate.scale(0.45) for d in teal], run_time=0.6)
        w = T("each path reweighted: unbiased", 24, MUTED).next_to(xl, DOWN, buff=0.18)
        self.play(FadeIn(w), run_time=0.4)
        ex = T("exact", 24, MUTED)
        exv = T(sci(FALLBACK["exact"]), 30, INK)
        rl = T("RDPO, 16,384 paths", 24, TEAL)
        live = T(sci(FALLBACK["estimate"]), 30, TEAL, "MEDIUM")
        grp = VGroup(ex, exv, rl, live).arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to(side + UP * 0.1)
        self.play(FadeIn(grp[:2]), run_time=0.4)
        self.wait(0.6)
        self.play(FadeIn(grp[2:]), run_time=0.5)
        self.wait(2.2)
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
        self.play(Transform(title, T("a rough map, solved in one pass", 32, TEAL, "MEDIUM").move_to(title)),
                  *[d.animate.move_to(c + (p - c) * 0.5) for d, p in zip(nodes, pos)], ring.animate.scale(0.5), run_time=0.8)
        self.play(*[d.animate.move_to(c) for d in nodes], ring.animate.scale(0.15), run_time=0.6)
        self.play(FadeOut(ring), FadeOut(nodes), run_time=0.3)

        # each answer divided by the exact value, log axis 0.5x to 100x
        bx, by, H = -0.6, -2.4, 4.6
        y = lambda r: by + H * (np.log10(r) - np.log10(0.5)) / (np.log10(100) - np.log10(0.5))
        axis = Line([bx, by, 0], [bx, by + H, 0], color=HAIR)
        ticks = VGroup(*[VGroup(Line([bx - 0.1, y(r), 0], [bx, y(r), 0], color=MUTED), T(f"{r:g}×", 22, MUTED).move_to([bx - 0.5, y(r), 0])) for r in (1, 10, 100)])
        exact = DashedLine([bx, y(1), 0], [5.0, y(1), 0], color=INK, dash_length=0.1)
        el = VGroup(T("2M-path", 22, INK), T("reference", 22, INK)).arrange(DOWN, buff=0.05, aligned_edge=LEFT).next_to(exact.get_end(), DOWN, buff=0.12).align_to(exact.get_end(), RIGHT).shift(LEFT * 0.15)
        self.play(Create(axis), FadeIn(ticks), Create(exact), FadeIn(el),
                  Transform(title, T("how close is the map to the real pool?", 30, INK, "MEDIUM").move_to(title)), run_time=0.6)
        g, gt = NEST["answer_over_truth"]["guide"], NEST["answer_over_truth"]["guide_tables"]
        # each answer as a stem from the 1x line: the length is how far off it is
        b1 = VGroup(Line([bx + 1.1, y(1), 0], [bx + 1.1, y(g), 0], color=TERRA, stroke_width=8), Dot([bx + 1.1, y(g), 0], radius=0.13, color=TERRA))
        l1 = VGroup(T("caps ignored", 22, TERRA), T(f"{g:.0f}× too high", 32, TERRA, "MEDIUM")).arrange(DOWN, buff=0.06).next_to(b1, RIGHT, buff=0.2).align_to(b1, UP)
        self.play(FadeIn(b1, shift=UP * 0.3), FadeIn(l1), run_time=0.7)
        self.wait(0.7)
        b2 = Dot([bx + 3.6, y(gt), 0], radius=0.13, color=TEAL)
        l2 = VGroup(T("with cap tables", 22, TEAL), T(f"{gt:.3f}×, on target", 32, TEAL, "MEDIUM")).arrange(DOWN, buff=0.06, aligned_edge=LEFT).next_to(b2, UP, buff=0.3)
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
        fixed = Line([fx, yb, 0], [fx, yb + 2.0, 0], color=NAVY, stroke_width=5)
        fl = T(f"{tw['fixed'] * 100:.2f}%", 30, NAVY, "MEDIUM").next_to(fixed, UP, buff=0.1).shift(LEFT * 0.45)
        self.play(Create(fixed), FadeIn(fl), run_time=0.7)
        self.wait(0.6)
        self.play(Transform(title, T("the twin: many possible worlds", 32, TEAL, "MEDIUM").move_to(title)), run_time=0.5)
        rng = np.random.default_rng(3)
        # many possible worlds as markers above the axis: schematic, not placed on the probability scale
        worlds = VGroup(*[Dot([x0 + 0.4 + (i % 24) * 0.38, 2.1 - (i // 24) * 0.3, 0], radius=0.06, color=TEAL, fill_opacity=0.35 + 0.5 * rng.random()) for i in range(72)])
        wl = T("possible worlds; darker = more likely given the data", 22, MUTED).next_to(worlds, DOWN, buff=0.14).align_to(worlds, LEFT)
        self.play(LaggedStart(*[FadeIn(w) for w in worlds], lag_ratio=0.015), FadeIn(wl), run_time=1.6)
        ux = X(tw["uncertain"])
        m = Line([ux, yb, 0], [ux, yb + 2.0, 0], color=TERRA, stroke_width=5)
        ml = T(f"{tw['uncertain'] * 100:.2f}%", 30, TERRA, "MEDIUM").next_to(m, UP, buff=0.1).shift(RIGHT * 0.45)
        self.play(Create(m), FadeIn(ml), run_time=0.9)
        gain = T(f"+{tw['rel'] * 100:.0f}% vs inputs fixed", 28, INK).to_edge(UP, buff=1.25)
        self.play(FadeIn(gain), run_time=0.4)
        self.wait(1.8)
        self.play(*[FadeOut(o) for o in (worlds, wl, m, ml, fixed, fl, gain, title, axis, ticks, cap)], run_time=0.5)

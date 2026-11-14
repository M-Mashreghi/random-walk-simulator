#!/usr/bin/env python3
"""
Random Walk Simulator
=====================

An interactive, customizable random walk simulator built with NumPy and Matplotlib.

First example: a coin. Heads moves the walker one step up, tails one step down.
The window shows
  * many sample paths over time, with the theoretical mean and a +/- 2 sigma band,
  * the distribution of positions at the current step (simulation vs exact theory),
  * the tree of ALL possible paths for the first few flips (Pascal's triangle for a coin),
    where you can flip the coin yourself and watch your path.

Other built-in models: a six-sided die, custom steps ("value:weight"),
Gaussian (continuous) steps and a 2D lattice walk.

Run
---
    python random_walk.py                      # interactive window
    python random_walk.py --model die          # start with another model
    python random_walk.py --steps 500 --walks 1000 --p 0.55 --seed 7
    python random_walk.py --save walk.png      # no window, just save a picture

Add your own model
------------------
Subclass ``Model`` (see ``CoinModel`` below) and register it in ``MODELS``.
A discrete model only has to return a list of ``Outcome(step, prob, label)``;
the paths, histogram, exact theoretical distribution and tree are built automatically.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    model: str = "coin"
    steps: int = 100          # n, number of steps per walk
    walks: int = 200          # number of independent walks
    seed: int = 2026
    # coin
    p: float = 0.5            # P(heads)
    up: float = 1.0           # step for heads
    down: float = -1.0        # step for tails
    # die: step for faces 1..6
    die_faces: str = "-3, -2, -1, 1, 2, 3"
    # custom: "value:weight, value:weight, ..."
    custom: str = "2:1, -1:2"
    # gaussian
    mu: float = 0.0
    sigma: float = 1.0
    # 2D lattice: weights for Right, Left, Up, Down
    weights2d: str = "1, 1, 1, 1"
    # display
    speed: float = 60.0       # animation speed, steps per second
    tree_depth: int = 6
    show_band: bool = True


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass
class Outcome:
    step: float | tuple[float, float]
    prob: float
    label: str      # short label shown in sequences, e.g. "H" / "T"


def _normalize(outs: list[Outcome]) -> list[Outcome]:
    total = sum(o.prob for o in outs)
    if not total > 0:
        raise ValueError("Weights must add up to a positive number.")
    return [Outcome(o.step, o.prob / total, o.label) for o in outs]


def _numbers(text: str) -> list[float]:
    return [float(x) for x in text.replace(";", ",").split(",") if x.strip()]


class Model:
    """Base class. Override ``outcomes`` (discrete) or ``sample`` + ``moments`` (continuous)."""

    name = "model"
    dim = 1
    discrete = True
    rule = ""

    def outcomes(self, s: Settings) -> list[Outcome]:
        raise NotImplementedError

    def sample(self, rng: np.random.Generator, shape: tuple[int, int], s: Settings):
        """Return (steps, outcome_index). steps has shape (walks, n) or (walks, n, 2)."""
        outs = self.outcomes(s)
        probs = np.array([o.prob for o in outs], dtype=float)
        idx = rng.choice(len(outs), size=shape, p=probs / probs.sum())
        table = np.array([o.step for o in outs], dtype=float)
        return table[idx], idx

    def moments(self, s: Settings):
        """Mean and variance of a single step (1D) or ((mx, my), (vx, vy)) for 2D."""
        outs = self.outcomes(s)
        p = np.array([o.prob for o in outs])
        v = np.array([o.step for o in outs], dtype=float)
        if self.dim == 1:
            mu = float(p @ v)
            return mu, float(p @ v**2 - mu**2)
        mu = p @ v
        return tuple(mu), tuple(p @ v**2 - mu**2)


class CoinModel(Model):
    name = "Coin"
    rule = "S(t+1) = S(t) + (heads ? up : down)"

    def outcomes(self, s):
        return [Outcome(s.up, s.p, "H"), Outcome(s.down, 1 - s.p, "T")]


class DieModel(Model):
    name = "Die"
    rule = "S(t+1) = S(t) + step(face)"

    def outcomes(self, s):
        faces = _numbers(s.die_faces)
        if len(faces) != 6:
            raise ValueError("Die faces need exactly 6 numbers, e.g. -3,-2,-1,1,2,3")
        return [Outcome(v, 1 / 6, str(i + 1)) for i, v in enumerate(faces)]


class CustomModel(Model):
    name = "Custom steps"
    rule = "S(t+1) = S(t) + X,  X ~ your weights"

    def outcomes(self, s):
        outs = []
        for item in s.custom.replace(";", ",").split(","):
            if not item.strip():
                continue
            parts = item.split(":")
            v = float(parts[0])
            w = float(parts[1]) if len(parts) > 1 else 1.0
            if w < 0:
                raise ValueError("Weights must be >= 0")
            outs.append(Outcome(v, w, f"{v:+g}"))
        if not outs:
            raise ValueError('Write steps like "2:1, -1:2"')
        return _normalize(outs)


class GaussianModel(Model):
    name = "Gaussian"
    discrete = False
    rule = "S(t+1) = S(t) + N(mu, sigma^2)"

    def sample(self, rng, shape, s):
        return rng.normal(s.mu, s.sigma, size=shape), None

    def moments(self, s):
        return s.mu, s.sigma**2


class Lattice2DModel(Model):
    name = "2D lattice"
    dim = 2
    rule = "(x, y) -> (x+-1, y) or (x, y+-1)"

    def outcomes(self, s):
        w = _numbers(s.weights2d)
        if len(w) != 4:
            raise ValueError("2D weights need 4 numbers: right, left, up, down")
        return _normalize([Outcome((1, 0), w[0], "R"), Outcome((-1, 0), w[1], "L"),
                           Outcome((0, 1), w[2], "U"), Outcome((0, -1), w[3], "D")])


MODELS: dict[str, Model] = {
    "coin": CoinModel(),
    "die": DieModel(),
    "custom": CustomModel(),
    "gaussian": GaussianModel(),
    "2d": Lattice2DModel(),
}

# ---------------------------------------------------------------------------
# Simulation and theory
# ---------------------------------------------------------------------------


@dataclass
class Result:
    model: Model
    settings: Settings
    pos: np.ndarray                 # (walks, n+1) or (walks, n+1, 2)
    idx: np.ndarray | None          # (walks, n) outcome index per step, discrete only
    outcomes: list[Outcome] | None
    mu: object
    var: object
    first_return: np.ndarray = field(default=None)   # first step back at 0 (n+1 = never)


def simulate(s: Settings) -> Result:
    model = MODELS[s.model]
    rng = np.random.default_rng(s.seed)
    steps, idx = model.sample(rng, (s.walks, s.steps), s)
    if model.dim == 1:
        pos = np.concatenate([np.zeros((s.walks, 1)), np.cumsum(steps, axis=1)], axis=1)
        at_zero = np.isclose(pos[:, 1:], 0.0, atol=1e-9)
    else:
        pos = np.concatenate([np.zeros((s.walks, 1, 2)), np.cumsum(steps, axis=1)], axis=1)
        at_zero = np.all(np.isclose(pos[:, 1:], 0.0, atol=1e-9), axis=2)
    first = np.where(at_zero.any(axis=1), at_zero.argmax(axis=1) + 1, s.steps + 1)
    outs = model.outcomes(s) if model.discrete else None
    mu, var = model.moments(s)
    return Result(model, s, pos, idx, outs, mu, var, first)


class ExactDistribution:
    """Exact distribution of S_t for discrete 1D models whose steps lie on a lattice.

    Uses repeated convolution and caches the last result, so animating forward is cheap.
    Returns None when the steps are not on a reasonable lattice.
    """

    def __init__(self, outcomes: list[Outcome], max_width: int = 400):
        self.ok = False
        steps = np.array([o.step for o in outcomes], dtype=float)
        probs = np.array([o.prob for o in outcomes], dtype=float)
        ints = np.round(steps * 1_000_000).astype(np.int64)
        nz = ints[ints != 0]
        unit = int(np.gcd.reduce(np.abs(nz))) if nz.size else 1_000_000
        k = ints // unit
        if k.max() - k.min() > max_width:
            return
        self.unit = unit / 1_000_000
        self.kmin = int(k.min())
        self.kernel = np.zeros(int(k.max() - k.min()) + 1)
        np.add.at(self.kernel, k - k.min(), probs)
        self.t, self.pmf = 0, np.array([1.0])
        self.ok = True

    def at(self, t: int):
        if not self.ok:
            return None
        if t < self.t:
            self.t, self.pmf = 0, np.array([1.0])
        while self.t < t:
            self.pmf = np.convolve(self.pmf, self.kernel)
            self.t += 1
        values = (self.kmin * t + np.arange(self.pmf.size)) * self.unit
        mask = self.pmf > 1e-12
        return values[mask], self.pmf[mask]


def normal_cdf(x, mu, sd):
    return 0.5 * (1 + np.vectorize(math.erf)((np.asarray(x) - mu) / (sd * math.sqrt(2))))


def path_tree(outcomes: list[Outcome], depth: int):
    """All reachable positions per level: list of dicts {position: (count, probability)}."""
    levels = [{0.0: (1, 1.0)}]
    for _ in range(depth):
        nxt: dict[float, list] = {}
        for x, (c, p) in levels[-1].items():
            for o in outcomes:
                if o.prob <= 0:
                    continue
                key = round(x + o.step, 9)
                cc, pp = nxt.get(key, (0, 0.0))
                nxt[key] = (cc + c, pp + p * o.prob)
        levels.append(nxt)
    return levels


# ---------------------------------------------------------------------------
# Interactive window
# ---------------------------------------------------------------------------


class App:
    BG = "#F4F6F9"
    INK = "#172031"
    MUTED = "#5B6578"
    UP = "#C2701A"
    DOWN = "#3E55C6"
    BAND = "#17203112"

    def __init__(self, settings: Settings, interactive: bool = True):
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.colors import LinearSegmentedColormap

        self.plt = plt
        self.LineCollection = LineCollection
        self.s = settings
        self.interactive = interactive
        self.cmap = LinearSegmentedColormap.from_list("walk", [self.DOWN, "#8C95A6", self.UP])
        self.highlight = 0
        self.manual: list[int] = []
        self.playing = False
        self.t = settings.steps
        self.res: Result | None = None
        self.exact: ExactDistribution | None = None

        plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#C9D0DB", "axes.labelcolor": self.MUTED,
                             "xtick.color": self.MUTED, "ytick.color": self.MUTED})
        self.fig = plt.figure(figsize=(15, 9.4), facecolor=self.BG)
        try:
            self.fig.canvas.manager.set_window_title("Random Walk Simulator")
        except Exception:
            pass
        self.ax_paths = self.fig.add_axes([0.30, 0.47, 0.47, 0.45])
        self.ax_hist = self.fig.add_axes([0.785, 0.47, 0.17, 0.45])
        self.ax_tree = self.fig.add_axes([0.30, 0.06, 0.47, 0.30])
        for ax in (self.ax_paths, self.ax_hist, self.ax_tree):
            ax.set_facecolor("white")
        self.title = self.fig.text(0.30, 0.955, "", fontsize=15, fontweight="bold", color=self.INK)
        self.subtitle = self.fig.text(0.30, 0.932, "", fontsize=9.5, color=self.MUTED)
        self.stats_text = self.fig.text(0.785, 0.36, "", fontsize=9, color=self.INK, va="top",
                                        family="monospace", linespacing=1.55)
        self.seq_text = self.fig.text(0.30, 0.402, "", fontsize=8.5, color=self.MUTED, family="monospace")
        self.error_text = self.fig.text(0.015, 0.012, "", fontsize=9, color="#C2362B")

        if interactive:
            self._build_widgets()
            self.fig.canvas.mpl_connect("button_press_event", self._on_click)
            self.timer = self.fig.canvas.new_timer(interval=30)
            self.timer.add_callback(self._tick)
        self.resimulate(restart=False)

    # ---------------- widgets ----------------
    def _build_widgets(self):
        from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox

        s, f = self.s, self.fig
        f.text(0.015, 0.955, "Random Walk", fontsize=15, fontweight="bold", color=self.INK)
        f.text(0.015, 0.932, "Model", fontsize=9, color=self.MUTED)
        keys = list(MODELS)
        ax = f.add_axes([0.015, 0.775, 0.20, 0.15], facecolor=self.BG)
        ax.set_frame_on(False)
        self.radio = RadioButtons(ax, [MODELS[k].name for k in keys], active=keys.index(s.model),
                                  activecolor=self.INK)
        self.radio.on_clicked(lambda label: self._set("model", keys[[MODELS[k].name for k in keys].index(label)], restart=True))

        self.sliders = {}

        def slider(name, label, y, lo, hi, val, step=None, fmt=None):
            a = f.add_axes([0.095, y, 0.115, 0.022], facecolor="white")
            sl = Slider(a, label, lo, hi, valinit=val, valstep=step, color=self.INK, initcolor="none")
            sl.label.set_fontsize(8.5)
            sl.valtext.set_fontsize(8.5)
            if fmt:
                sl.valtext.set_text(fmt(val))
            sl.on_changed(lambda v, n=name: self._set(n, v))
            self.sliders[name] = sl

        y = 0.74
        f.text(0.015, y + 0.012, "Coin", fontsize=8.5, color=self.MUTED, fontweight="bold")
        slider("p", "P(heads)", y - 0.025, 0, 1, s.p, 0.01)
        slider("up", "Heads step", y - 0.055, -3, 3, s.up, 0.5)
        slider("down", "Tails step", y - 0.085, -3, 3, s.down, 0.5)
        y = 0.615
        f.text(0.015, y + 0.012, "Gaussian", fontsize=8.5, color=self.MUTED, fontweight="bold")
        slider("mu", "Mean (mu)", y - 0.025, -1, 1, s.mu, 0.01)
        slider("sigma", "Std (sigma)", y - 0.055, 0.05, 3, s.sigma, 0.05)

        self.boxes = {}

        def box(name, label, y):
            a = f.add_axes([0.095, y, 0.115, 0.03])
            tb = TextBox(a, label, initial=getattr(s, name), color="white", hovercolor="#EEF1F5")
            tb.label.set_fontsize(8.5)
            tb.on_submit(lambda text, n=name: self._set(n, text))
            self.boxes[name] = tb

        y = 0.515
        f.text(0.015, y + 0.01, "Die / Custom / 2D  (press Enter)", fontsize=8.5, color=self.MUTED, fontweight="bold")
        box("die_faces", "Die faces 1-6", y - 0.035)
        box("custom", "value:weight", y - 0.072)
        box("weights2d", "2D R,L,U,D", y - 0.109)

        y = 0.36
        f.text(0.015, y + 0.012, "Simulation", fontsize=8.5, color=self.MUTED, fontweight="bold")
        slider("steps", "Steps (n)", y - 0.025, 5, 1000, s.steps, 1)
        slider("walks", "Walks", y - 0.055, 1, 2000, s.walks, 1)
        slider("seed", "Seed", y - 0.085, 0, 9999, s.seed, 1)
        slider("speed", "Speed (steps/s)", y - 0.115, 1, 500, s.speed, 1)

        def button(label, rect, cb, primary=False):
            a = f.add_axes(rect)
            b = Button(a, label, color=self.INK if primary else "white", hovercolor="#3A4458" if primary else "#E4E8EF")
            b.label.set_fontsize(9)
            if primary:
                b.label.set_color("white")
            b.on_clicked(cb)
            return b

        self.btn_play = button("Play", [0.015, 0.175, 0.062, 0.035], self._toggle_play, primary=True)
        self.btn_end = button("Show end", [0.082, 0.175, 0.062, 0.035], self._show_end)
        self.btn_new = button("New seed", [0.149, 0.175, 0.062, 0.035], self._new_seed)

        a = f.add_axes([0.015, 0.125, 0.2, 0.04], facecolor=self.BG)
        a.set_frame_on(False)
        self.check = CheckButtons(a, ["Theory: mean and +/-2 sigma band"], [s.show_band])
        self.check.on_clicked(lambda _l: self._set("show_band", not self.s.show_band))

        f.text(0.015, 0.105, "Tree of all paths", fontsize=8.5, color=self.MUTED, fontweight="bold")
        slider("tree_depth", "Tree depth", 0.07, 1, 12, s.tree_depth, 1)
        self.btn_h = button("Heads / 1st", [0.015, 0.025, 0.047, 0.035], lambda _e: self._manual(0))
        self.btn_t = button("Tails / 2nd", [0.066, 0.025, 0.047, 0.035], lambda _e: self._manual(1))
        self.btn_r = button("Random", [0.117, 0.025, 0.047, 0.035], lambda _e: self._manual(None))
        self.btn_c = button("Clear", [0.168, 0.025, 0.047, 0.035], self._clear_tree)

    # ---------------- callbacks ----------------
    def _set(self, name, value, restart=False):
        if name in ("steps", "walks", "seed", "tree_depth"):
            value = int(round(value))
        setattr(self.s, name, value)
        if name == "speed":
            return
        if name == "tree_depth":
            self.manual = self.manual[: self.s.tree_depth]
            self.draw_tree()
            self.fig.canvas.draw_idle()
            return
        if name == "show_band":
            self.draw_frame()
            return
        self.resimulate(restart=restart)

    def _toggle_play(self, _e=None):
        if self.playing:
            self.playing = False
            self.timer.stop()
        else:
            if self.t >= self.s.steps:
                self.t = 0
            self.playing = True
            self.timer.start()
        self.btn_play.label.set_text("Pause" if self.playing else "Play")
        self.fig.canvas.draw_idle()

    def _show_end(self, _e=None):
        if self.playing:
            self._toggle_play()
        self.t = self.s.steps
        self.draw_frame()

    def _new_seed(self, _e=None):
        self.sliders["seed"].set_val(int(np.random.randint(0, 10000)))

    def _tick(self):
        self.t = min(self.s.steps, self.t + max(1, int(round(self.s.speed * 0.03))))
        if self.t >= self.s.steps and self.playing:
            self._toggle_play()
        self.draw_frame()

    def _on_click(self, event):
        r = self.res
        if event.inaxes is not self.ax_paths or r is None or event.xdata is None:
            return
        t = int(self.t)
        if r.model.dim == 1:
            k = int(np.clip(round(event.xdata), 0, t))
            self.highlight = int(np.argmin(np.abs(r.pos[:, k] - event.ydata)))
        else:
            d = np.hypot(r.pos[:, t, 0] - event.xdata, r.pos[:, t, 1] - event.ydata)
            self.highlight = int(np.argmin(d))
        self.draw_frame()

    def _manual(self, k):
        r = self.res
        if r is None or not (r.model.discrete and r.model.dim == 1):
            return
        if len(self.manual) >= self.s.tree_depth:
            return
        if k is None:
            p = np.array([o.prob for o in r.outcomes])
            k = int(np.random.choice(len(p), p=p / p.sum()))
        if k >= len(r.outcomes) or r.outcomes[k].prob <= 0:
            return
        self.manual.append(k)
        self.draw_tree()
        self.fig.canvas.draw_idle()

    def _clear_tree(self, _e=None):
        self.manual = []
        self.draw_tree()
        self.fig.canvas.draw_idle()

    # ---------------- simulation ----------------
    def resimulate(self, restart=False):
        try:
            res = simulate(self.s)
            self.error_text.set_text("")
        except ValueError as err:
            self.error_text.set_text(f"Input error: {err}")
            self.fig.canvas.draw_idle()
            return
        at_end = self.res is None or self.t >= self.res.settings.steps
        self.res = res
        self.exact = ExactDistribution(res.outcomes) if (res.model.discrete and res.model.dim == 1) else None
        if self.highlight >= self.s.walks:
            self.highlight = 0
        if res.outcomes and any(k >= len(res.outcomes) for k in self.manual):
            self.manual = []
        self.t = self.s.steps if at_end else min(self.t, self.s.steps)
        self._setup_paths()
        self.draw_tree()
        if restart and self.interactive:
            self.t = self.s.steps
            if self.playing:
                self._toggle_play()
            self._toggle_play()
        self.draw_frame()

    def _setup_paths(self):
        r, ax = self.res, self.ax_paths
        ax.clear()
        n, M = self.s.steps, self.s.walks
        self.title.set_text(f"{r.model.name} random walk")
        self.subtitle.set_text(r.model.rule + "    (click the chart to highlight a walk)")
        alpha = float(np.clip(2.2 / math.sqrt(M), 0.05, 0.85))
        lw = 0.7 if M > 600 else 1.0 if M > 100 else 1.5
        if r.model.dim == 1:
            end = r.pos[:, -1]
            scale = np.abs(end).max() or 1
            colors = self.cmap(0.5 + 0.5 * end / scale)
            colors[:, 3] = alpha
            lo, hi = r.pos.min(), r.pos.max()
            sd = math.sqrt(r.var * n)
            lo, hi = min(lo, r.mu * n - 2.2 * sd, 0), max(hi, r.mu * n + 2.2 * sd, 0)
            span = max(r.pos.max() - r.pos.min(), 1)
            lo, hi = max(lo, r.pos.min() - 0.15 * span - 1), min(hi, r.pos.max() + 0.15 * span + 1)
            pad = 0.04 * (hi - lo)
            self.ylim = (lo - pad, hi + pad)
            ax.set_xlim(0, n)
            ax.set_ylim(*self.ylim)
            ax.axhline(0, color="#C9D0DB", lw=1, zorder=0)
            ax.set_xlabel("step")
            ax.set_ylabel("position")
            ts = np.linspace(0, n, 300)
            self.band = ax.fill_between(ts, r.mu * ts - 2 * np.sqrt(r.var * ts), r.mu * ts + 2 * np.sqrt(r.var * ts),
                                        color=self.BAND, lw=0.8, ls=":", edgecolor=self.MUTED, zorder=1)
            (self.mean_line,) = ax.plot([0, n], [0, r.mu * n], ls="--", color=self.INK, lw=1.4, zorder=3)
            self.tline = ax.axvline(0, color="#C9D0DB", lw=1, zorder=0)
        else:
            end = r.pos[:, -1]
            ang = np.arctan2(end[:, 1], end[:, 0])
            colors = self.cmap(0.5 + 0.5 * np.cos(ang - math.pi / 4))
            colors[:, 3] = alpha
            R = max(np.abs(r.pos).max(), 1) * 1.06
            ax.set_xlim(-R, R)
            ax.set_ylim(-R, R)
            ax.set_aspect("equal", adjustable="box")
            ax.axhline(0, color="#C9D0DB", lw=1, zorder=0)
            ax.axvline(0, color="#C9D0DB", lw=1, zorder=0)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            from matplotlib.patches import Circle
            self.circle = Circle((0, 0), 0, facecolor=self.BAND, edgecolor=self.INK, ls="--", lw=1.2, zorder=1)
            ax.add_patch(self.circle)
            self.dots = ax.scatter([], [], s=6, zorder=4)
            self.dot_colors = colors.copy()
            self.dot_colors[:, 3] = 0.9
        if r.model.dim == 1:
            ax.set_aspect("auto")
        ax.grid(True, color="#E4E8EF", lw=0.8)
        ax.set_axisbelow(True)
        self.lc = self.LineCollection([], colors=colors, linewidths=lw, zorder=2)
        ax.add_collection(self.lc)
        (self.hl_halo,) = ax.plot([], [], color="white", lw=4.5, zorder=5)
        (self.hl,) = ax.plot([], [], color=self.INK, lw=2, zorder=6)
        (self.hl_dot,) = ax.plot([], [], "o", color=self.INK, ms=5, zorder=7)

    # ---------------- drawing ----------------
    def draw_frame(self):
        r = self.res
        if r is None:
            return
        t = int(self.t)
        n = self.s.steps
        h = self.highlight
        if r.model.dim == 1:
            xs = np.broadcast_to(np.arange(t + 1), (self.s.walks, t + 1))
            self.lc.set_segments(np.stack([xs, r.pos[:, : t + 1]], axis=2))
            self.hl.set_data(np.arange(t + 1), r.pos[h, : t + 1])
            self.hl_halo.set_data(np.arange(t + 1), r.pos[h, : t + 1])
            self.hl_dot.set_data([t], [r.pos[h, t]])
            self.band.set_visible(self.s.show_band)
            self.mean_line.set_visible(self.s.show_band)
            self.tline.set_xdata([t, t])
            self.tline.set_visible(t < n)
        else:
            self.lc.set_segments(r.pos[:, : t + 1, :])
            self.hl.set_data(r.pos[h, : t + 1, 0], r.pos[h, : t + 1, 1])
            self.hl_halo.set_data(r.pos[h, : t + 1, 0], r.pos[h, : t + 1, 1])
            self.hl_dot.set_data([r.pos[h, t, 0]], [r.pos[h, t, 1]])
            self.dots.set_offsets(r.pos[:, t, :])
            self.dots.set_color(self.dot_colors)
            (mx, my), (vx, vy) = r.mu, r.var
            self.circle.set_center((mx * t, my * t))
            self.circle.set_radius(math.sqrt((vx + vy) * t))
            self.circle.set_visible(self.s.show_band)
        self.draw_hist(t)
        self.draw_stats(t)
        self.fig.canvas.draw_idle()

    def draw_hist(self, t):
        r, ax = self.res, self.ax_hist
        ax.clear()
        ax.grid(True, color="#E4E8EF", lw=0.8)
        ax.set_axisbelow(True)
        M = self.s.walks
        if r.model.dim == 1:
            vals = r.pos[:, t]
            ax.set_ylim(*self.ylim)
            scale = max(abs(self.ylim[0]), abs(self.ylim[1])) or 1
            ex = self.exact.at(t) if self.exact else None
            if ex is not None and ex[0].size <= 1100:
                support, theory = ex
                keys = np.round(support / self.exact.unit).astype(np.int64)
                emp_keys, counts = np.unique(np.round(vals / self.exact.unit).astype(np.int64), return_counts=True)
                emp = dict(zip(emp_keys.tolist(), counts.tolist()))
                freq = np.array([emp.get(k, 0) for k in keys.tolist()]) / M
                gap = np.diff(support).min() if support.size > 1 else 1.0
                ax.barh(support, freq, height=0.8 * gap, color=self.cmap(0.5 + 0.5 * support / scale), alpha=0.85)
                ax.plot(theory, support, "-o", color=self.INK, ms=2.5 if support.size < 60 else 0, lw=1.3)
                kind = "exact"
            else:
                bins = np.linspace(*self.ylim, int(np.clip(1.6 * math.sqrt(M), 12, 60)) + 1)
                counts, edges = np.histogram(vals, bins=bins)
                centers = 0.5 * (edges[1:] + edges[:-1])
                ax.barh(centers, counts / M, height=np.diff(edges) * 0.9,
                        color=self.cmap(0.5 + 0.5 * centers / scale), alpha=0.85)
                if t > 0 and r.var > 0:
                    sd = math.sqrt(r.var * t)
                    th = normal_cdf(edges[1:], r.mu * t, sd) - normal_cdf(edges[:-1], r.mu * t, sd)
                    ax.plot(th, centers, "-", color=self.INK, lw=1.3)
                kind = "normal approx."
            ax.set_title(f"Position at step {t}", fontsize=9, color=self.MUTED, loc="left")
            ax.set_xlabel(f"fraction   (line: theory, {kind})")
            ax.tick_params(labelleft=False)
        else:
            d = np.hypot(r.pos[:, t, 0], r.pos[:, t, 1])
            Rm = max(np.hypot(r.pos[..., 0], r.pos[..., 1]).max(), 1)
            edges = np.linspace(0, Rm, int(np.clip(1.4 * math.sqrt(M), 10, 40)) + 1)
            counts, _ = np.histogram(d, bins=edges)
            centers = 0.5 * (edges[1:] + edges[:-1])
            ax.barh(centers, counts / M, height=np.diff(edges) * 0.9, color="#8C95A6")
            (mx, my), (vx, vy) = r.mu, r.var
            if t > 0 and abs(mx) < 1e-9 and abs(my) < 1e-9 and abs(vx - vy) < 1e-9:
                s2 = vx * t
                F = 1 - np.exp(-edges**2 / (2 * s2))
                ax.plot(np.diff(F), centers, color=self.INK, lw=1.3)
            ax.set_ylim(0, Rm)
            ax.set_title(f"Distance from origin, step {t}", fontsize=9, color=self.MUTED, loc="left")
            ax.set_xlabel("fraction   (line: Rayleigh theory)")

    def draw_stats(self, t):
        r = self.res
        M = self.s.walks
        returned = np.mean(r.first_return <= t)
        if r.model.dim == 1:
            v = r.pos[:, t]
            lines = [
                f"Step            {t} / {self.s.steps}",
                f"                 sim     theory",
                f"Mean       {v.mean():9.2f}  {r.mu * t:9.2f}",
                f"Std dev    {v.std():9.2f}  {math.sqrt(r.var * t):9.2f}",
                f"Above 0    {np.mean(v > 0):9.1%}",
                f"Returned 0 {returned:9.1%}",
                f"Min / max  {v.min():g} / {v.max():g}",
            ]
            if self.exact and self.exact.ok:
                sup, pr = self.exact.at(t)
                lines[4] += f"  {pr[sup > 1e-9].sum():9.1%}"
        else:
            d = np.hypot(r.pos[:, t, 0], r.pos[:, t, 1])
            (mx, my), (vx, vy) = r.mu, r.var
            rms_th = math.sqrt((vx + vy) * t + (mx**2 + my**2) * t * t)
            lines = [
                f"Step            {t} / {self.s.steps}",
                f"                 sim     theory",
                f"Mean dist  {d.mean():9.2f}",
                f"RMS dist   {math.sqrt((d**2).mean()):9.2f}  {rms_th:9.2f}",
                f"Back at 0  {returned:9.1%}",
            ]
        self.stats_text.set_text("\n".join(lines))
        if r.idx is not None:
            seq = "".join(r.outcomes[k].label if len(r.outcomes[k].label) == 1 else "." for k in r.idx[self.highlight, :t][:110])
            if r.model.name == "Custom steps":
                seq = " ".join(r.outcomes[k].label for k in r.idx[self.highlight, :t][:40])
            more = " ..." if t > 110 else ""
            self.seq_text.set_text(f"Highlighted walk #{self.highlight + 1}:  {seq}{more}")
        else:
            self.seq_text.set_text(f"Highlighted walk #{self.highlight + 1}")

    def draw_tree(self):
        r, ax = self.res, self.ax_tree
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        if r is None or not (r.model.discrete and r.model.dim == 1):
            ax.text(0.5, 0.5, "The tree of all paths is drawn for discrete 1D models\n(coin, die, custom steps).",
                    ha="center", va="center", color=self.MUTED, transform=ax.transAxes)
            return
        outs = r.outcomes
        depth = self.s.tree_depth
        levels = path_tree(outs, depth)
        while depth > 1 and len(levels[depth]) > 80:
            depth -= 1
        levels = levels[: depth + 1]
        self.manual = self.manual[:depth]
        allv = [v for lv in levels for v in lv]
        vmin, vmax = min(allv + [0]), max(allv + [0])
        if vmax - vmin < 1e-9:
            vmin, vmax = vmin - 1, vmax + 1
        scale = max(abs(vmin), abs(vmax)) or 1
        # edges
        segs = []
        for t, lv in enumerate(levels[:-1]):
            for v in lv:
                for o in outs:
                    if o.prob > 0:
                        segs.append([(t, v), (t + 1, v + o.step)])
        ax.add_collection(self.LineCollection(segs, colors="#8C95A6", alpha=0.35, linewidths=0.8, zorder=1))
        # manual path
        x = 0.0
        path = [(0, 0.0)]
        prob = 1.0
        for i, k in enumerate(self.manual):
            x += outs[k].step
            prob *= outs[k].prob
            path.append((i + 1, x))
        if len(path) > 1:
            px, py = zip(*path)
            ax.plot(px, py, color="white", lw=6, zorder=2)
            ax.plot(px, py, color=self.INK, lw=2.6, zorder=3)
        # nodes
        big = depth <= 8 and max(len(lv) for lv in levels) <= 15
        size = 330 if big else max(20, 2400 / max(len(lv) for lv in levels))
        for t, lv in enumerate(levels):
            pmax = max(p for _, p in lv.values())
            vs = np.array(list(lv.keys()))
            cs = np.array([c for c, _ in lv.values()])
            ps = np.array([p for _, p in lv.values()])
            col = self.cmap(0.5 + 0.5 * vs / scale)
            col[:, 3] = np.clip(0.2 + 0.8 * ps / pmax, 0, 1)
            ax.scatter(np.full(vs.size, t), vs, s=size, c=col, edgecolors=self.cmap(0.5 + 0.5 * vs / scale), lw=1, zorder=4)
            if big:
                for v, c, p in zip(vs, cs, ps):
                    ax.text(t, v, f"{c:,}", ha="center", va="center", fontsize=7.5, fontweight="bold",
                            color="white" if p / pmax > 0.6 else self.INK, zorder=5)
            if t == depth and (vmax - vmin) / max(len(lv), 1) > 0:
                if len(lv) <= 25:
                    for v, p in zip(vs, ps):
                        ax.text(t + 0.18, v, f"{p:.1%}", va="center", fontsize=7.5, color=self.MUTED)
        for t, v in path:
            ax.scatter([t], [v], s=size, facecolors="none", edgecolors=self.INK, lw=2.2, zorder=6)
        ax.set_xlim(-0.4, depth + 0.75)
        pad = 0.08 * (vmax - vmin)
        ax.set_ylim(vmin - pad, vmax + pad)
        ax.set_xticks(range(depth + 1))
        ax.set_xlabel("flip number")
        ax.axhline(0, color="#C9D0DB", lw=1, ls=":", zorder=0)
        n_paths = sum(1 for o in outs if o.prob > 0) ** depth
        msg = (f"Tree of all paths:  {n_paths:,} possible paths in {depth} flips, "
               f"but only {len(levels[-1])} different end points.   Numbers = paths reaching each point.")
        ax.set_title(msg, fontsize=9, color=self.MUTED, loc="left")
        if self.manual:
            end = levels[len(self.manual)].get(round(x, 9), (0, 0))
            ax.text(0.01, 0.97,
                    f"Your path: {' '.join(outs[k].label for k in self.manual)}   position {x:g}\n"
                    f"P(this exact path) = {prob:.4g}    paths ending here = {end[0]:,}    P(end here) = {end[1]:.1%}",
                    transform=ax.transAxes, va="top", fontsize=8.5, color=self.INK,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#D3D9E3"))
        else:
            ax.text(0.01, 0.97, "Use the buttons on the left to flip the coin and walk through the tree.",
                    transform=ax.transAxes, va="top", fontsize=8.5, color=self.MUTED)

    def show(self):
        if self.interactive:
            self.t = 0
            self._toggle_play()
        self.plt.show()


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> tuple[Settings, str | None]:
    s = Settings()
    ap = argparse.ArgumentParser(description="Interactive random walk simulator")
    ap.add_argument("--model", choices=list(MODELS), default=s.model)
    ap.add_argument("--steps", type=int, default=s.steps)
    ap.add_argument("--walks", type=int, default=s.walks)
    ap.add_argument("--seed", type=int, default=s.seed)
    ap.add_argument("--p", type=float, default=s.p, help="coin: probability of heads")
    ap.add_argument("--up", type=float, default=s.up, help="coin: step for heads")
    ap.add_argument("--down", type=float, default=s.down, help="coin: step for tails")
    ap.add_argument("--die-faces", default=s.die_faces)
    ap.add_argument("--custom", default=s.custom, help='e.g. "2:1, -1:2"')
    ap.add_argument("--mu", type=float, default=s.mu)
    ap.add_argument("--sigma", type=float, default=s.sigma)
    ap.add_argument("--weights2d", default=s.weights2d, help="right,left,up,down")
    ap.add_argument("--tree-depth", type=int, default=s.tree_depth)
    ap.add_argument("--save", metavar="FILE.png", help="save a picture of the final state and exit (no window)")
    ap.add_argument("--selftest", type=float, metavar="SECONDS", default=0,
                    help="open the window, close it after SECONDS and write RandomWalk-selftest.txt")
    a = ap.parse_args(argv)
    for k in vars(a):
        if hasattr(s, k):
            setattr(s, k, getattr(a, k))
    s.steps = max(1, s.steps)
    s.walks = max(1, s.walks)
    return s, a.save, a.selftest


def _log_dir():
    """Folder next to the exe (frozen) or the script, for the self-test file and error log."""
    import os
    base = sys.executable if getattr(sys, "frozen", False) else __file__
    return os.path.dirname(os.path.abspath(base))


def _say(msg: str):
    # A windowed exe has no console: sys.stdout is None there.
    if sys.stdout is not None:
        print(msg)


def main(argv=None):
    settings, save, selftest = parse_args(argv)
    import matplotlib
    if save:
        matplotlib.use("Agg")
    else:
        # Name the GUI backend explicitly. PyInstaller bundles only the backends it sees in
        # matplotlib.use() calls, so without this the exe has no window backend at all.
        matplotlib.use("TkAgg")
    app = App(settings, interactive=not save)
    if save:
        app.manual = [0, 1, 0][: settings.tree_depth]
        app.draw_tree()
        app.fig.savefig(save, dpi=110, facecolor=app.BG)
        _say(f"Saved {save}")
        return 0
    if selftest:
        # Open the real window, let it animate for a moment, then close it and leave a note.
        closer = app.fig.canvas.new_timer(interval=int(selftest * 1000))
        closer.single_shot = True
        closer.add_callback(lambda: app.plt.close(app.fig))
        closer.start()
    app.show()
    if selftest:
        import os
        backend = matplotlib.get_backend()
        with open(os.path.join(_log_dir(), "RandomWalk-selftest.txt"), "w") as fh:
            fh.write(f"ok backend={backend} t={app.t}\n")
    return 0


def _report_crash():
    """Show a crash instead of exiting silently, and save the traceback next to the program."""
    import os
    import traceback
    text = traceback.format_exc()
    path = os.path.join(_log_dir(), "RandomWalk-error.log")
    try:
        with open(path, "w") as fh:
            fh.write(text)
    except OSError:
        path = "(could not write log file)"
    _say(text)
    try:
        import tkinter
        from tkinter import messagebox
        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror("Random Walk Simulator", f"The program hit an error:\n\n{text[-1500:]}\n\nSaved to {path}")
        root.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        _report_crash()
        sys.exit(1)

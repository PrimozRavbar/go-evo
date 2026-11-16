"""Is evolution happening? Tools to detect a selection signal.

1. ``bench`` curves: mean win rate of the population vs a *fixed* panel (random genomes +
   uniform-random mover). Absolute yardstick that does not drift with the population.
2. Survivor advantage: in generation g+1, do last generation's winners (survivors) beat the
   newborn children more than 50% of the time? Under real selection + heritable strength they
   should; in the random-selection control they should not.
3. Head-to-head vs ancestors: final population vs generation-0 (and intermediate) checkpoints,
   every pairing, both colours.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .evolve import load_history
from .match import GameConfig
from .parallel import GameRunner
from .player import load_population


def bench_curve(hist):
    g = [h["gen"] for h in hist if "bench_mean" in h]
    m = [h["bench_mean"] for h in hist if "bench_mean" in h]
    return np.array(g), np.array(m)


def survivor_advantage(hist):
    """Per generation: (mean win fraction of survivors, mean win fraction of newborns)."""
    out = []
    for h in hist[1:]:
        g = h["gen"]
        surv, new = [], []
        prev = set(hist[g - 1]["ids"])
        for gid, w, l in zip(h["ids"], h["wins"], h["losses"]):
            frac = w / max(w + l, 1)
            (surv if gid in prev else new).append(frac)
        if surv and new:
            out.append((g, float(np.mean(surv)), float(np.mean(new))))
    return np.array(out)


def head_to_head(pop_a, pop_b, net_cfg, game_cfg: GameConfig, workers=None, seed=0, cfg_b=None):
    """Win rate of population A vs population B (all pairs, both colours).

    ``cfg_b``: network config of B when it differs from A's (e.g. conv vs MLP).
    """
    a = [("genome", f"A{n}", g) for n, g in zip(range(len(pop_a)), pop_a)]
    b = [("genome", f"B{n}", g) + ((cfg_b,) if cfg_b else ()) for n, g in zip(range(len(pop_b)), pop_b)]
    pairs, signs = [], []
    for x in a:
        for y in b:
            pairs += [(x, y), (y, x)]
            signs += [1, -1]
    with GameRunner(workers) as r:
        recs = r.play(pairs, net_cfg, game_cfg, seed=seed)
    v = [1.0 if rec.winner == s else 0.5 if rec.winner == 0 else 0.0 for rec, s in zip(recs, signs)]
    return float(np.mean(v)), len(v)


def ancestor_matrix(run_dir, gens, workers=None):
    """Final population vs each listed checkpoint generation."""
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    final = sorted(f for f in os.listdir(run_dir) if f.startswith("gen") and f.endswith(".pt"))[-1]
    fin, net_cfg, _, _ = load_population(os.path.join(run_dir, final))
    game_cfg = GameConfig(**cfg["game"])
    res = {}
    for g in gens:
        p = os.path.join(run_dir, f"gen{g:04d}.pt")
        if os.path.exists(p) and not p.endswith(final):
            anc, _, _, _ = load_population(p)
            res[g] = head_to_head(fin, anc, net_cfg, game_cfg, workers)
    return final, res


def plot_runs(run_dirs: dict[str, str], path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    colors = {"sel": "#1f6fd1", "ctl": "#9a9a9a", "ko_sel": "#d1561f", "ko_ctl": "#e8b28f"}
    for label, d in run_dirs.items():
        hist = load_history(d)
        kind = ("ko_" if label.startswith("ko_") else "") + ("sel" if "sel" in label else "ctl")
        c = colors[kind]
        ls = "-" if label.endswith("0") else "--"
        g, m = bench_curve(hist)
        axs[0].plot(g, m, ls, color=c, label=label)
        sa = survivor_advantage(hist)
        if len(sa):
            diff = sa[:, 1] - sa[:, 2]
            k = 10
            sm = np.convolve(diff, np.ones(k) / k, mode="valid")
            axs[1].plot(sa[k - 1:, 0], sm, ls, color=c, label=label)
        axs[2].plot([h["gen"] for h in hist], [h["mean_moves"] for h in hist], ls, color=c, alpha=.7, label=label)
    axs[0].axhline(0.5, color="k", lw=.5)
    axs[0].set_title("win rate vs fixed panel (random genomes + random mover)")
    axs[0].set_xlabel("generation")
    axs[1].axhline(0, color="k", lw=.5)
    axs[1].set_title("survivor − newborn win fraction (10-gen moving avg)")
    axs[1].set_xlabel("generation")
    axs[2].set_title("mean game length (moves)")
    axs[2].set_xlabel("generation")
    for a in axs:
        a.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_progress(run_dirs: dict[str, str], path: str, smooth: int = 5):
    """Phase 2c plot: parents' win rate vs the random mover, qualifier vs hall of fame, game length."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def sm(x, y, k):
        x, y = np.asarray(x), np.asarray(y, float)
        k = max(1, min(k, len(y)))
        return x[k - 1:], np.convolve(y, np.ones(k) / k, mode="valid")

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    for label, d in run_dirs.items():
        hist = load_history(d)
        c = "#d1561f" if "sel" in label else "#9a9a9a"
        g = np.array([h["gen"] for h in hist if "bench_by_opp" in h])
        rm = [h["bench_by_opp"].get("random_mover", np.nan) for h in hist if "bench_by_opp" in h]
        axs[0].plot(*sm(g, rm, smooth), color=c, label=label)
        q = [(h["gen"], h["qualifier_mean"]) for h in hist if "qualifier_mean" in h]
        if q:
            qg, qv = zip(*q)
            axs[1].plot(*sm(qg, qv, 25), color=c, label=label)
        mv = [h["mean_moves"] for h in hist]
        axs[2].plot(*sm([h["gen"] for h in hist], mv, 25), color=c, label=label)
    axs[0].axhline(0.5, color="k", lw=.6)
    axs[0].set_title(f"parents' win rate vs random mover ({smooth}-point avg)")
    axs[1].axhline(0.5, color="k", lw=.6)
    axs[1].set_title("qualifier win rate vs past champions (25-gen avg)")
    axs[2].set_title("mean game length (25-gen avg)")
    for a in axs:
        a.set_xlabel("generation")
        a.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)

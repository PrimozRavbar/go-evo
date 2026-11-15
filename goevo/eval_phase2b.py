"""Head-to-head evaluation of the phase 2b runs (writes runs/report/phase2b_eval.txt).

Note: the phase-2a checkpoints (runs/sel_s*, runs/ctl_s*) were deleted afterwards to keep the
repo small; comparisons that need them are skipped when they are missing.
"""
import os
import json

import numpy as np

from .analysis import head_to_head
from .evolve import EvoConfig, initial_population
from .match import GameConfig
from .parallel import GameRunner
from .player import RandomPlayer, load_population
from .parallel import spec_of


def vs_random_mover(genomes, net, gc, games_each=50, workers=4, seed=0):
    rm = spec_of(RandomPlayer("rm"))
    pairs, signs = [], []
    for i, g in enumerate(genomes):
        s = ("genome", f"G{i}", g)
        for k in range(games_each):
            pairs.append((s, rm) if k % 2 == 0 else (rm, s))
            signs.append(1 if k % 2 == 0 else -1)
    with GameRunner(workers) as r:
        recs = r.play(pairs, net, gc, seed=seed)
    return float(np.mean([rec.winner == s for rec, s in zip(recs, signs)])), len(recs)


def main():
    L = lambda r: load_population(f"runs/{r}/gen0300.pt")[0] if os.path.exists(f"runs/{r}/gen0300.pt") else []
    net = load_population("runs/ko_sel_s0/gen0300.pt")[1]
    gc = GameConfig(board_size=19, backend="native")
    cfg = json.load(open("runs/ko_sel_s0/config.json"))["evo"]
    gen0 = [g.genome for g in initial_population(EvoConfig(pop_size=cfg["pop_size"], seed=cfg["seed"]), net)[0]]
    pops = {"ko_sel": L("ko_sel_s0"), "ko_ctl": L("ko_ctl_s0"),
            "2a_sel": L("sel_s0") + L("sel_s1"), "2a_ctl": L("ctl_s0") + L("ctl_s1"), "gen0": gen0}
    out = []
    def log(s):
        print(s, flush=True); out.append(s)
    log("win rate of the first set; all pairings x both colours (deterministic play)")
    for a, b in [("ko_sel", "ko_ctl"), ("ko_sel", "gen0"), ("ko_ctl", "gen0"), ("ko_sel", "2a_sel"),
                 ("ko_ctl", "2a_sel"), ("ko_sel", "2a_ctl"), ("2a_sel", "gen0")]:
        if not len(pops[a]) or not len(pops[b]):
            continue
        wr, n = head_to_head(pops[a], pops[b], net, gc, workers=4)
        se = (wr * (1 - wr) / n) ** 0.5
        log(f"{a:>7} vs {b:<7} {wr:.3f} ± {se:.3f}  (n={n})")
    log("\nwin rate vs uniform-random mover (stochastic opponent, 50 games per genome)")
    for k in ["ko_sel", "ko_ctl", "2a_sel", "2a_ctl", "gen0"]:
        g = pops[k] if k != "gen0" else gen0[:8]
        if not len(g):
            continue
        wr, n = vs_random_mover(g, net, gc)
        log(f"{k:>7}: {wr:.3f} ± {(wr * (1 - wr) / n) ** 0.5:.3f}  (n={n})")
    open("runs/report/phase2b_eval.txt", "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

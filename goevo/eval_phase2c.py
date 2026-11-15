"""Head-to-head evaluation of the phase 2c runs (writes runs/report/phase2c_eval.txt).

All games use the current rules (fast engine, pass only when forced).
"""
import json
import os

from .analysis import head_to_head
from .eval_phase2b import vs_random_mover
from .evolve import EvoConfig, initial_population
from .match import GameConfig
from .player import load_population


def main(gen=2000, workers=4):
    L = lambda p: load_population(p)[0] if os.path.exists(p) else []
    net = load_population(f"runs/c_sel_s0/gen{gen:04d}.pt")[1]
    gc = GameConfig(board_size=19)
    cfg = json.load(open("runs/c_sel_s0/config.json"))["evo"]
    gen0 = [g.genome for g in initial_population(EvoConfig(pop_size=cfg["pop_size"], seed=cfg["seed"]), net)[0]]
    pops = {"c_sel": L(f"runs/c_sel_s0/gen{gen:04d}.pt"), "c_ctl": L(f"runs/c_ctl_s0/gen{gen:04d}.pt"),
            "ko_sel": L("runs/ko_sel_s0/gen0300.pt"), "gen0": gen0[:16]}
    out = []
    def log(s):
        print(s, flush=True); out.append(s)
    log("win rate of the first set; all pairings x both colours (deterministic play)")
    for a, b in [("c_sel", "c_ctl"), ("c_sel", "gen0"), ("c_ctl", "gen0"), ("c_sel", "ko_sel")]:
        if len(pops[a]) and len(pops[b]):
            wr, n = head_to_head(pops[a], pops[b], net, gc, workers=workers)
            log(f"{a:>6} vs {b:<6} {wr:.3f} ± {(wr * (1 - wr) / n) ** 0.5:.3f}  (n={n})")
    log("\nwin rate vs uniform-random mover (100 games per genome)")
    for k in ["c_sel", "c_ctl", "ko_sel", "gen0"]:
        g = pops[k] if k != "gen0" else gen0[:8]
        if len(g):
            wr, n = vs_random_mover(g, net, gc, games_each=100, workers=workers)
            log(f"{k:>6}: {wr:.3f} ± {(wr * (1 - wr) / n) ** 0.5:.3f}  (n={n})")
    open("runs/report/phase2c_eval.txt", "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

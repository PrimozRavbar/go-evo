"""Evaluation of the phase 2d (conv) runs; writes runs/report/phase2d_eval.txt."""
import json
import sys

from .analysis import head_to_head
from .eval_phase2b import vs_random_mover
from .evolve import EvoConfig, initial_population
from .match import GameConfig
from .player import load_population


def main(gen=1000, root="runs", workers=4):
    sel, net, *_ = load_population(f"{root}/d_sel_s0/gen{gen:04d}.pt")
    ctl = load_population(f"{root}/d_ctl_s0/gen{gen:04d}.pt")[0]
    mlp, mlp_net, *_ = load_population("runs/c_sel_s0/gen2000.pt")
    cfg = json.load(open(f"{root}/d_sel_s0/config.json"))["evo"]
    gen0 = [g.genome for g in initial_population(EvoConfig(pop_size=cfg["pop_size"], seed=cfg["seed"]), net)[0]]
    gc = GameConfig(board_size=19)
    out = []
    def log(s):
        print(s, flush=True); out.append(s)
    log(f"phase 2d, generation {gen}. Win rate of the first set; all pairings x both colours")
    for name, b, cb in [("d_ctl", ctl, None), ("gen0", gen0[:16], None), ("c_sel (MLP)", mlp, mlp_net)]:
        wr, n = head_to_head(sel, b, net, gc, workers=workers, cfg_b=cb)
        log(f"d_sel vs {name:<12} {wr:.3f} ± {(wr * (1 - wr) / n) ** 0.5:.3f}  (n={n})")
    wr, n = head_to_head(ctl, gen0[:16], net, gc, workers=workers)
    log(f"d_ctl vs {'gen0':<12} {wr:.3f} ± {(wr * (1 - wr) / n) ** 0.5:.3f}  (n={n})")
    log("\nwin rate vs uniform-random mover (100 games per genome)")
    for name, g in [("d_sel", sel), ("d_ctl", ctl), ("gen0", gen0[:8])]:
        wr, n = vs_random_mover(g, net, gc, games_each=100, workers=workers)
        log(f"{name:>6}: {wr:.3f} ± {(wr * (1 - wr) / n) ** 0.5:.3f}  (n={n})")
    open("runs/report/phase2d_eval.txt", "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:2]))

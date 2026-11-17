"""Phase 4 evaluation: final parents vs own ancestors, the phase 2d conv champions,
the phase 3 low-rank champions and the random mover. Writes runs/report/phase4_eval.txt."""
import os
import sys

from .analysis import head_to_head
from .eval_phase2b import vs_random_mover
from .match import GameConfig
from .player import load_population


def main(run="work/p4_s0", final=4000, ancestors=(0, 1000, 2000), workers=4):
    fin, net, *_ = load_population(f"{run}/gen{final:04d}.pt")
    gc = GameConfig(board_size=19)
    out = []
    def log(s):
        print(s, flush=True); out.append(s)
    def h2h(label, pop, cfg_b=None):
        wr, n = head_to_head(fin, pop, net, gc, workers=workers, cfg_b=cfg_b)
        log(f"vs {label:<22} {wr:.3f} ± {(wr * (1 - wr) / n) ** 0.5:.3f}  (n={n})")
    log(f"phase 4 ({run}), final generation {final}. Win rate of the final parents (all pairings x both colours)")
    for a in ancestors:
        h2h(f"own gen {a}", load_population(f"{run}/gen{a:04d}.pt")[0])
    for label, path in [("2d conv champions", "runs/d_sel_s0/gen1000.pt"),
                        ("3 low-rank champions", "runs/p3_r32_s0/gen10000.pt")]:
        if os.path.exists(path):
            pop, cfg_b, *_ = load_population(path)
            h2h(label, pop, cfg_b)
    wr, n = vs_random_mover(fin, net, gc, games_each=100, workers=workers)
    log(f"vs {'random mover':<22} {wr:.3f} ± {(wr * (1 - wr) / n) ** 0.5:.3f}  (n={n})")
    os.makedirs("runs/report", exist_ok=True)
    open("runs/report/phase4_eval.txt", "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:2])

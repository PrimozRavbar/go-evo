"""Short phase 3 evaluation: final parents vs own ancestors, vs the phase 2d conv
champions and vs the random mover. Writes runs/report/phase3_eval.txt."""
import os
import sys

from .analysis import head_to_head
from .eval_phase2b import vs_random_mover
from .match import GameConfig
from .player import load_population


def main(run="work/p3_r32_s0", final=10000, ancestors=(1000, 5000), workers=4):
    fin, net, *_ = load_population(f"{run}/gen{final:04d}.pt")
    gc = GameConfig(board_size=19)
    out = []
    def log(s):
        print(s, flush=True); out.append(s)
    log(f"phase 3 ({run}), final generation {final}. Win rate of the final parents (all pairings x both colours)")
    for a in ancestors:
        anc = load_population(f"{run}/gen{a:04d}.pt")[0]
        wr, n = head_to_head(fin, anc, net, gc, workers=workers)
        log(f"vs own gen {a:<5}       {wr:.3f} ± {(wr * (1 - wr) / n) ** 0.5:.3f}  (n={n})")
    if os.path.exists("runs/d_sel_s0/gen1000.pt"):
        conv, conv_net, *_ = load_population("runs/d_sel_s0/gen1000.pt")
        wr, n = head_to_head(fin, conv, net, gc, workers=workers, cfg_b=conv_net)
        log(f"vs 2d conv champions  {wr:.3f} ± {(wr * (1 - wr) / n) ** 0.5:.3f}  (n={n})")
    wr, n = vs_random_mover(fin, net, gc, games_each=100, workers=workers)
    log(f"vs random mover       {wr:.3f} ± {(wr * (1 - wr) / n) ** 0.5:.3f}  (n={n})")
    open("runs/report/phase3_eval.txt", "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:2])

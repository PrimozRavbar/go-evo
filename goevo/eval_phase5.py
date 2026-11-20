"""Phase 5 evaluation: final PPO policies (last 4) vs the phase 4 evolved parents, the phase 2d
conv / phase 3 low-rank champions, PPO's own hourly checkpoints and the random mover.
Same masked-argmax players as the evolved runs. Writes runs/report/phase5_eval.txt."""
import os
import sys

from .analysis import head_to_head
from .eval_phase2b import vs_random_mover
from .match import GameConfig
from .player import load_population


def main(run="work/p5_s0", ancestors=(0, 2, 4), workers=4, out_file="runs/report/phase5_eval.txt",
         rm_games=0, evo=False, gc=None, others=None):
    fin, net, _, extra = load_population(f"{run}/final.pt")
    gc = gc or GameConfig(board_size=19)
    out = []

    def log(s):
        print(s, flush=True)
        out.append(s)

    def h2h(label, pop, cfg_b=None):
        wr, n = head_to_head(fin, pop, net, gc, workers=workers, cfg_b=cfg_b)
        log(f"vs {label:<24} {wr:.3f} ± {(wr * (1 - wr) / n) ** 0.5:.3f}  (n={n})")
    log(f"phase 5 PPO ({run}): {extra.get('it')} iterations, {extra.get('games')} games, "
        f"{extra.get('hours', 0):.2f} h. Win rate of the last 4 policies (all pairings x both colours)")
    for h in ancestors:
        p = f"{run}/h{h:02d}.pt"
        if os.path.exists(p):
            h2h(f"own hour {h}", load_population(p)[0])
    others = others if others is not None else [] if os.path.abspath(run) == os.path.abspath("runs/p5_s0") or run.endswith("p5_s0") else \
        [("phase 5 PPO (small net)", "runs/p5_s0/final.pt")]
    evo_refs = [("phase 4 evolved parents", "runs/p4_s0/gen4000.pt"),
                ("2d conv champions", "runs/d_sel_s0/gen1000.pt"),
                ("3 low-rank champions", "runs/p3_r32_s0/gen10000.pt")] if evo else []
    for label, path in others + evo_refs:
        if os.path.exists(path):
            pop, cfg_b, *_ = load_population(path)
            h2h(label, pop, cfg_b)
    if rm_games:  # the random mover is no longer informative: off by default
        wr, n = vs_random_mover(fin, net, gc, games_each=rm_games, workers=workers)
        log(f"vs {'random mover':<24} {wr:.3f} ± {(wr * (1 - wr) / n) ** 0.5:.3f}  (n={n})")
    os.makedirs("runs/report", exist_ok=True)
    open(out_file, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:2])  # e.g. python -c "from goevo.eval_phase5 import main; main('work/p5b_s0', (0, 2, 4, 6), out_file='runs/report/phase5b_eval.txt')"

"""Phase 2 CLI: evolve MLP Go players (19x19 by default).

Phase 2b defaults: 32 genomes, knockout bracket + qualifier vs random mover / hall of fame,
4 parents, sigma 0.03, only the final parents are checkpointed.

    python -m goevo.phase2 --generations 300 --seed 0 --out runs/ko_sel_s0
    python -m goevo.phase2 --generations 300 --seed 0 --selection random --out runs/ko_ctl_s0  # control

Phase 2a settings (round robin, winning-record rule, 10 genomes, sigma 0.1):
    python -m goevo.phase2 --tournament roundrobin --pop 10 --parents 0 --sigma 0.1 --checkpoint-every 50 --all-checkpoint
"""
from __future__ import annotations

import argparse

from .engine import pettingzoo_available
from .evolve import EvoConfig, run_evolution
from .match import GameConfig
from .player import NetConfig


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=19)
    ap.add_argument("--komi", type=float, default=7.5)
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--backend", default="native", choices=["auto", "pettingzoo", "native"],
                    help="native is faster and verified identical to pettingzoo")
    ap.add_argument("--max-moves", type=int, default=None, help="default 3*N*N (1083 on 19x19)")
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--tournament", default="knockout", choices=["knockout", "roundrobin"])
    ap.add_argument("--games", type=int, default=5, help="games per genome (roundrobin)")
    ap.add_argument("--parents", type=int, default=4, help="0 = 'winning record' rule (roundrobin)")
    ap.add_argument("--generations", type=int, default=300)
    ap.add_argument("--sigma", type=float, default=0.03)
    ap.add_argument("--no-elitism", action="store_true")
    ap.add_argument("--selection", default="fitness", choices=["fitness", "winners", "random"])
    ap.add_argument("--external-top", type=int, default=8, help="top-k play the qualifier (0 = off)")
    ap.add_argument("--external-weight", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench-every", type=int, default=10)
    ap.add_argument("--checkpoint-every", type=int, default=0, help="0 = only the final generation")
    ap.add_argument("--all-checkpoint", action="store_true", help="save whole population, not just parents")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default="runs/evo")
    a = ap.parse_args(argv)

    backend = a.backend
    if backend == "auto":
        backend = "pettingzoo" if pettingzoo_available() else "native"
    net = NetConfig(board_size=a.size, num_layers=a.layers, hidden=a.hidden)
    game = GameConfig(board_size=a.size, komi=a.komi, backend=backend, max_moves=a.max_moves)
    evo = EvoConfig(pop_size=a.pop, tournament=a.tournament, games_per_player=a.games,
                    n_parents=a.parents, generations=a.generations, sigma=a.sigma,
                    elitism=not a.no_elitism, selection=a.selection, external_top=a.external_top,
                    external_weight=a.external_weight, seed=a.seed, bench_every=a.bench_every,
                    checkpoint_every=a.checkpoint_every, checkpoint_parents_only=not a.all_checkpoint,
                    out_dir=a.out, workers=a.workers)
    print(f"{a.size}x{a.size}, backend={backend}, max_moves={game.resolved().max_moves}, "
          f"genome={net.genome_size:,} params, {evo.tournament}, pop={evo.pop_size}, "
          f"parents={evo.n_parents}, sigma={evo.sigma}, selection={evo.selection}")
    return run_evolution(evo, net, game)


if __name__ == "__main__":
    main()

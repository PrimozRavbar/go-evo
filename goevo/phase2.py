"""Phase 2 CLI: evolve MLP Go players (19x19 by default).

    python -m goevo.phase2 --generations 300 --selection winners --seed 0 --out runs/sel_s0
    python -m goevo.phase2 --generations 300 --selection random  --seed 0 --out runs/ctl_s0   # control
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
                    help="native is ~1.5x faster and verified identical to pettingzoo")
    ap.add_argument("--max-moves", type=int, default=None, help="default 3*N*N (1083 on 19x19)")
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--games", type=int, default=5)
    ap.add_argument("--generations", type=int, default=200)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--no-elitism", action="store_true")
    ap.add_argument("--selection", default="winners", choices=["winners", "random"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench-every", type=int, default=10)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default="runs/evo")
    a = ap.parse_args(argv)

    backend = a.backend
    if backend == "auto":
        backend = "pettingzoo" if pettingzoo_available() else "native"
    net = NetConfig(board_size=a.size, num_layers=a.layers, hidden=a.hidden)
    game = GameConfig(board_size=a.size, komi=a.komi, backend=backend, max_moves=a.max_moves)
    evo = EvoConfig(pop_size=a.pop, games_per_player=a.games, generations=a.generations, sigma=a.sigma,
                    elitism=not a.no_elitism, selection=a.selection, seed=a.seed,
                    bench_every=a.bench_every, checkpoint_every=a.checkpoint_every,
                    out_dir=a.out, workers=a.workers)
    print(f"{a.size}x{a.size}, backend={backend}, max_moves={game.resolved().max_moves}, "
          f"genome={net.genome_size:,} params")
    return run_evolution(evo, net, game)


if __name__ == "__main__":
    main()

"""Phase 1: 10 random genomes, each plays 5 headless games.

    python -m goevo.phase1 --pop 10 --games 5 --backend auto [--watch 0] [--out results]
"""
from __future__ import annotations

import argparse
import json
import os

from .engine import pettingzoo_available
from .match import GameConfig
from .player import NetConfig, save_population
from .tournament import random_population, run_tournament
from .viz import ascii_replay


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--games", type=int, default=5, help="games per player")
    ap.add_argument("--size", type=int, default=9)
    ap.add_argument("--komi", type=float, default=7.5)
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--backend", default="auto", choices=["auto", "pettingzoo", "native"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--watch", type=int, default=None, help="ASCII-replay game #k afterwards")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)

    backend = a.backend
    if backend == "auto":
        backend = "pettingzoo" if pettingzoo_available() else "native"
    print(f"Go backend: {backend}")

    net_cfg = NetConfig(board_size=a.size, num_layers=a.layers, hidden=a.hidden)
    game_cfg = GameConfig(board_size=a.size, komi=a.komi, backend=backend, temperature=a.temperature)
    players = random_population(a.pop, net_cfg, seed=a.seed)
    print(f"{a.pop} players, {net_cfg.num_layers}-layer MLP, {net_cfg.genome_size:,} params each")

    res = run_tournament(players, a.games, game_cfg, seed=a.seed)
    print()
    print(res.table())
    print(f"\n{len(res.records)} games in {res.seconds:.1f}s")

    os.makedirs(a.out, exist_ok=True)
    save_population(os.path.join(a.out, "population.pt"), [p.genome for p in players], net_cfg,
                    [p.name for p in players], {"fitness": res.fitnesses().tolist()})
    with open(os.path.join(a.out, "games.json"), "w") as f:
        json.dump([{"black": r.black, "white": r.white, "result": r.result_string, "score": r.score,
                    "moves": r.moves, "backend": r.backend, "sgf": r.to_sgf()} for r in res.records], f)
    print(f"saved population + game records to {a.out}/")

    if a.watch is not None:
        ascii_replay(res.records[a.watch])
    return res


if __name__ == "__main__":
    main()

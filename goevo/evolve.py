"""Phase 2: neuro-evolution of MLP Go players (mutation only, no crossover).

Each generation:
  1. every genome plays ``games_per_player`` games (distinct opponents, colours balanced);
  2. *winners* = genomes with more wins than losses (fallback: the best-scoring ones);
  3. only winners have children: the next generation is the winners themselves (elitism,
     optional) plus Gaussian-mutated copies of winners, split evenly among them.

Selection signal is measured against a **control run** (``selection="random"``) that is
identical except that the "winners" are chosen at random (same number as a real run would
pick), plus a fixed benchmark: every ``bench_every`` generations each genome plays a fixed
panel of random genomes and a uniform-random mover with both colours.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field

import numpy as np

from .match import GameConfig
from .parallel import GameRunner, spec_of
from .player import NetConfig, Player, RandomPlayer, layer_scales, mutate, random_genome, save_population
from .tournament import Standing, make_schedule


@dataclass
class EvoConfig:
    pop_size: int = 10
    games_per_player: int = 5
    generations: int = 200
    sigma: float = 0.1             # mutation std, relative to each layer's init std
    elitism: bool = True           # winners survive unchanged into the next generation
    selection: str = "winners"     # "winners" (real) | "random" (control, no selection)
    seed: int = 0
    bench_every: int = 5           # 0 = never
    bench_panel: int = 5           # fixed random genomes in the benchmark panel
    bench_vs_random_mover: bool = True
    checkpoint_every: int = 25
    out_dir: str = "runs/evo"
    workers: int | None = None


@dataclass
class Genome:
    id: int
    genome: np.ndarray
    parent: int | None
    born: int

    @property
    def name(self) -> str:
        return f"#{self.id}"


def _standings(pop, schedule, records):
    st = [Standing(g.name) for g in pop]
    for (bi, wi), rec in zip(schedule, records):
        for idx, sign in ((bi, 1), (wi, -1)):
            s = st[idx]
            s.games += 1
            s.margin += sign * rec.score
            if rec.winner == 0:
                s.draws += 1
            elif rec.winner == sign:
                s.wins += 1
            else:
                s.losses += 1
    return st


def pick_winners(st: list[Standing]) -> list[int]:
    w = [i for i, s in enumerate(st) if s.wins > s.losses]
    if not w:  # nobody has a winning record: take those with the most points
        best = max(s.points for s in st)
        w = [i for i, s in enumerate(st) if s.points == best]
    return sorted(w, key=lambda i: -st[i].fitness)


def make_panel(net_cfg: NetConfig, n: int, seed: int = 12345):
    rng = np.random.default_rng(seed)
    return [Player(random_genome(net_cfg, rng), net_cfg, name=f"panel{i}") for i in range(n)]


def benchmark(runner, pop_specs, opponents, net_cfg, game_cfg, seed=0):
    """Win rate of each spec against every opponent spec, both colours."""
    pairs, owner = [], []
    for i, s in enumerate(pop_specs):
        for o in opponents:
            pairs += [(s, o), (o, s)]
            owner += [(i, 1, o[1]), (i, -1, o[1])]
    recs = runner.play(pairs, net_cfg, game_cfg, seed=seed)
    wins = np.zeros(len(pop_specs))
    games = np.zeros(len(pop_specs))
    by_opp: dict[str, list[float]] = {}
    for (i, sign, oname), r in zip(owner, recs):
        v = 1.0 if r.winner == sign else 0.5 if r.winner == 0 else 0.0
        wins[i] += v
        games[i] += 1
        by_opp.setdefault(oname, []).append(v)
    return wins / np.maximum(games, 1), {k: float(np.mean(v)) for k, v in by_opp.items()}


def save_checkpoint(path, pop, net_cfg, gen, extra=None):
    save_population(path, [g.genome for g in pop], net_cfg, [g.name for g in pop],
                    {"generation": gen, "ids": [g.id for g in pop],
                     "parents": [g.parent for g in pop], "born": [g.born for g in pop], **(extra or {})})


def run_evolution(evo: EvoConfig, net_cfg: NetConfig = NetConfig(),
                  game_cfg: GameConfig = GameConfig(), verbose: bool = True,
                  runner: GameRunner | None = None) -> list[dict]:
    os.makedirs(evo.out_dir, exist_ok=True)
    with open(os.path.join(evo.out_dir, "config.json"), "w") as f:
        json.dump({"evo": asdict(evo), "net": asdict(net_cfg), "game": asdict(game_cfg.resolved())}, f, indent=1)
    rng = np.random.default_rng(evo.seed)
    scales = layer_scales(net_cfg)
    next_id = 0
    pop = []
    for _ in range(evo.pop_size):
        pop.append(Genome(next_id, random_genome(net_cfg, rng), None, 0))
        next_id += 1

    panel = [spec_of(p) for p in make_panel(net_cfg, evo.bench_panel)]
    if evo.bench_vs_random_mover:
        panel.append(spec_of(RandomPlayer("random_mover")))

    own_runner = runner is None
    runner = runner or GameRunner(evo.workers)
    history: list[dict] = []
    hist_path = os.path.join(evo.out_dir, "history.jsonl")
    open(hist_path, "w").close()
    t_start = time.time()
    try:
        for gen in range(evo.generations + 1):
            t0 = time.time()
            specs = [("genome", g.name, g.genome) for g in pop]
            schedule = make_schedule(len(pop), evo.games_per_player, seed=evo.seed * 7919 + gen)
            recs = runner.play([(specs[b], specs[w]) for b, w in schedule], net_cfg, game_cfg,
                               seed=evo.seed * 1_000_003 + gen)
            st = _standings(pop, schedule, recs)
            winners = pick_winners(st)
            if evo.selection == "random":
                winners = sorted(rng.choice(len(pop), size=len(winners), replace=False).tolist())

            h = {
                "gen": gen,
                "ids": [g.id for g in pop],
                "parents": [g.parent for g in pop],
                "wins": [s.wins for s in st],
                "losses": [s.losses for s in st],
                "margin": [s.margin / max(s.games, 1) for s in st],
                "winners": [pop[i].id for i in winners],
                "n_winners": len(winners),
                "mean_moves": float(np.mean([len(r.moves) for r in recs])),
                "black_win_rate": float(np.mean([r.winner == 1 for r in recs])),
                "mean_abs_score": float(np.mean([abs(r.score) for r in recs])),
                "truncated": int(sum(r.truncated for r in recs)),
                "mean_age": float(np.mean([gen - g.born for g in pop])),
            }

            if evo.bench_every and gen % evo.bench_every == 0:
                wr, by_opp = benchmark(runner, specs, panel, net_cfg, game_cfg, seed=gen)
                h["bench_winrate"] = wr.tolist()
                h["bench_mean"] = float(wr.mean())
                h["bench_best"] = float(wr.max())
                h["bench_by_opp"] = by_opp

            if evo.checkpoint_every and (gen % evo.checkpoint_every == 0 or gen == evo.generations):
                save_checkpoint(os.path.join(evo.out_dir, f"gen{gen:04d}.pt"), pop, net_cfg, gen,
                                {"wins": h["wins"]})

            h["seconds"] = time.time() - t0
            history.append(h)
            with open(hist_path, "a") as f:
                f.write(json.dumps(h) + "\n")
            if verbose:
                b = f" bench mean={h['bench_mean']:.3f} best={h['bench_best']:.3f}" if "bench_mean" in h else ""
                print(f"[{evo.selection}] gen {gen:4d} winners={len(winners):2d} "
                      f"moves={h['mean_moves']:.0f} age={h['mean_age']:.1f}{b} ({h['seconds']:.1f}s)",
                      flush=True)

            if gen == evo.generations:
                break
            # --- reproduction: only winners have children, mutation only ---
            parents = [pop[i] for i in winners]
            new_pop = list(parents) if evo.elitism else []
            k = 0
            order = list(rng.permutation(len(parents)))
            while len(new_pop) < evo.pop_size:
                par = parents[order[k % len(parents)]]
                new_pop.append(Genome(next_id, mutate(par.genome, net_cfg, evo.sigma, rng, scales),
                                      par.id, gen + 1))
                next_id += 1
                k += 1
            pop = new_pop[: evo.pop_size]
    finally:
        if own_runner:
            runner.close()
    if verbose:
        print(f"done in {(time.time() - t_start) / 60:.1f} min -> {evo.out_dir}")
    return history


def load_history(out_dir: str) -> list[dict]:
    with open(os.path.join(out_dir, "history.jsonl")) as f:
        return [json.loads(l) for l in f if l.strip()]

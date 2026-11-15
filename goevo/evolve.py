"""Neuro-evolution of MLP Go players (mutation only, no crossover).

Each generation:
  1. **Tournament** among the population:
     * ``"knockout"`` (default): random single-elimination bracket, one game per match
       (pop - 1 games). Rank = rounds won.
     * ``"roundrobin"``: every genome plays ``games_per_player`` games (phase-2a scheme).
  2. **Qualifier vs past champions** (knockout only, ``external_top`` > 0): the top genomes
     also play ``hof_games`` games against champions of earlier generations (hall of fame;
     prevents cycling / forgetting). fitness = rounds won + ``external_weight`` * qualifier
     win rate. (The uniform-random mover is only a yardstick, not part of fitness, unless
     ``qualifier_vs_random_mover`` is set.)
  3. **Selection**: only the best ``n_parents`` genomes have children (``n_parents = 0`` with
     round robin = phase-2a rule "more wins than losses"). Parents survive (elitism) and the
     rest of the population is Gaussian-mutated copies, split evenly among parents.

``selection="random"`` is the control: identical, except the parents are picked at random
(and no qualifier games are played, since they would not influence anything).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict

import numpy as np

from .match import GameConfig
from .parallel import GameRunner, spec_of
from .player import NetConfig, Player, RandomPlayer, layer_scales, mutate, random_genome, save_population
from .tournament import Standing, make_schedule


@dataclass
class EvoConfig:
    pop_size: int = 64
    tournament: str = "knockout"   # knockout | roundrobin
    games_per_player: int = 5      # roundrobin only
    n_parents: int = 4             # parents per generation (0 + roundrobin: "winning record" rule)
    generations: int = 1000
    sigma: float = 0.03            # mutation std, relative to each layer's init std
    elitism: bool = True           # parents survive unchanged into the next generation
    selection: str = "fitness"     # "fitness" (real) | "random" (control). "winners" == "fitness"
    external_top: int = 8          # top-k of the bracket play qualifier games (0 = off)
    external_weight: float = 1.5   # weight of qualifier win rate in fitness
    hof_size: int = 50             # hall of fame of past champions (qualifier opponents)
    hof_games: int = 4             # qualifier games vs past champions (half black, half white)
    qualifier_vs_random_mover: bool = False
    seed: int = 0
    bench_every: int = 10          # 0 = never; benchmarks the parents
    bench_panel: int = 0           # fixed random genomes in the benchmark panel
    bench_vs_random_mover: bool = True
    bench_rm_games: int = 24       # games per parent vs the random mover
    checkpoint_every: int = 250    # parents-only checkpoints (0 = only the final one)
    checkpoint_parents_only: bool = True  # save only the strongest genomes (the parents)
    state_every: int = 25          # resumable full state (state.pt, overwritten; not for git)
    resume: bool = False           # continue from out_dir/state.pt if it exists
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

    @property
    def spec(self):
        return ("genome", self.name, self.genome)


def initial_population(evo: EvoConfig, net_cfg: NetConfig):
    """Generation 0 (reproducible from the seed). Returns (genomes, rng to continue with)."""
    rng = np.random.default_rng(evo.seed)
    pop = [Genome(i, random_genome(net_cfg, rng), None, 0) for i in range(evo.pop_size)]
    return pop, rng


def _record_result(st: Standing, sign: int, rec):
    st.games += 1
    st.margin += sign * rec.score
    if rec.winner == 0:
        st.draws += 1
    elif rec.winner == sign:
        st.wins += 1
    else:
        st.losses += 1


def roundrobin(runner, pop, evo, net_cfg, game_cfg, gen):
    st = [Standing(g.name) for g in pop]
    schedule = make_schedule(len(pop), evo.games_per_player, seed=evo.seed * 7919 + gen)
    recs = runner.play([(pop[b].spec, pop[w].spec) for b, w in schedule], net_cfg, game_cfg,
                       seed=evo.seed * 1_000_003 + gen)
    for (bi, wi), rec in zip(schedule, recs):
        _record_result(st[bi], 1, rec)
        _record_result(st[wi], -1, rec)
    return st, recs


def knockout(runner, pop, rng, net_cfg, game_cfg, seed):
    """Single elimination, one game per match, random colours. Odd one out gets a bye."""
    st = [Standing(g.name) for g in pop]
    alive = list(rng.permutation(len(pop)))
    all_recs = []
    rnd = 0
    while len(alive) > 1:
        bye = alive.pop() if len(alive) % 2 else None
        pairs = []
        for k in range(0, len(alive), 2):
            a, b = alive[k], alive[k + 1]
            if rng.random() < 0.5:
                a, b = b, a
            pairs.append((a, b))  # (black, white)
        recs = runner.play([(pop[a].spec, pop[b].spec) for a, b in pairs], net_cfg, game_cfg,
                           seed=seed * 101 + rnd)
        nxt = []
        for (a, b), rec in zip(pairs, recs):
            _record_result(st[a], 1, rec)
            _record_result(st[b], -1, rec)
            nxt.append(a if rec.winner == 1 else b)  # komi x.5 -> no draws; draw -> white
        if bye is not None:
            nxt.append(bye)
        all_recs += recs
        alive = nxt
        rnd += 1
    return st, all_recs


def qualifier(runner, cands, hof, rng, net_cfg, game_cfg, seed, hof_games=4, vs_random_mover=False):
    """Each candidate plays ``hof_games`` vs sampled past champions (both colours) and,
    optionally, 2 games vs the random mover. Returns {index: win rate}."""
    rm = spec_of(RandomPlayer("random_mover"))
    pairs, owner = [], []
    for i, g in cands:
        opps = [(rm, 1), (rm, -1)] if vs_random_mover else []
        if hof and hof_games > 0:
            n_ch = max(1, hof_games // 2)
            for j in rng.choice(len(hof), size=n_ch, replace=len(hof) < n_ch):
                opps += [(hof[j], 1), (hof[j], -1)]
        if not opps:
            continue
        for o, sign in opps:
            pairs.append((g.spec, o) if sign == 1 else (o, g.spec))
            owner.append((i, sign))
    if not pairs:
        return {}
    recs = runner.play(pairs, net_cfg, game_cfg, seed=seed)
    score: dict[int, list[float]] = {}
    for (i, sign), r in zip(owner, recs):
        score.setdefault(i, []).append(1.0 if r.winner == sign else 0.5 if r.winner == 0 else 0.0)
    return {i: float(np.mean(v)) for i, v in score.items()}


def pick_winners(st: list[Standing]) -> list[int]:
    """Phase-2a rule: more wins than losses (fallback: the best-scoring ones)."""
    w = [i for i, s in enumerate(st) if s.wins > s.losses]
    if not w:
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
    if evo.selection == "winners":
        evo.selection = "fitness"
    os.makedirs(evo.out_dir, exist_ok=True)
    with open(os.path.join(evo.out_dir, "config.json"), "w") as f:
        json.dump({"evo": asdict(evo), "net": asdict(net_cfg), "game": asdict(game_cfg.resolved())}, f, indent=1)
    scales = layer_scales(net_cfg)
    pop, rng = initial_population(evo, net_cfg)
    next_id = len(pop)
    real = evo.selection == "fitness"

    panel = [spec_of(p) for p in make_panel(net_cfg, evo.bench_panel)] if evo.bench_panel else []
    if evo.bench_vs_random_mover:
        panel += [spec_of(RandomPlayer("random_mover"))] * max(1, evo.bench_rm_games // 2)
    hof: list[tuple] = []  # specs of past champions

    history: list[dict] = []
    hist_path = os.path.join(evo.out_dir, "history.jsonl")
    state_path = os.path.join(evo.out_dir, "state.pt")
    start_gen = 0
    if evo.resume and os.path.exists(state_path):
        st0 = load_state(state_path)
        pop, hof, next_id, start_gen = st0["pop"], st0["hof"], st0["next_id"], st0["next_gen"]
        rng.bit_generator.state = st0["rng"]
        history = [h for h in load_history(evo.out_dir) if h["gen"] < start_gen]
        if verbose:
            print(f"resumed from {state_path} at generation {start_gen}")
    with open(hist_path, "w") as f:
        for h in history:
            f.write(json.dumps(h) + "\n")

    own_runner = runner is None
    runner = runner or GameRunner(evo.workers)
    t_start = time.time()
    try:
        for gen in range(start_gen, evo.generations + 1):
            t0 = time.time()
            game_seed = evo.seed * 1_000_003 + gen
            if evo.tournament == "knockout":
                st, recs = knockout(runner, pop, rng, net_cfg, game_cfg, game_seed)
            else:
                st, recs = roundrobin(runner, pop, evo, net_cfg, game_cfg, gen)

            # fitness: tournament points (+ qualifier vs fixed opponents for the top-k)
            fit = np.array([s.fitness for s in st])
            qual = {}
            if real and evo.tournament == "knockout" and evo.external_top > 0:
                top = list(np.argsort(-fit, kind="stable")[: evo.external_top])
                qual = qualifier(runner, [(int(i), pop[i]) for i in top], hof, rng, net_cfg,
                                 game_cfg, game_seed + 7, evo.hof_games, evo.qualifier_vs_random_mover)
                for i, q in qual.items():
                    fit[i] += evo.external_weight * q
            order = [int(i) for i in np.argsort(-fit, kind="stable")]

            if evo.n_parents > 0:
                n_par = evo.n_parents
                parents_idx = order[:n_par]
            else:  # phase-2a rule
                parents_idx = pick_winners(st)
                n_par = len(parents_idx)
            if not real:
                parents_idx = sorted(rng.choice(len(pop), size=n_par, replace=False).tolist())
            if real:
                hof.append(pop[order[0]].spec)
                if len(hof) > evo.hof_size:  # keep the first champion, drop a random other one
                    hof.pop(int(rng.integers(1, len(hof))))

            h = {
                "gen": gen,
                "ids": [g.id for g in pop],
                "parents": [g.parent for g in pop],
                "wins": [s.wins for s in st],
                "losses": [s.losses for s in st],
                "fitness": [float(x) for x in fit],
                "qualifier": {pop[i].id: q for i, q in qual.items()},
                "winners": [pop[i].id for i in parents_idx],
                "n_winners": len(parents_idx),
                "games": len(recs),
                "mean_moves": float(np.mean([len(r.moves) for r in recs])),
                "black_win_rate": float(np.mean([r.winner == 1 for r in recs])),
                "mean_abs_score": float(np.mean([abs(r.score) for r in recs])),
                "truncated": int(sum(r.truncated for r in recs)),
                "mean_age": float(np.mean([gen - g.born for g in pop])),
            }
            if qual:
                h["qualifier_mean"] = float(np.mean(list(qual.values())))

            if evo.bench_every and gen % evo.bench_every == 0:
                specs = [pop[i].spec for i in parents_idx]
                wr, by_opp = benchmark(runner, specs, panel, net_cfg, game_cfg, seed=gen)
                h["bench_winrate"] = wr.tolist()
                h["bench_mean"] = float(wr.mean())
                h["bench_best"] = float(wr.max())
                h["bench_by_opp"] = by_opp

            last = gen == evo.generations
            if last or (evo.checkpoint_every and gen % evo.checkpoint_every == 0):
                idx = list(parents_idx) if evo.checkpoint_parents_only else list(range(len(pop)))
                save_checkpoint(os.path.join(evo.out_dir, f"gen{gen:04d}.pt"), [pop[i] for i in idx],
                                net_cfg, gen, {"fitness": [float(fit[i]) for i in idx],
                                 "selection": evo.selection})

            h["seconds"] = time.time() - t0
            history.append(h)
            with open(hist_path, "a") as f:
                f.write(json.dumps(h) + "\n")
            if verbose:
                q = f" qual={h['qualifier_mean']:.2f}" if "qualifier_mean" in h else ""
                b = f" bench={h['bench_mean']:.3f} (rm {h['bench_by_opp'].get('random_mover', float('nan')):.2f})" \
                    if "bench_mean" in h else ""
                print(f"[{evo.selection}] gen {gen:4d} parents={len(parents_idx)} games={len(recs)} "
                      f"moves={h['mean_moves']:.0f} age={h['mean_age']:.1f}{q}{b} ({h['seconds']:.1f}s)",
                      flush=True)
            if last:
                break

            # --- reproduction: only parents have children, mutation only ---
            parents = [pop[i] for i in parents_idx]
            new_pop = list(parents) if evo.elitism else []
            k = 0
            while len(new_pop) < evo.pop_size:
                par = parents[k % len(parents)]
                new_pop.append(Genome(next_id, mutate(par.genome, net_cfg, evo.sigma, rng, scales),
                                      par.id, gen + 1))
                next_id += 1
                k += 1
            pop = new_pop[: evo.pop_size]
            if evo.state_every and (gen + 1) % evo.state_every == 0:
                save_state(state_path, pop, hof, next_id, gen + 1, rng)
    finally:
        if own_runner:
            runner.close()
    if verbose:
        print(f"done in {(time.time() - t_start) / 60:.1f} min -> {evo.out_dir}")
    return history


def save_state(path, pop, hof, next_id, next_gen, rng):
    """Everything needed to resume (overwritten in place; git-ignored)."""
    import torch
    tmp = path + ".tmp"
    torch.save({"pop": [(g.id, g.genome, g.parent, g.born) for g in pop],
                "hof": [(h[1], h[2]) for h in hof], "next_id": next_id, "next_gen": next_gen,
                "rng": rng.bit_generator.state}, tmp)
    os.replace(tmp, path)


def load_state(path):
    import torch
    d = torch.load(path, weights_only=False)
    return {"pop": [Genome(i, g, p, b) for i, g, p, b in d["pop"]],
            "hof": [("genome", n, g) for n, g in d["hof"]],
            "next_id": d["next_id"], "next_gen": d["next_gen"], "rng": d["rng"]}


def load_history(out_dir: str) -> list[dict]:
    with open(os.path.join(out_dir, "history.jsonl")) as f:
        return [json.loads(l) for l in f if l.strip()]

"""Population + tournament: every genome plays ``games_per_player`` games."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .match import GameConfig, GameRecord, play_game
from .player import NetConfig, Player, random_genome


def random_population(n: int, net_cfg: NetConfig, seed: int = 0) -> list[Player]:
    rng = np.random.default_rng(seed)
    return [Player(random_genome(net_cfg, rng), net_cfg, name=f"g{i:02d}") for i in range(n)]


def make_schedule(n_players: int, games_per_player: int, seed: int = 0) -> list[tuple[int, int]]:
    """Pairings (black_idx, white_idx) so each player plays exactly ``games_per_player`` games
    against distinct opponents (circle-method round robin), with colours balanced."""
    assert n_players % 2 == 0, "use an even population size"
    assert games_per_player <= n_players - 1
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(n_players))
    fixed, rest = order[0], order[1:]
    rounds = []
    for _ in range(n_players - 1):
        lineup = [fixed] + rest
        rounds.append([(lineup[i], lineup[-1 - i]) for i in range(n_players // 2)])
        rest = rest[-1:] + rest[:-1]
    games = []
    blacks = np.zeros(n_players, dtype=int)
    for rnd in rounds[:games_per_player]:
        for a, b in rnd:
            # give black to whoever has had it less often
            if blacks[a] > blacks[b] or (blacks[a] == blacks[b] and rng.random() < 0.5):
                a, b = b, a
            blacks[a] += 1
            games.append((int(a), int(b)))
    return games


@dataclass
class Standing:
    name: str
    games: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    margin: float = 0.0  # sum of own-perspective score margins

    @property
    def points(self) -> float:
        return self.wins + 0.5 * self.draws

    @property
    def fitness(self) -> float:
        # wins dominate; average margin breaks ties
        return self.points + 1e-3 * self.margin / max(self.games, 1)


@dataclass
class TournamentResult:
    players: list[Player]
    records: list[GameRecord] = field(default_factory=list)
    standings: list[Standing] = field(default_factory=list)
    seconds: float = 0.0

    def table(self) -> str:
        rows = sorted(zip(self.players, self.standings), key=lambda t: -t[1].fitness)
        out = [f"{'rank':>4} {'name':>6} {'G':>2} {'W':>2} {'D':>2} {'L':>2} {'avg margin':>10} {'fitness':>8}"]
        for k, (p, s) in enumerate(rows, 1):
            out.append(f"{k:>4} {s.name:>6} {s.games:>2} {s.wins:>2} {s.draws:>2} {s.losses:>2} "
                       f"{s.margin / max(s.games, 1):>10.1f} {s.fitness:>8.3f}")
        return "\n".join(out)

    def fitnesses(self) -> np.ndarray:
        return np.array([s.fitness for s in self.standings])


def run_tournament(players: list[Player], games_per_player: int = 5,
                   game_cfg: GameConfig = GameConfig(), seed: int = 0,
                   verbose: bool = True) -> TournamentResult:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    res = TournamentResult(players, standings=[Standing(p.name) for p in players])
    schedule = make_schedule(len(players), games_per_player, seed)
    for k, (bi, wi) in enumerate(schedule):
        rec = play_game(players[bi], players[wi], game_cfg, rng)
        res.records.append(rec)
        for idx, sign in ((bi, 1), (wi, -1)):
            s = res.standings[idx]
            s.games += 1
            s.margin += sign * rec.score
            if rec.winner == 0:
                s.draws += 1
            elif rec.winner == sign:
                s.wins += 1
            else:
                s.losses += 1
        if verbose:
            print(f"game {k + 1:>2}/{len(schedule)}: {rec.black} (B) vs {rec.white} (W) -> "
                  f"{rec.result_string:<18} moves={len(rec.moves)}")
    res.seconds = time.time() - t0
    return res

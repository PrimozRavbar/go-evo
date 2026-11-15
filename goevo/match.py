"""Playing a single game between two players (headless), with full record."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .engine import BLACK, WHITE, EMPTY, make_game, GoGame


@dataclass
class GameConfig:
    board_size: int = 9
    komi: float = 7.5
    backend: str = "auto"          # auto | pettingzoo | native
    max_moves: int | None = None   # hard cap; default 3 * N * N
    min_moves_before_pass: int | None = None  # pass masked before this; default N*N//2
    forbid_own_eye_fill: bool = True  # never fill your own single-point eye
    temperature: float = 0.0       # 0 = deterministic argmax

    def resolved(self) -> "GameConfig":
        n2 = self.board_size ** 2
        return GameConfig(self.board_size, self.komi, self.backend,
                          self.max_moves if self.max_moves is not None else 3 * n2,
                          self.min_moves_before_pass if self.min_moves_before_pass is not None else n2 // 2,
                          self.forbid_own_eye_fill, self.temperature)


@dataclass
class GameRecord:
    black: str
    white: str
    board_size: int
    komi: float
    backend: str
    moves: list[int] = field(default_factory=list)
    boards: list[np.ndarray] = field(default_factory=list)  # boards[i] = position before move i; last = final
    score: float = 0.0       # black-perspective area score incl. komi
    winner: int = 0          # +1 black, -1 white, 0 draw
    truncated: bool = False  # hit max_moves

    @property
    def winner_name(self) -> str | None:
        return self.black if self.winner == BLACK else self.white if self.winner == WHITE else None

    @property
    def result_string(self) -> str:
        s = self.score
        r = f"B+{s:.1f}" if s > 0 else f"W+{-s:.1f}" if s < 0 else "Draw"
        return r + (" (move cap)" if self.truncated else "")

    def move_str(self, a: int) -> str:
        n = self.board_size
        if a == n * n:
            return "pass"
        r, c = divmod(a, n)
        return f"{'ABCDEFGHJKLMNOPQRST'[c]}{n - r}"

    def to_sgf(self) -> str:
        n = self.board_size
        letters = "abcdefghijklmnopqrs"
        res = "B+{:.1f}".format(self.score) if self.winner == BLACK else \
              "W+{:.1f}".format(-self.score) if self.winner == WHITE else "0"
        out = [f"(;GM[1]FF[4]SZ[{n}]KM[{self.komi}]RU[Chinese]PB[{self.black}]PW[{self.white}]RE[{res}]"]
        color = "B"
        for a in self.moves:
            if a == n * n:
                out.append(f";{color}[]")
            else:
                r, c = divmod(a, n)
                out.append(f";{color}[{letters[c]}{letters[r]}]")
            color = "W" if color == "B" else "B"
        return "".join(out) + ")"


def own_eye_mask(game: GoGame) -> np.ndarray:
    """True at empty points that are a single-point eye of the player to move.

    Point is an eye if all orthogonal neighbours are own stones and (to avoid
    false eyes) at most one diagonal is an opponent stone (none on the edge).
    """
    b = game.board
    me = game.to_play
    n = b.shape[0]
    pad = np.full((n + 2, n + 2), 2, dtype=np.int8)  # 2 = off-board
    pad[1:-1, 1:-1] = b
    orth = [pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]]
    diag = [pad[:-2, :-2], pad[:-2, 2:], pad[2:, :-2], pad[2:, 2:]]
    eye = b == EMPTY
    for o in orth:
        eye &= (o == me) | (o == 2)
    opp_diag = sum((d == -me).astype(np.int8) for d in diag)
    edge = sum((d == 2).astype(np.int8) for d in diag) > 0
    eye &= np.where(edge, opp_diag == 0, opp_diag <= 1)
    return eye.ravel()


def play_game(black, white, cfg: GameConfig = GameConfig(), rng: np.random.Generator | None = None,
              on_move=None) -> GameRecord:
    """Play one game. ``on_move(game, action, record)`` is called after each move (for live viewing)."""
    cfg = cfg.resolved()
    rng = rng or np.random.default_rng()
    game = make_game(cfg.board_size, cfg.komi, cfg.backend)
    rec = GameRecord(black.name, white.name, cfg.board_size, cfg.komi, game.backend_name)
    players = {BLACK: black, WHITE: white}
    while not game.is_over():
        if game.move_number >= cfg.max_moves:
            rec.truncated = True
            break
        rec.boards.append(game.board.copy())
        legal = game.legal_mask()
        allowed = legal.copy()
        if cfg.forbid_own_eye_fill:
            allowed[:-1] &= ~own_eye_mask(game)
        stone_moves = allowed[:-1].any()
        # Pass is only an option late enough in the game, or when nothing else is possible.
        allowed[-1] = (game.move_number >= cfg.min_moves_before_pass) or not stone_moves
        if not allowed.any():
            allowed[-1] = True
        a = players[game.to_play].select_move(game, allowed, legal, cfg.max_moves,
                                              cfg.temperature, rng)
        game.play(a)
        rec.moves.append(a)
        if on_move is not None:
            on_move(game, a, rec)
    rec.boards.append(game.board.copy())
    rec.score = float(game.score())
    rec.winner = game.winner()
    return rec

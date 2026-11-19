"""Area counting for display: who owns each point, and the running score.

Same rule as the engine's final score (Tromp-Taylor area): your stones plus the empty regions that
touch only your stones, minus komi for White. During a game, regions touching both colours count for
nobody and dead stones still count as alive, so the running score is a rough guide until the end.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .engine import BLACK, EMPTY, WHITE


def owners(board: np.ndarray, max_region: int | None = None) -> np.ndarray:
    """+1 / -1 for empty points enclosed only by black / white stones, 0 elsewhere.

    ``max_region``: empty regions larger than this count for nobody. The running display uses
    a quarter of the board, so the first stone does not "own" the whole empty board; finished
    games have no regions that large, so their count is unchanged."""
    n = board.shape[0]
    out = np.zeros_like(board, dtype=np.int8)
    seen = np.zeros(board.shape, dtype=bool)
    for r0 in range(n):
        for c0 in range(n):
            if board[r0, c0] != EMPTY or seen[r0, c0]:
                continue
            stack, region, borders = [(r0, c0)], [], set()
            seen[r0, c0] = True
            while stack:
                r, c = stack.pop()
                region.append((r, c))
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < n and 0 <= cc < n:
                        v = board[rr, cc]
                        if v == EMPTY and not seen[rr, cc]:
                            seen[rr, cc] = True
                            stack.append((rr, cc))
                        elif v != EMPTY:
                            borders.add(int(v))
            if len(borders) == 1 and (max_region is None or len(region) <= max_region):
                o = borders.pop()
                for r, c in region:
                    out[r, c] = o
    return out


@dataclass
class Count:
    black_stones: int
    black_area: int
    white_stones: int
    white_area: int
    komi: float

    @property
    def black(self) -> int:
        return self.black_stones + self.black_area

    @property
    def white(self) -> float:
        return self.white_stones + self.white_area + self.komi

    @property
    def margin(self) -> float:  # black's view, komi included (same as the engine's score)
        return self.black - self.white

    @property
    def result(self) -> str:
        m = self.margin
        return f"B+{m:.1f}" if m > 0 else f"W+{-m:.1f}" if m < 0 else "even"

    def text(self) -> str:
        return (f"Black {self.black} ({self.black_stones} stones + {self.black_area} area)  ·  "
                f"White {self.white:g} ({self.white_stones} + {self.white_area} + komi {self.komi:g})  →  {self.result}")


def count(board: np.ndarray, komi: float = 7.5, exact: bool = False) -> Count:
    """Running count; ``exact=True`` gives exactly the engine's final-score rule."""
    o = owners(board, None if exact else board.size // 4)
    return Count(int((board == BLACK).sum()), int((o == BLACK).sum()),
                 int((board == WHITE).sum()), int((o == WHITE).sum()), komi)


def score_curve(boards, komi: float = 7.5) -> np.ndarray:
    """Running margin (black's view, komi included) after each position of a game."""
    return np.array([count(b, komi).margin for b in boards])

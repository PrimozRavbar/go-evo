"""Go engines behind one small interface.

Two backends are provided:

* ``PettingZooGo`` – wraps the existing, well-tested Go API from
  `PettingZoo <https://pettingzoo.farama.org/environments/classic/go/>`_
  (``pettingzoo.classic.go_v5``, built on the minigo Go engine).
* ``NativeGo`` – our own pure-numpy implementation, used automatically when
  PettingZoo is not installed / does not import.

Both use the same rules, which match PettingZoo's:
  * area (Chinese / Tromp-Taylor style) scoring with komi,
  * suicide is illegal,
  * simple ko (immediate recapture of a single stone is illegal),
  * the game ends after two consecutive passes.

Action encoding (shared by both): ``row * N + col`` for a stone, ``N * N`` for pass.
Board encoding: ``+1`` black, ``-1`` white, ``0`` empty. Black moves first.
"""
from __future__ import annotations

import numpy as np

BLACK, WHITE, EMPTY = 1, -1, 0


class GoGame:
    """Common interface. Subclasses implement the abstract bits."""

    backend_name = "abstract"

    def __init__(self, size: int = 9, komi: float = 7.5):
        self.size = size
        self.komi = komi
        self.history: list[int] = []  # actions played, in order

    # --- to implement -----------------------------------------------------
    @property
    def board(self) -> np.ndarray:  # (N, N) int8, +1 black / -1 white
        raise NotImplementedError

    @property
    def to_play(self) -> int:  # BLACK or WHITE
        raise NotImplementedError

    def legal_mask(self) -> np.ndarray:  # (N*N+1,) bool, last entry = pass
        raise NotImplementedError

    def play(self, action: int) -> None:
        raise NotImplementedError

    def is_over(self) -> bool:
        raise NotImplementedError

    def score(self) -> float:  # area score from black's view, komi included
        raise NotImplementedError

    # --- shared helpers ---------------------------------------------------
    @property
    def pass_action(self) -> int:
        return self.size * self.size

    @property
    def move_number(self) -> int:
        return len(self.history)

    def winner(self) -> int:
        s = self.score()
        return BLACK if s > 0 else WHITE if s < 0 else 0

    def result_string(self) -> str:
        s = self.score()
        return f"B+{s:.1f}" if s > 0 else f"W+{-s:.1f}" if s < 0 else "Draw"

    def action_to_str(self, a: int) -> str:
        if a == self.pass_action:
            return "pass"
        r, c = divmod(a, self.size)
        return f"{'ABCDEFGHJKLMNOPQRST'[c]}{self.size - r}"

    def __str__(self) -> str:
        return board_to_str(self.board)


def board_to_str(board: np.ndarray) -> str:
    n = board.shape[0]
    cols = "ABCDEFGHJKLMNOPQRST"[:n]
    lines = ["   " + " ".join(cols)]
    sym = {BLACK: "X", WHITE: "O", EMPTY: "."}
    for r in range(n):
        lines.append(f"{n - r:2d} " + " ".join(sym[int(v)] for v in board[r]))
    return "\n".join(lines)


def area_score(board: np.ndarray, komi: float) -> float:
    """Tromp-Taylor area score from black's perspective (komi subtracted)."""
    n = board.shape[0]
    seen = np.zeros_like(board, dtype=bool)
    black = int((board == BLACK).sum())
    white = int((board == WHITE).sum())
    for r0 in range(n):
        for c0 in range(n):
            if board[r0, c0] != EMPTY or seen[r0, c0]:
                continue
            stack, region, borders = [(r0, c0)], 0, set()
            seen[r0, c0] = True
            while stack:
                r, c = stack.pop()
                region += 1
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < n and 0 <= cc < n:
                        v = board[rr, cc]
                        if v == EMPTY and not seen[rr, cc]:
                            seen[rr, cc] = True
                            stack.append((rr, cc))
                        elif v != EMPTY:
                            borders.add(int(v))
            if borders == {BLACK}:
                black += region
            elif borders == {WHITE}:
                white += region
    return black - white - komi


# ---------------------------------------------------------------------------
# Our own engine
# ---------------------------------------------------------------------------
class NativeGo(GoGame):
    backend_name = "native"

    def __init__(self, size: int = 9, komi: float = 7.5):
        super().__init__(size, komi)
        self._board = np.zeros((size, size), dtype=np.int8)
        self._to_play = BLACK
        self._ko: tuple[int, int] | None = None
        self._passes = 0
        n = size
        self._nbrs = {
            (r, c): [(rr, cc) for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                     if 0 <= rr < n and 0 <= cc < n]
            for r in range(n) for c in range(n)
        }

    @property
    def board(self):
        return self._board

    @property
    def to_play(self):
        return self._to_play

    def _group(self, board, p):
        color = board[p]
        stack, group, libs = [p], {p}, set()
        while stack:
            q = stack.pop()
            for nb in self._nbrs[q]:
                v = board[nb]
                if v == EMPTY:
                    libs.add(nb)
                elif v == color and nb not in group:
                    group.add(nb)
                    stack.append(nb)
        return group, libs

    def _try(self, p, color):
        """Return (new_board, captured) or None if illegal."""
        if self._board[p] != EMPTY or p == self._ko:
            return None
        b = self._board.copy()
        b[p] = color
        captured = []
        for nb in self._nbrs[p]:
            if b[nb] == -color:
                g, libs = self._group(b, nb)
                if not libs:
                    for q in g:
                        b[q] = EMPTY
                    captured.extend(g)
        if not captured:
            _, libs = self._group(b, p)
            if not libs:
                return None  # suicide
        return b, captured

    def _group_libs(self):
        """Label every group once: returns (group_id board, {gid: liberty count})."""
        b = self._board
        gid = -np.ones(b.shape, dtype=np.int32)
        nlibs = {}
        k = 0
        for r, c in np.argwhere(b != EMPTY):
            p = (int(r), int(c))
            if gid[p] >= 0:
                continue
            group, libs = self._group(b, p)
            for q in group:
                gid[q] = k
            nlibs[k] = len(libs)
            k += 1
        return gid, nlibs

    def legal_mask(self):
        n = self.size
        b = self._board
        me = self._to_play
        mask = np.zeros(n * n + 1, dtype=bool)
        mask[-1] = True
        mask[:-1] = (b == EMPTY).ravel()
        # Points with no empty neighbour may be suicide; check those against group liberties.
        pad = np.ones((n + 2, n + 2), dtype=bool)
        pad[1:-1, 1:-1] = b != EMPTY
        surrounded = (b == EMPTY) & pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
        if surrounded.any():
            gid, nlibs = self._group_libs()
            for r, c in np.argwhere(surrounded):
                p = (int(r), int(c))
                ok = False
                for nb in self._nbrs[p]:
                    libs = nlibs[gid[nb]]
                    if (b[nb] == me and libs > 1) or (b[nb] == -me and libs == 1):
                        ok = True
                        break
                mask[r * n + c] = ok
        if self._ko is not None:
            mask[self._ko[0] * n + self._ko[1]] = False
        return mask

    def play(self, action):
        action = int(action)
        if self.is_over():
            raise ValueError("game is over")
        if action == self.pass_action:
            self._passes += 1
            self._ko = None
        else:
            p = divmod(action, self.size)
            color = self._to_play
            koish = all(self._board[nb] == -color for nb in self._nbrs[p])
            res = self._try(p, color)
            if res is None:
                raise ValueError(f"illegal move {self.action_to_str(action)}")
            self._board, captured = res
            self._ko = captured[0] if (len(captured) == 1 and koish) else None
            self._passes = 0
        self._to_play = -self._to_play
        self.history.append(action)

    def is_over(self):
        return self._passes >= 2

    def score(self):
        return area_score(self._board, self.komi)


# ---------------------------------------------------------------------------
# PettingZoo backend
# ---------------------------------------------------------------------------
class PettingZooGo(GoGame):
    """Drives ``pettingzoo.classic.go_v5`` through its standard AEC API."""

    backend_name = "pettingzoo"

    def __init__(self, size: int = 9, komi: float = 7.5):
        super().__init__(size, komi)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from pettingzoo.classic import go_v5

        self.env = go_v5.env(board_size=size, komi=komi)
        self.env.reset()
        self._over = False

    @property
    def _pos(self):
        return self.env.unwrapped._go  # minigo Position (board, to_play, ...)

    @property
    def board(self):
        return self._pos.board

    @property
    def to_play(self):
        return int(self._pos.to_play)

    def legal_mask(self):
        obs, _, _, _, _ = self.env.last()
        return obs["action_mask"].astype(bool)

    def play(self, action):
        if self._over:
            raise ValueError("game is over")
        if not self.legal_mask()[action]:
            raise ValueError(f"illegal move {self.action_to_str(action)}")
        self.env.step(int(action))
        self.history.append(int(action))
        if any(self.env.terminations.values()) or any(self.env.truncations.values()):
            self._over = True

    def is_over(self):
        return self._over

    def score(self):
        return float(self._pos.score())


def pettingzoo_available() -> bool:
    try:
        from pettingzoo.classic import go_v5  # noqa: F401
        return True
    except Exception:
        return False


def numba_available() -> bool:
    try:
        import numba  # noqa: F401
        return True
    except Exception:
        return False


def make_game(size: int = 19, komi: float = 7.5, backend: str = "auto") -> GoGame:
    """backend: 'fast' (numba), 'pettingzoo', 'native' (pure python), or 'auto'
    (fast if numba is installed, else PettingZoo, else native). All three play identically."""
    if backend == "auto":
        backend = "fast" if numba_available() else "pettingzoo" if pettingzoo_available() else "native"
    if backend == "fast":
        from .fastgo import FastGo
        return FastGo(size, komi)
    if backend == "pettingzoo":
        return PettingZooGo(size, komi)
    if backend == "native":
        return NativeGo(size, komi)
    raise ValueError(f"unknown backend {backend!r}")

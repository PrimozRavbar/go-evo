"""Numba-compiled Go engine (same rules as NativeGo / PettingZoo, ~10x faster).

Board is stored flat with a 1-point border: index = (r + 1) * (N + 2) + (c + 1);
border cells hold OFF. Colours: BLACK = 1, WHITE = -1, EMPTY = 0.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .engine import BLACK, WHITE, EMPTY, GoGame

OFF = 3


@njit(cache=True)
def _flood(board, start, nbr, gid, label, stack, lib_mark, stamp):
    """Label the group containing ``start`` with ``label``; return its liberty count."""
    color = board[start]
    sp = 0
    stack[sp] = start
    sp += 1
    gid[start] = label
    libs = 0
    while sp > 0:
        sp -= 1
        p = stack[sp]
        for k in range(4):
            q = p + nbr[k]
            v = board[q]
            if v == EMPTY:
                if lib_mark[q] != stamp:
                    lib_mark[q] = stamp
                    libs += 1
            elif v == color and gid[q] != label:
                gid[q] = label
                stack[sp] = q
                sp += 1
    return libs


@njit(cache=True)
def _remove(board, start, nbr, stack):
    color = board[start]
    sp = 0
    stack[sp] = start
    sp += 1
    board[start] = EMPTY
    n = 0
    while sp > 0:
        sp -= 1
        p = stack[sp]
        n += 1
        for k in range(4):
            q = p + nbr[k]
            if board[q] == color:
                board[q] = EMPTY
                stack[sp] = q
                sp += 1
    return n


@njit(cache=True)
def _group_libs(board, nbr, gid, nlibs, stack, lib_mark):
    """Label all groups; nlibs[label] = liberties. Returns number of groups."""
    gid[:] = -1
    stamp = 0
    for p in range(board.size):
        v = board[p]
        if (v == BLACK or v == WHITE) and gid[p] < 0:
            stamp += 1
            nlibs[stamp] = _flood(board, p, nbr, gid, stamp, stack, lib_mark, stamp)
    return stamp


@njit(cache=True)
def legal_mask_nb(board, color, ko, n, nbr, gid, nlibs, stack, lib_mark):
    W = n + 2
    mask = np.zeros(n * n + 1, dtype=np.bool_)
    mask[n * n] = True
    lib_mark[:] = 0
    labelled = False
    for r in range(n):
        for c in range(n):
            p = (r + 1) * W + c + 1
            if board[p] != EMPTY or p == ko:
                continue
            ok = False
            for k in range(4):
                if board[p + nbr[k]] == EMPTY:
                    ok = True
                    break
            if not ok:
                if not labelled:
                    _group_libs(board, nbr, gid, nlibs, stack, lib_mark)
                    labelled = True
                for k in range(4):
                    q = p + nbr[k]
                    v = board[q]
                    if v == color and nlibs[gid[q]] > 1:
                        ok = True
                        break
                    if v == -color and nlibs[gid[q]] == 1:
                        ok = True
                        break
            mask[r * n + c] = ok
    return mask


@njit(cache=True)
def play_nb(board, p, color, nbr, gid, stack, lib_mark):
    """Place a stone (assumed legal). Returns the new ko point (or -1)."""
    koish = True
    for k in range(4):
        v = board[p + nbr[k]]
        if v != -color and v != OFF:
            koish = False
    board[p] = color
    captured = 0
    last_cap = -1
    gid[:] = -1
    lib_mark[:] = 0
    stamp = 0
    for k in range(4):
        q = p + nbr[k]
        if board[q] == -color:
            stamp += 1
            if _flood(board, q, nbr, gid, stamp, stack, lib_mark, stamp) == 0:
                captured += _remove(board, q, nbr, stack)
                last_cap = q
    if captured == 1 and koish:
        return last_cap
    return -1


@njit(cache=True)
def liberty_map_nb(board, n, nbr, gid, nlibs, stack, lib_mark):
    lib_mark[:] = 0
    _group_libs(board, nbr, gid, nlibs, stack, lib_mark)
    W = n + 2
    out = np.zeros(n * n, dtype=np.int64)
    for r in range(n):
        for c in range(n):
            p = (r + 1) * W + c + 1
            if board[p] == BLACK or board[p] == WHITE:
                out[r * n + c] = nlibs[gid[p]]
    return out


@njit(cache=True)
def own_eye_nb(board, color, n):
    W = n + 2
    out = np.zeros(n * n, dtype=np.bool_)
    for r in range(n):
        for c in range(n):
            p = (r + 1) * W + c + 1
            if board[p] != EMPTY:
                continue
            eye = True
            for q in (p - W, p + W, p - 1, p + 1):
                v = board[q]
                if v != color and v != OFF:
                    eye = False
                    break
            if not eye:
                continue
            opp = 0
            edge = False
            for q in (p - W - 1, p - W + 1, p + W - 1, p + W + 1):
                v = board[q]
                if v == OFF:
                    edge = True
                elif v == -color:
                    opp += 1
            out[r * n + c] = (opp == 0) if edge else (opp <= 1)
    return out


@njit(cache=True)
def area_score_nb(board, n, komi, stack, seen):
    W = n + 2
    seen[:] = False
    black = 0
    white = 0
    for p in range(board.size):
        if board[p] == BLACK:
            black += 1
        elif board[p] == WHITE:
            white += 1
    for p in range(board.size):
        if board[p] != EMPTY or seen[p]:
            continue
        sp = 0
        stack[sp] = p
        sp += 1
        seen[p] = True
        region = 0
        touch_b = False
        touch_w = False
        while sp > 0:
            sp -= 1
            q = stack[sp]
            region += 1
            for d in (-W, W, -1, 1):
                t = q + d
                v = board[t]
                if v == EMPTY and not seen[t]:
                    seen[t] = True
                    stack[sp] = t
                    sp += 1
                elif v == BLACK:
                    touch_b = True
                elif v == WHITE:
                    touch_w = True
        if touch_b and not touch_w:
            black += region
        elif touch_w and not touch_b:
            white += region
    return black - white - komi


class FastGo(GoGame):
    backend_name = "fast"

    def __init__(self, size: int = 19, komi: float = 7.5):
        super().__init__(size, komi)
        n, W = size, size + 2
        self._W = W
        self._b = np.full(W * W, OFF, dtype=np.int8)
        for r in range(n):
            self._b[(r + 1) * W + 1:(r + 1) * W + 1 + n] = EMPTY
        self._nbr = np.array([-W, W, -1, 1], dtype=np.int64)
        self._gid = np.empty(W * W, dtype=np.int64)
        self._nlibs = np.zeros(W * W + 1, dtype=np.int64)
        self._stack = np.empty(W * W, dtype=np.int64)
        self._mark = np.zeros(W * W, dtype=np.int64)
        self._seen = np.zeros(W * W, dtype=np.bool_)
        self._to_play = BLACK
        self._ko = -1
        self._passes = 0
        self._mask_cache = None

    def _idx(self, a):
        r, c = divmod(a, self.size)
        return (r + 1) * self._W + c + 1

    @property
    def board(self):
        n = self.size
        return self._b.reshape(self._W, self._W)[1:n + 1, 1:n + 1]

    @property
    def to_play(self):
        return self._to_play

    def legal_mask(self):
        if self._mask_cache is None:
            self._mask_cache = legal_mask_nb(self._b, self._to_play, self._ko, self.size, self._nbr,
                                             self._gid, self._nlibs, self._stack, self._mark)
        return self._mask_cache

    def liberty_map(self):
        return liberty_map_nb(self._b, self.size, self._nbr, self._gid, self._nlibs, self._stack,
                              self._mark).reshape(self.size, self.size)

    def own_eye_mask(self):
        return own_eye_nb(self._b, self._to_play, self.size)

    def play(self, action):
        action = int(action)
        if self.is_over():
            raise ValueError("game is over")
        if not self.legal_mask()[action]:
            raise ValueError(f"illegal move {self.action_to_str(action)}")
        if action == self.pass_action:
            self._passes += 1
            self._ko = -1
        else:
            self._ko = play_nb(self._b, self._idx(action), self._to_play, self._nbr, self._gid,
                               self._stack, self._mark)
            self._passes = 0
        self._to_play = -self._to_play
        self._mask_cache = None
        self.history.append(action)

    def is_over(self):
        return self._passes >= 2

    def score(self):
        return float(area_score_nb(self._b, self.size, self.komi, self._stack, self._seen))

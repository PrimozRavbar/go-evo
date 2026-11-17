"""Play against a network by clicking on the board (ipywidgets; works in Colab and Jupyter).

    from goevo.clickboard import ClickGame
    ClickGame(opponent_player, human_color=1).show()     # 1 = you play black, -1 = white

Same rules as training and evaluation (``GameConfig``): pass only when no stone move is
possible (it then happens automatically), no filling your own single-point eye, move cap.
"""
from __future__ import annotations

import numpy as np

from .engine import BLACK, WHITE, make_game
from .match import GameConfig
from .ppo import allowed_moves

COLS = "ABCDEFGHJKLMNOPQRST"
CELL = "28px"

# Each button is one intersection: CSS draws the two grid lines through its centre (half lines
# on the edges), the star points, the stone (80% of the line spacing) and a red dot on the last move.
CSS = """<style>
.gocell { --mark: none; --stone: none;
  --vs: 1px 100%; --vp: center center; --hs: 100% 1px; --hp: center center;
  background-color: #dcb35c !important;
  background-image: var(--mark), var(--stone), linear-gradient(#333, #333), linear-gradient(#333, #333) !important;
  background-size: 100% 100%, 100% 100%, var(--vs), var(--hs) !important;
  background-position: center, center, var(--vp), var(--hp) !important;
  background-repeat: no-repeat !important;
  border: none !important; box-shadow: none !important; outline: none !important;
  padding: 0 !important; margin: 0 !important; min-width: 0 !important; border-radius: 0 !important; }
.gocell.top    { --vs: 1px 50%; --vp: center bottom; }
.gocell.bottom { --vs: 1px 50%; --vp: center top; }
.gocell.left   { --hs: 50% 1px; --hp: right center; }
.gocell.right  { --hs: 50% 1px; --hp: left center; }
.gocell.star   { --stone: radial-gradient(circle, #333 0 2.5px, transparent 3.5px); }
.gocell.b { --stone: radial-gradient(circle at 42% 40%, #555 0, #111 7px, #000 10.5px, transparent 11.5px); }
.gocell.w { --stone: radial-gradient(circle at 42% 40%, #fff 0, #eee 7px, #bbb 10px, #666 10.5px, transparent 11.5px); }
.gocell.last { --mark: radial-gradient(circle, #e01010 0 3.5px, transparent 4.5px); }
.gocell.e:hover { --mark: radial-gradient(circle, rgba(0,0,0,0.28) 0 10px, transparent 11px); }
.golab { margin: 0 !important; padding: 0 !important; height: 28px !important; min-height: 0 !important;
  line-height: 28px !important; font: 11px sans-serif; color: #333; overflow: hidden; }
.goboard { background-color: #dcb35c !important; gap: 0 !important; padding: 4px !important;
  grid-auto-rows: 28px !important; align-items: stretch !important; width: max-content !important; }
.goboard > * { margin: 0 !important; }
</style>"""


class ClickGame:
    def __init__(self, opponent, human_color: int = BLACK, cfg: GameConfig | None = None,
                 temperature: float = 0.0, seed: int = 0):
        import ipywidgets as w
        self.w = w
        self.ai, self.me = opponent, human_color
        self.gc = (cfg or GameConfig(board_size=19)).resolved()
        self.temperature, self.rng = temperature, np.random.default_rng(seed)
        n = self.n = self.gc.board_size
        self.buttons = []
        stars = {3, n // 2, n - 4} if n >= 13 else {2, n // 2, n - 3}
        self._base = []
        for r in range(n):
            for c in range(n):
                k = ["gocell"] + [cl for cl, on in (("top", r == 0), ("bottom", r == n - 1),
                                                     ("left", c == 0), ("right", c == n - 1)) if on]
                self._base.append(tuple(k + (["star"] if r in stars and c in stars else [])))
        lab = lambda t, just: w.Label(t, layout=w.Layout(width=CELL, height=CELL, margin="0", justify_content=just))
        cells = [lab("", "center")] + [lab(c, "center") for c in COLS[:n]]
        for r in range(n):
            cells.append(lab(str(n - r), "center"))
            for c in range(n):
                b = w.Button(description="", layout=w.Layout(width=CELL, height=CELL))
                b.on_click(lambda _, a=r * n + c: self.click(a))
                self.buttons.append(b)
                cells.append(b)
        for x in cells:
            if isinstance(x, w.Label):
                x.add_class("golab")
        self.grid = w.GridBox(cells, layout=w.Layout(grid_template_columns=f"repeat({n + 1}, {CELL})",
                                                     grid_template_rows=f"repeat({n + 1}, {CELL})",
                                                     grid_gap="0px"))
        self.grid.add_class("goboard")
        self.status = w.HTML()
        self.new_btn = w.Button(description="New game", button_style="info")
        self.new_btn.on_click(lambda _: self.reset())
        self.reset()

    # ------------------------------------------------------------------ game flow
    def reset(self):
        self.game = make_game(self.n, self.gc.komi, self.gc.backend)
        self.moves, self.last = [], None
        self.msg = ""
        self._advance()

    def _allowed(self):
        legal = self.game.legal_mask()
        return legal, allowed_moves(self.game, legal, self.gc)

    def _over(self):
        return self.game.is_over() or self.game.move_number >= self.gc.max_moves

    def _play(self, a):
        self.game.play(a)
        self.moves.append(a)
        self.last = a

    def _advance(self):
        """Let the network move (and auto-pass for you when you have no stone move)."""
        while not self._over():
            legal, allowed = self._allowed()
            if self.game.to_play == self.me:
                if allowed[:-1].any():
                    break
                self.msg = "You had no allowed stone move: passed for you."
                self._play(self.n * self.n)
                continue
            a = self.ai.select_move(self.game, allowed, legal, self.gc.max_moves, self.temperature, self.rng)
            self._play(a)
        self._render()

    def click(self, a):
        if self._over() or self.game.to_play != self.me:
            return
        _, allowed = self._allowed()
        if not allowed[a]:
            self.msg = f"{self._name(a)} is not allowed (occupied, suicide, ko or your own eye)."
            self._render()
            return
        self.msg = ""
        self._play(a)
        self.status.value = "<b>thinking…</b>"
        self._advance()

    # ------------------------------------------------------------------ drawing
    def _name(self, a):
        return "pass" if a == self.n * self.n else f"{COLS[a % self.n]}{self.n - a // self.n}"

    def _render(self):
        b = self.game.board.ravel()
        for i, btn in enumerate(self.buttons):
            st = ("b",) if b[i] == BLACK else ("w",) if b[i] == WHITE else ("e",)
            cls = self._base[i] + st + (("last",) if i == self.last else ())
            if tuple(btn._dom_classes) != cls:
                btn._dom_classes = cls
        you = "Black" if self.me == BLACK else "White"
        last = f"last move: {self._name(self.moves[-1])} ({'B' if len(self.moves) % 2 else 'W'})" if self.moves else ""
        if self._over():
            s = self.game.score()
            res = f"B+{s:.1f}" if s > 0 else f"W+{-s:.1f}" if s < 0 else "draw"
            won = (s > 0) == (self.me == BLACK) and s != 0
            head = f"<b>Game over: {res}. {'You win!' if won else 'You lose.'}</b>"
        else:
            head = f"You play <b>{you}</b> vs <b>{self.ai.name}</b>. Your move."
        self.status.value = (f"{head}<br>move {self.game.move_number} &nbsp; {last}"
                             f"<br><span style='color:#b00'>{self.msg}</span>")

    def show(self):
        from IPython.display import display
        display(self.w.VBox([self.w.HTML(CSS), self.status, self.grid, self.new_btn]))
        return self

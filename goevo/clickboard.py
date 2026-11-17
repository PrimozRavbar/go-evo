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
WOOD, CELL = "#dcb35c", "28px"


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
        cells = [w.Label("")] + [w.Label(c, layout=w.Layout(width=CELL, justify_content="center")) for c in COLS[:n]]
        for r in range(n):
            cells.append(w.Label(str(n - r), layout=w.Layout(width=CELL, justify_content="flex-end")))
            for c in range(n):
                b = w.Button(description="", layout=w.Layout(width=CELL, height=CELL, padding="0", margin="0",
                                                             border="1px solid #8b6914"))
                b.style.button_color = WOOD
                b.on_click(lambda _, a=r * n + c: self.click(a))
                self.buttons.append(b)
                cells.append(b)
        self.grid = w.GridBox(cells, layout=w.Layout(grid_template_columns=f"repeat({n + 1}, {CELL})",
                                                     grid_gap="0px"))
        self.status = w.HTML()
        self.new_btn = w.Button(description="New game", button_style="info")
        self.new_btn.on_click(lambda _: self.reset())
        self.reset()

    # ------------------------------------------------------------------ game flow
    def reset(self):
        self.game = make_game(self.n, self.gc.komi, self.gc.backend)
        self.moves, self.last = [], None
        self._shown = np.full(self.n * self.n, 9, dtype=np.int8)
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
        for i in np.flatnonzero(b != self._shown):
            btn = self.buttons[i]
            btn.style.button_color = "black" if b[i] == BLACK else "white" if b[i] == WHITE else WOOD
        self._shown = b.copy()
        for i, btn in enumerate(self.buttons):  # red border on the last move
            border = "3px solid red" if i == self.last else "1px solid #8b6914"
            if btn.layout.border != border:
                btn.layout.border = border
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
        display(self.w.VBox([self.status, self.grid, self.new_btn]))
        return self

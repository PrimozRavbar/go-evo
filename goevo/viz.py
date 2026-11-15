"""Watching games: ASCII, matplotlib frames, Colab/Jupyter animations and live play."""
from __future__ import annotations

import time

import numpy as np

from .engine import BLACK, WHITE, board_to_str
from .match import GameRecord, GameConfig, play_game


def ascii_replay(rec: GameRecord, every: int = 1, delay: float = 0.0) -> None:
    for i in range(0, len(rec.moves), every):
        print(f"move {i + 1}: {'B' if i % 2 == 0 else 'W'} {rec.move_str(rec.moves[i])}")
        print(board_to_str(rec.boards[i + 1]))
        print()
        if delay:
            time.sleep(delay)
    print(f"{rec.black} (B) vs {rec.white} (W): {rec.result_string}")


def draw_board(board: np.ndarray, ax=None, last_move: int | None = None, title: str = ""):
    import matplotlib.pyplot as plt

    n = board.shape[0]
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.clear()
    ax.set_facecolor("#dcb35c")
    for i in range(n):
        ax.plot([0, n - 1], [i, i], color="k", lw=0.8, zorder=1)
        ax.plot([i, i], [0, n - 1], color="k", lw=0.8, zorder=1)
    if n == 9:
        for r, c in [(2, 2), (2, 6), (6, 2), (6, 6), (4, 4)]:
            ax.plot(c, n - 1 - r, "k.", ms=6, zorder=1)
    rs, cs = np.nonzero(board == BLACK)
    ax.scatter(cs, n - 1 - rs, s=3000 / n * 3, c="black", edgecolors="black", zorder=2)
    rs, cs = np.nonzero(board == WHITE)
    ax.scatter(cs, n - 1 - rs, s=3000 / n * 3, c="white", edgecolors="black", zorder=2)
    if last_move is not None and last_move < n * n:
        r, c = divmod(last_move, n)
        ax.scatter([c], [n - 1 - r], s=60, c="red", zorder=3)
    ax.set_xticks(range(n))
    ax.set_xticklabels(list("ABCDEFGHJKLMNOPQRST"[:n]))
    ax.set_yticks(range(n))
    ax.set_yticklabels([str(i + 1) for i in range(n)])
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_aspect("equal")
    ax.set_title(title)
    return ax


def _frame_title(rec: GameRecord, i: int) -> str:
    if i == 0:
        return f"{rec.black} (B) vs {rec.white} (W) — start"
    who = "B" if (i - 1) % 2 == 0 else "W"
    t = f"move {i}/{len(rec.moves)}: {who} {rec.move_str(rec.moves[i - 1])}"
    if i == len(rec.moves):
        t += f" — {rec.result_string}"
    return t


def animate_game(rec: GameRecord, interval_ms: int = 300, every: int = 1, dpi: int = 60):
    """Return an HTML animation (with play/pause/step controls) — display it in Colab.

    ``every`` shows only every k-th position (useful for long 19x19 games)."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from IPython.display import HTML

    matplotlib.rcParams["animation.embed_limit"] = 200  # MB; 19x19 games have hundreds of frames
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    frames = list(range(0, len(rec.boards), every))
    if frames[-1] != len(rec.boards) - 1:
        frames.append(len(rec.boards) - 1)

    def update(i):
        draw_board(rec.boards[i], ax, rec.moves[i - 1] if i else None, _frame_title(rec, i))

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=interval_ms)
    html = HTML(anim.to_jshtml())
    plt.close(fig)
    return html


def slider_viewer(rec: GameRecord):
    """ipywidgets slider to scrub through a recorded game."""
    import matplotlib.pyplot as plt
    import ipywidgets as w

    def show(i):
        fig, ax = plt.subplots(figsize=(5, 5))
        draw_board(rec.boards[i], ax, rec.moves[i - 1] if i else None, _frame_title(rec, i))
        plt.show()

    return w.interact(show, i=w.IntSlider(min=0, max=len(rec.boards) - 1, step=1, value=0,
                                          description="move"))


def watch_live(black, white, cfg: GameConfig = GameConfig(), delay: float = 0.3,
               rng: np.random.Generator | None = None) -> GameRecord:
    """Play a new game and redraw the board after every move (Colab / Jupyter)."""
    import matplotlib.pyplot as plt
    from IPython.display import clear_output, display

    def on_move(game, a, rec):
        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(5, 5))
        i = len(rec.moves)
        draw_board(game.board, ax, a,
                   f"{rec.black} (B) vs {rec.white} (W) — move {i}: "
                   f"{'B' if (i - 1) % 2 == 0 else 'W'} {rec.move_str(a)}")
        display(fig)
        plt.close(fig)
        time.sleep(delay)

    rec = play_game(black, white, cfg, rng, on_move=on_move)
    print(f"Result: {rec.result_string} ({len(rec.moves)} moves)")
    return rec

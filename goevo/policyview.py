"""The network's output layer next to the board: move probabilities (or raw logits) as a heatmap.

    from goevo.policyview import PolicyClickGame, play_game_with_policy, policy_slider
    PolicyClickGame(opponent_player, human_color=1).show()      # click board, heatmap on its left
    rec = play_game_with_policy(black, white, cfg, rng)         # GameRecord + rec.policy
    policy_slider(rec)                                          # heatmap | board, move by move

Convention everywhere: the left board shows the position *before* move i with the distribution
the player to move had there, and a blue ring on the move it then played; the right board shows
the position *after* move i (red dot on the move).

Display modes:
* ``prob``: softmax of the logits over the allowed moves (temperature 1, the network's own
  distribution; play may use another temperature). Colour scale 0 .. the top probability.
* ``log prob``: log10 of the same probabilities, fixed scale -4 .. 0.
* ``logit``: the raw last-layer output for every point, legal or not, scale min .. max.
Points that are empty but not allowed (ko, suicide, own eye under phase 1-5 rules) get a grey x.
"""
from __future__ import annotations

import numpy as np

from .engine import BLACK, WHITE
from .match import GameConfig, GameRecord, play_game
from .player import encode_planes, encode_state
from .clickboard import CSS, ClickGame
from .ppo import allowed_moves

COLS = "ABCDEFGHJKLMNOPQRST"
MODES = ("prob", "log prob", "logit")
TOP_K = 6          # points labelled with their value
LOG_FLOOR = -4.0   # log10 prob scale bottom
CMAP = "YlOrRd"


# ----------------------------------------------------------------------------- policy
def policy(player, game, allowed, legal, max_moves: int) -> dict | None:
    """Last-layer output of ``player`` for the position in ``game`` (side to move's view).
    None for players without a network (random mover, you)."""
    if not hasattr(player, "logits") or not hasattr(player, "cfg"):
        return None
    cfg = player.cfg
    feats = (encode_planes(game, legal, cfg.extra_planes) if cfg.arch in ("conv", "resnet")
             else encode_state(game, legal, max_moves))
    z = player.logits(feats).astype(np.float64)
    zm = np.where(allowed, z, -np.inf)
    p = np.exp(zm - zm.max())
    p /= p.sum()
    nz = p[p > 0]
    return {"logits": z, "allowed": allowed.copy(), "probs": p, "to_play": game.to_play,
            "entropy": float(-(nz * np.log(nz)).sum()), "name": player.name}


def policy_now(player, game, gc: GameConfig) -> dict | None:
    """``policy`` with the allowed moves worked out from the rules in ``gc``."""
    gc = gc.resolved()
    legal = game.legal_mask()
    return policy(player, game, allowed_moves(game, legal, gc), legal, gc.max_moves)


def _values(pol: dict, mode: str):
    """(values over the N*N+1 actions, vmin, vmax, label of each value)."""
    p, z, al = pol["probs"], pol["logits"], pol["allowed"]
    if mode == "logit":
        lo, hi = float(z.min()), float(z.max())
        return z, lo, hi if hi > lo else lo + 1, lambda a: f"{z[a]:.1f}"
    pct = lambda a: f"{100 * p[a]:.0f}%" if p[a] >= 0.095 else f"{100 * p[a]:.1f}"
    if mode == "log prob":
        v = np.where(al, np.log10(np.maximum(p, 10 ** LOG_FLOOR)), LOG_FLOOR)
        return v, LOG_FLOOR, 0.0, pct
    return p, 0.0, max(float(p.max()), 1e-9), pct


def _top(pol: dict, mode: str, board: np.ndarray, k: int = TOP_K):
    """Empty points to label: the k best by the displayed value."""
    n2 = board.size
    v = pol["logits"][:n2] if mode == "logit" else np.where(pol["allowed"][:n2], pol["probs"][:n2], -1)
    v = np.where(board.ravel() == 0, v, -np.inf)
    order = np.argsort(-v)[:k]
    if mode != "logit":
        order = [a for a in order if pol["probs"][a] >= 0.005]
    return list(order)


def _summary(pol: dict, mode: str) -> str:
    p, n2 = pol["probs"], len(pol["probs"]) - 1
    top = int(np.argmax(np.where(pol["allowed"], p, -1)))
    n = int(round(n2 ** 0.5))
    name = "pass" if top == n2 else f"{COLS[top % n]}{n - top // n}"
    return (f"top {name} {100 * p[top]:.1f}% · pass {100 * p[n2]:.1f}% (logit {pol['logits'][n2]:.2f})"
         f" · entropy {pol['entropy']:.2f} nats ≈ {np.exp(pol['entropy']):.1f} moves")


# ----------------------------------------------------------------------------- SVG (click board)
def policy_svg(board: np.ndarray, pol: dict | None, mode: str = "prob", ring: int | None = None,
               cell: int = 28, pad: int = 4, note: str = "") -> str:
    """Heatmap board as an SVG string, laid out like the click board (same size and labels)."""
    import matplotlib
    from matplotlib.colors import to_hex, to_rgb

    n = board.shape[0]
    n2 = n * n
    W = pad * 2 + cell * (n + 1)
    cx = lambda c: pad + cell * (c + 1) + cell / 2
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{W + 46}" '
           f'style="font-family:sans-serif">',
           f'<rect width="{W}" height="{W}" fill="#f7f4ec"/>']
    for i in range(n):
        out.append(f'<text x="{cx(i)}" y="{pad + cell / 2 + 4}" font-size="11" fill="#333" '
                   f'text-anchor="middle">{COLS[i]}</text>')
        out.append(f'<text x="{pad + cell / 2}" y="{cx(i) + 4}" font-size="11" fill="#333" '
                   f'text-anchor="middle">{n - i}</text>')
    cmap = matplotlib.colormaps[CMAP]
    if pol is not None:
        v, lo, hi, lab = _values(pol, mode)
        al = pol["allowed"]
        for a in range(n2):
            if mode != "logit" and not al[a]:
                continue
            t = min(max((v[a] - lo) / (hi - lo), 0.0), 1.0)
            r, c = divmod(a, n)
            out.append(f'<rect x="{cx(c) - cell / 2}" y="{cx(r) - cell / 2}" width="{cell}" '
                       f'height="{cell}" fill="{to_hex(cmap(t))}"/>')
    for i in range(n):  # faint grid
        a, b = cx(0), cx(n - 1)
        out.append(f'<line x1="{a}" y1="{cx(i)}" x2="{b}" y2="{cx(i)}" stroke="#000" '
                   f'stroke-opacity="0.18" stroke-width="1"/>')
        out.append(f'<line x1="{cx(i)}" y1="{a}" x2="{cx(i)}" y2="{b}" stroke="#000" '
                   f'stroke-opacity="0.18" stroke-width="1"/>')
    rs = cell * 0.3
    for a in np.flatnonzero(board.ravel() != 0):
        r, c = divmod(int(a), n)
        fill = "#111" if board[r, c] == BLACK else "#fff"
        out.append(f'<circle cx="{cx(c)}" cy="{cx(r)}" r="{rs}" fill="{fill}" stroke="#333" '
                   f'stroke-width="1"/>')
    if pol is not None:
        al = pol["allowed"]
        for a in np.flatnonzero((board.ravel() == 0) & ~al[:n2]):
            r, c = divmod(int(a), n)
            d = cell * 0.15
            out.append(f'<path d="M{cx(c) - d},{cx(r) - d}L{cx(c) + d},{cx(r) + d}M{cx(c) - d},'
                       f'{cx(r) + d}L{cx(c) + d},{cx(r) - d}" stroke="#888" stroke-width="1.5"/>')
        for a in _top(pol, mode, board):
            r, c = divmod(int(a), n)
            if board[r, c] != 0:
                continue
            t = min(max((v[a] - lo) / (hi - lo), 0.0), 1.0)
            rgb = to_rgb(cmap(t))
            col = "#fff" if 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] < 0.5 else "#000"
            out.append(f'<text x="{cx(c)}" y="{cx(r) + 3.5}" font-size="9.5" font-weight="bold" '
                       f'fill="{col}" text-anchor="middle">{lab(a)}</text>')
    if ring is not None and ring < n2:
        r, c = divmod(ring, n)
        out.append(f'<circle cx="{cx(c)}" cy="{cx(r)}" r="{cell * 0.44}" fill="none" '
                   f'stroke="#1565c0" stroke-width="2.5"/>')
    # legend: gradient bar + range, then one summary line
    y = W + 6
    gx, gw = pad + cell, W - 2 * (pad + cell)
    stops = "".join(f'<stop offset="{k / 10}" stop-color="{to_hex(cmap(k / 10))}"/>' for k in range(11))
    out.append(f'<defs><linearGradient id="g">{stops}</linearGradient></defs>'
               f'<rect x="{gx}" y="{y}" width="{gw}" height="9" fill="url(#g)" stroke="#999" stroke-width="0.5"/>')
    if pol is not None:
        fmt = {"prob": lambda x: f"{100 * x:.1f}%", "log prob": lambda x: f"10^{x:.0f}",
               "logit": lambda x: f"{x:.1f}"}[mode]
        out.append(f'<text x="{gx}" y="{y + 21}" font-size="10" fill="#333">{fmt(lo)}</text>')
        out.append(f'<text x="{gx + gw}" y="{y + 21}" font-size="10" fill="#333" '
                   f'text-anchor="end">{fmt(hi)}</text>')
        out.append(f'<text x="{gx + gw / 2}" y="{y + 21}" font-size="10" fill="#333" '
                   f'text-anchor="middle">{mode}</text>')
        out.append(f'<text x="{pad}" y="{y + 36}" font-size="10.5" fill="#333">{_summary(pol, mode)}</text>')
    else:
        out.append(f'<text x="{pad}" y="{y + 30}" font-size="11" fill="#555">{note or "no network output"}</text>')
    out.append("</svg>")
    return "".join(out)


# ----------------------------------------------------------------------------- click game
class PolicyClickGame(ClickGame):
    """``ClickGame`` with the network's output drawn on a second board to its left.

    "Show": *net's last move* = the distribution the network had when it chose its last move
    (drawn on the position it saw, blue ring on the move it played); *your move now* = what the
    same network would play in your current position."""

    def __init__(self, opponent, human_color: int = BLACK, cfg: GameConfig | None = None,
                 temperature: float = 0.0, seed: int = 0, mode: str = "prob"):
        import ipywidgets as w
        self.heat = w.HTML()
        self.ai_pol = None  # (board the net saw, policy, move it played)
        self.mode_dd = w.Dropdown(options=MODES, value=mode, description="Show",
                                  layout=w.Layout(width="190px"))
        self.which_dd = w.Dropdown(options=["net's last move", "your move now"], value="net's last move",
                                   description="for", layout=w.Layout(width="230px"))
        super().__init__(opponent, human_color, cfg, temperature, seed)
        self.mode_dd.observe(lambda _: self._render(), names="value")
        self.which_dd.observe(lambda _: self._render(), names="value")

    def reset(self):
        self.ai_pol = None
        super().reset()

    def _play(self, a):
        if self.game.to_play != self.me and not self._over():
            legal, allowed = self._allowed()
            pol = policy(self.ai, self.game, allowed, legal, self.gc.max_moves)
            self.ai_pol = (self.game.board.copy(), pol, a)
        super()._play(a)

    def _render(self):
        super()._render()
        if not hasattr(self, "mode_dd"):
            return
        mode = self.mode_dd.value
        if self.which_dd.value == "your move now":
            if self._over() or self.game.to_play != self.me:
                svg = policy_svg(self.game.board, None, mode, note="game over")
            else:
                legal, allowed = self._allowed()
                pol = policy(self.ai, self.game, allowed, legal, self.gc.max_moves)
                svg = policy_svg(self.game.board, pol, mode,
                                 note=f"{self.ai.name} has no network output")
            head = f"<b>{self.ai.name}</b>'s output for <b>your</b> position"
        else:
            if self.ai_pol is None:
                svg = policy_svg(self.game.board, None, mode, note="the network has not moved yet")
            else:
                b, pol, a = self.ai_pol
                svg = policy_svg(b, pol, mode, ring=a, note=f"{self.ai.name} has no network output")
            mv = f" ({self._name(self.ai_pol[2])}, blue ring)" if self.ai_pol else ""
            head = f"<b>{self.ai.name}</b>'s output for its last move{mv}"
        self.heat.value = f"<div style='font:13px sans-serif;margin:0 0 4px 4px'>{head}</div>{svg}"

    def show(self):
        from IPython.display import display
        w = self.w
        right = w.VBox([self.status, self.grid,
                        w.HBox([self.pass_btn, self.end_btn, self.new_btn, self.area_box])])
        left = w.VBox([w.HBox([self.mode_dd, self.which_dd]), self.heat],
                      layout=w.Layout(margin="0 16px 0 0"))
        display(w.VBox([w.HTML(CSS), w.HBox([left, right], layout=w.Layout(align_items="flex-start"))]))
        return self



# ----------------------------------------------------------------------------- recorded games
class _Recorder:
    """Wraps a player: plays exactly as it would, and logs its output before every move."""

    def __init__(self, player, log: dict):
        self._p, self._log = player, log
        self.name = player.name
        if hasattr(player, "cfg"):
            self.cfg = player.cfg

    def select_move(self, game, allowed, legal, max_moves, temperature=0.0, rng=None):
        self._log[game.move_number] = policy(self._p, game, allowed, legal, max_moves)
        return self._p.select_move(game, allowed, legal, max_moves, temperature, rng)


def play_game_with_policy(black, white, cfg: GameConfig = GameConfig(),
                          rng: np.random.Generator | None = None, on_move=None) -> GameRecord:
    """``play_game`` plus ``rec.policy[i]``: the output of the player who made move i (None for
    the random mover and you), on ``rec.boards[i]``. The moves are the same as without it."""
    log: dict = {}
    rec = play_game(_Recorder(black, log), _Recorder(white, log), cfg, rng,
                    on_move=(lambda g, a, r: on_move(g, a, r, log.get(len(r.moves) - 1))) if on_move else None)
    rec.policy = [log.get(i) for i in range(len(rec.moves))]
    return rec


def draw_policy(board: np.ndarray, pol: dict | None, ax=None, mode: str = "prob",
                ring: int | None = None, title: str = ""):
    """Matplotlib version of the heatmap board (same orientation as ``viz.draw_board``)."""
    import matplotlib
    import matplotlib.pyplot as plt

    n = board.shape[0]
    n2 = n * n
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.clear()
    ax.set_facecolor("#f7f4ec")
    cmap = matplotlib.colormaps[CMAP]
    if pol is not None:
        v, lo, hi, lab = _values(pol, mode)
        img = v[:n2].reshape(n, n).astype(float)
        if mode != "logit":
            img = np.where(pol["allowed"][:n2].reshape(n, n), img, np.nan)
        ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, extent=(-0.5, n - 0.5, -0.5, n - 0.5),
                  origin="upper", interpolation="nearest", zorder=0)
    for i in range(n):
        ax.plot([0, n - 1], [i, i], color="k", lw=0.5, alpha=0.2, zorder=1)
        ax.plot([i, i], [0, n - 1], color="k", lw=0.5, alpha=0.2, zorder=1)
    s = 3000 / n * 3 * 0.3
    for v_, col in ((BLACK, "black"), (WHITE, "white")):
        rs, cs = np.nonzero(board == v_)
        ax.scatter(cs, n - 1 - rs, s=s, c=col, edgecolors="#333", linewidths=0.8, zorder=2)
    if pol is not None:
        bad = (board.ravel() == 0) & ~pol["allowed"][:n2]
        rs, cs = np.divmod(np.flatnonzero(bad), n)
        ax.scatter(cs, n - 1 - rs, s=s * 0.5, marker="x", c="#888", linewidths=1, zorder=2)
        for a in _top(pol, mode, board):
            r, c = divmod(int(a), n)
            if board[r, c] != 0:
                continue
            t = min(max((v[a] - lo) / (hi - lo), 0.0), 1.0)
            rgb = cmap(t)[:3]
            col = "white" if 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] < 0.5 else "black"
            ax.text(c, n - 1 - r, lab(a), ha="center", va="center", fontsize=6, fontweight="bold",
                    color=col, zorder=3)
        fmt = {"prob": lambda x: f"{100 * x:.1f}%", "log prob": lambda x: f"1e{x:.0f}",
               "logit": lambda x: f"{x:.1f}"}[mode]
        sub = f"{mode} {fmt(lo)} … {fmt(hi)}\n{_summary(pol, mode)}"
    else:
        sub = "no network output (random mover or you)"
    if ring is not None and ring < n2:
        r, c = divmod(ring, n)
        ax.scatter([c], [n - 1 - r], s=s * 2.6, facecolors="none", edgecolors="#1565c0",
                   linewidths=2, zorder=4)
    ax.set_xticks(range(n))
    ax.set_xticklabels(list(COLS[:n]))
    ax.set_yticks(range(n))
    ax.set_yticklabels([str(i + 1) for i in range(n)])
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_aspect("equal")
    ax.set_title(f"{title}\n{sub}" if title else sub, fontsize=8)
    return ax


def _pair(rec: GameRecord, i: int, axl, axr, mode: str, score: bool):
    """Frame for move i (1-based): left = output that chose move i, right = board after it."""
    from .viz import draw_board, _frame_title

    a = rec.moves[i - 1]
    who = "B" if (i - 1) % 2 == 0 else "W"
    name = rec.black if who == "B" else rec.white
    pol = rec.policy[i - 1] if getattr(rec, "policy", None) else None
    draw_policy(rec.boards[i - 1], pol, axl, mode, ring=a,
                title=f"{name} ({who}) before move {i}: played {rec.move_str(a)}")
    draw_board(rec.boards[i], axr, a, _frame_title(rec, i), score=score, komi=rec.komi)


def policy_slider(rec: GameRecord, mode: str = "prob", score: bool = True):
    """ipywidgets slider: heatmap of the mover's output (left) and the board after the move (right)."""
    import matplotlib.pyplot as plt
    import ipywidgets as w

    def show(i, mode):
        fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 5.9), layout="constrained")
        _pair(rec, i, axl, axr, mode, score)
        plt.show()

    return w.interact(show, i=w.IntSlider(min=1, max=len(rec.moves), step=1, value=1, description="move",
                                          layout=w.Layout(width="600px")),
                      mode=w.Dropdown(options=MODES, value=mode, description="show"))


def animate_policy(rec: GameRecord, interval_ms: int = 300, every: int = 1, dpi: int = 60,
                   mode: str = "prob", score: bool = True):
    """HTML animation (play / pause / step) of heatmap | board pairs, every k-th move."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from IPython.display import HTML

    matplotlib.rcParams["animation.embed_limit"] = 400
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 5.9), dpi=dpi, layout="constrained")
    frames = list(range(1, len(rec.moves) + 1, every))
    if frames[-1] != len(rec.moves):
        frames.append(len(rec.moves))

    def update(i):
        _pair(rec, i, axl, axr, mode, score)

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=interval_ms)
    html = HTML(anim.to_jshtml())
    plt.close(fig)
    return html


def watch_live_policy(black, white, cfg: GameConfig = GameConfig(), delay: float = 0.3,
                      rng: np.random.Generator | None = None, mode: str = "prob",
                      score: bool = True) -> GameRecord:
    """Play a new game; after every move redraw the mover's output (left) and the board (right)."""
    import time
    import matplotlib.pyplot as plt
    from IPython.display import clear_output, display

    def on_move(game, a, rec, pol):
        from .viz import draw_board
        i = len(rec.moves)
        before = rec.boards[-1]
        clear_output(wait=True)
        fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 5.9), layout="constrained")
        who = "B" if (i - 1) % 2 == 0 else "W"
        name = rec.black if who == "B" else rec.white
        draw_policy(before, pol, axl, mode, ring=a, title=f"{name} ({who}) before move {i}: played {rec.move_str(a)}")
        draw_board(game.board, axr, a, f"{rec.black} (B) vs {rec.white} (W) — move {i}: {who} {rec.move_str(a)}",
                   score=score, komi=rec.komi)
        display(fig)
        plt.close(fig)
        time.sleep(delay)

    rec = play_game_with_policy(black, white, cfg, rng, on_move=on_move)
    print(f"Result: {rec.result_string} ({len(rec.moves)} moves)")
    return rec

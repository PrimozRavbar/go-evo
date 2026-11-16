"""Genome-encoded Go players.

A *genome* is a flat float32 vector holding every weight and bias of a fixed
architecture (``NetConfig.arch``):

* ``"mlp"``  - ``num_layers`` Linear layers (default 10, hidden 128), phases 1-2c.
* ``"lowrank"`` - MLP (``num_layers`` Linear layers, width ``hidden``) where every weight
  matrix is a pure low-rank product W = A @ B, A (out x rank), B (rank x in). The genome
  holds only A, B and the biases (phase 3, parameter-efficient evolution).
* ``"conv"`` - ``num_layers`` 3x3 convolutions with ``channels`` channels (zero padding),
  then a 1x1 convolution giving one logit per point and a linear pass logit computed
  from the globally averaged features. Inputs are ``CONV_PLANES`` feature planes,
  including liberty counts. The weights are shared over the board, so the genome
  size does not depend on the board size.

Inference runs in numpy (much faster than torch at batch size 1); ``Player.net``
builds the equivalent torch module for checks and export.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn

from .engine import EMPTY, GoGame

NUM_FEATURE_PLANES = 4  # MLP: own stones, opponent stones, empty, legal-for-me
# conv: own, opponent, empty, legal, own group with 1 / 2 / >=3 liberties,
#       opponent group with 1 / 2 / >=3 liberties, constant ones (shows the board edge)
CONV_PLANES = 11


@dataclass
class NetConfig:
    board_size: int = 19
    num_layers: int = 10       # MLP: Linear layers; conv: 3x3 conv layers
    hidden: int = 128          # MLP: width of every hidden layer
    activation: str = "tanh"   # tanh | relu | gelu
    arch: str = "mlp"          # mlp | lowrank | conv
    channels: int = 16         # conv: channels of every conv layer
    rank: int = 32             # lowrank: rank r of every W = A @ B

    def param_blocks(self) -> list[tuple[tuple[int, ...], float]]:
        """[(shape, init std)] of every parameter tensor, in genome order."""
        gain = _GAIN[self.activation]
        if self.arch == "lowrank":
            out = []
            for i, o in self.layer_dims():
                s = (gain ** 2 / (i * self.rank)) ** 0.25  # var(AB entries) = gain^2 / fan_in
                out += [((o, self.rank), s), ((self.rank, i), s), ((o,), 0.1)]
            return out
        out = []
        for wshape, nb, fan_in in self.param_shapes():
            out += [(wshape, gain / np.sqrt(fan_in)), ((nb,), 0.1)]
        return out

    def param_shapes(self) -> list[tuple[tuple[int, ...], int, int]]:
        """[(weight shape, bias size, fan_in)] in genome order."""
        if self.arch == "conv":
            C = self.channels
            out = [((C, CONV_PLANES, 3, 3), C, CONV_PLANES * 9)]
            out += [((C, C, 3, 3), C, C * 9)] * (self.num_layers - 1)
            return out + [((1, C, 1, 1), 1, C), ((1, C), 1, C)]  # point head, pass head
        return [((o, i), o, i) for i, o in self.layer_dims()]

    @property
    def in_dim(self) -> int:
        return NUM_FEATURE_PLANES * self.board_size ** 2 + 1  # +1: "move progress" scalar

    @property
    def out_dim(self) -> int:
        return self.board_size ** 2 + 1  # every point + pass

    def layer_dims(self) -> list[tuple[int, int]]:
        assert self.num_layers >= 2
        dims = [self.in_dim] + [self.hidden] * (self.num_layers - 1) + [self.out_dim]
        return list(zip(dims[:-1], dims[1:]))

    @property
    def genome_size(self) -> int:
        return sum(int(np.prod(shape)) for shape, _ in self.param_blocks())


_ACT = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}
_GAIN = {"tanh": 5.0 / 3.0, "relu": 2 ** 0.5, "gelu": 2 ** 0.5}


class ConvNet(nn.Module):
    """Torch reference of the conv architecture (input (B, CONV_PLANES, N, N))."""

    def __init__(self, cfg: NetConfig):
        super().__init__()
        C = cfg.channels
        self.convs = nn.ModuleList([nn.Conv2d(CONV_PLANES if k == 0 else C, C, 3, padding=1)
                                    for k in range(cfg.num_layers)])
        self.point = nn.Conv2d(C, 1, 1)
        self.pass_ = nn.Linear(C, 1)
        self.act = _ACT[cfg.activation]()

    def forward(self, x):
        for conv in self.convs:
            x = self.act(conv(x))
        pts = self.point(x).flatten(1)
        return torch.cat([pts, self.pass_(x.mean((2, 3)))], 1)


class LowRankLinear(nn.Module):
    """y = x @ (A @ B).T + bias; parameters registered in genome order A, B, bias."""

    def __init__(self, i, o, r):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(o, r))
        self.B = nn.Parameter(torch.zeros(r, i))
        self.bias = nn.Parameter(torch.zeros(o))

    def forward(self, x):
        return (x @ self.B.T) @ self.A.T + self.bias


def build_net(cfg: NetConfig) -> nn.Module:
    if cfg.arch == "conv":
        return ConvNet(cfg)
    if cfg.arch == "lowrank":
        dims = cfg.layer_dims()
        layers = []
        for k, (i, o) in enumerate(dims):
            layers.append(LowRankLinear(i, o, cfg.rank))
            if k < len(dims) - 1:
                layers.append(_ACT[cfg.activation]())
        return nn.Sequential(*layers)
    layers: list[nn.Module] = []
    dims = cfg.layer_dims()
    for k, (i, o) in enumerate(dims):
        layers.append(nn.Linear(i, o))
        if k < len(dims) - 1:
            layers.append(_ACT[cfg.activation]())
    return nn.Sequential(*layers)


def random_genome(cfg: NetConfig, rng: np.random.Generator) -> np.ndarray:
    """Variance-preserving random init (so signals survive many layers)."""
    parts = [rng.normal(0.0, std, size=shape).ravel() for shape, std in cfg.param_blocks()]
    return np.concatenate(parts).astype(np.float32)


def encode_state(game: GoGame, legal: np.ndarray, max_moves: int) -> np.ndarray:
    """Features from the perspective of the player to move."""
    b = game.board
    me = game.to_play
    planes = [
        (b == me),
        (b == -me),
        (b == EMPTY),
        legal[:-1].reshape(b.shape),
    ]
    x = np.concatenate([p.astype(np.float32).ravel() for p in planes])
    progress = np.float32(min(game.move_number / max(max_moves, 1), 1.0))
    return np.append(x, progress)


def encode_planes(game: GoGame, legal: np.ndarray) -> np.ndarray:
    """(CONV_PLANES, N, N) float32 features for the conv net, from the mover's view."""
    b = game.board
    me = game.to_play
    libs = game.liberty_map()
    mine, theirs = b == me, b == -me
    x = np.empty((CONV_PLANES,) + b.shape, dtype=np.float32)
    x[0], x[1], x[2] = mine, theirs, b == EMPTY
    x[3] = legal[:-1].reshape(b.shape)
    for k, own in ((4, mine), (7, theirs)):
        x[k] = own & (libs == 1)
        x[k + 1] = own & (libs == 2)
        x[k + 2] = own & (libs >= 3)
    x[10] = 1.0
    return x


def _conv3x3(x, W2, b, n):
    """x (Cin, n, n), W2 (Cout, Cin*9) -> (Cout, n, n); zero padding."""
    xp = np.zeros((x.shape[0], n + 2, n + 2), dtype=np.float32)
    xp[:, 1:-1, 1:-1] = x
    cols = np.stack([xp[:, dy:dy + n, dx:dx + n] for dy in range(3) for dx in range(3)], axis=1)
    return (W2 @ cols.reshape(-1, n * n) + b[:, None]).reshape(-1, n, n)


class Player:
    """Phenotype: a genome loaded into a network (MLP or conv) that picks moves."""

    def __init__(self, genome: np.ndarray, cfg: NetConfig, name: str = "player",
                 device: str = "cpu"):
        assert genome.size == cfg.genome_size, (genome.size, cfg.genome_size)
        self.genome = genome
        self.cfg = cfg
        self.name = name
        self.device = device
        self._net = None  # torch module, built lazily (only needed for logits_torch / export)
        # numpy copies of the weights: for batch-1 inference numpy is much faster than torch
        self._layers = []
        off = 0
        if cfg.arch == "lowrank":
            blocks = []
            for shape, _ in cfg.param_blocks():
                nw = int(np.prod(shape))
                blocks.append(genome[off:off + nw].reshape(shape))
                off += nw
            for A, B, b in zip(blocks[0::3], blocks[1::3], blocks[2::3]):
                self._layers.append((B.T.copy(), A.T.copy(), b))  # x @ B.T @ A.T + b
        for wshape, nb, _ in (cfg.param_shapes() if cfg.arch != "lowrank" else []):
            nw = int(np.prod(wshape))
            W = genome[off:off + nw].reshape(wshape)
            off += nw
            b = genome[off:off + nb]
            off += nb
            if cfg.arch == "conv":
                W = W.reshape(wshape[0], -1).copy()  # (Cout, Cin*k*k), matches _conv3x3's columns
            else:
                W = W.T.copy()
            self._layers.append((W, b))
        self._act = {"tanh": np.tanh, "relu": lambda z: np.maximum(z, 0),
                     "gelu": lambda z: 0.5 * z * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))}[cfg.activation]

    def logits(self, features: np.ndarray) -> np.ndarray:
        x = features.astype(np.float32, copy=False)
        if self.cfg.arch == "conv":
            n = self.cfg.board_size
            for W, b in self._layers[:-2]:
                x = self._act(_conv3x3(x, W, b, n))
            (Wp, bp), (Wq, bq) = self._layers[-2:]
            flat = x.reshape(x.shape[0], -1)
            return np.concatenate([(Wp @ flat)[0] + bp, Wq @ flat.mean(1) + bq])
        last = len(self._layers) - 1
        if self.cfg.arch == "lowrank":
            for k, (Bt, At, b) in enumerate(self._layers):
                x = (x @ Bt) @ At + b
                if k < last:
                    x = self._act(x)
            return x
        for k, (W, b) in enumerate(self._layers):
            x = x @ W + b
            if k < last:
                x = self._act(x)
        return x

    @property
    def net(self) -> nn.Sequential:
        if self._net is None:
            self._net = build_net(self.cfg).to(self.device).eval()
            torch.nn.utils.vector_to_parameters(
                torch.as_tensor(self.genome, dtype=torch.float32, device=self.device),
                self._net.parameters())
        return self._net

    @torch.no_grad()
    def logits_torch(self, features: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.ascontiguousarray(features), device=self.device).unsqueeze(0)
        return self.net(x)[0].cpu().numpy()

    def select_move(self, game: GoGame, allowed: np.ndarray, legal: np.ndarray,
                    max_moves: int, temperature: float = 0.0,
                    rng: np.random.Generator | None = None) -> int:
        """Masked argmax (temperature 0) or softmax sampling over ``allowed``."""
        feats = encode_planes(game, legal) if self.cfg.arch == "conv" else encode_state(game, legal, max_moves)
        z = self.logits(feats).astype(np.float64)
        z[~allowed] = -np.inf
        if temperature <= 0:
            return int(np.argmax(z))
        rng = rng or np.random.default_rng()
        p = np.exp((z - z[allowed].max()) / temperature)
        p /= p.sum()
        return int(rng.choice(len(p), p=p))

    def __repr__(self):
        return f"Player({self.name!r}, {self.cfg.arch} {self.cfg.num_layers} layers, {self.genome.size} params)"


def layer_scales(cfg: NetConfig) -> np.ndarray:
    """Per-parameter init std (same layout as the genome) - used to scale mutations."""
    return np.concatenate([np.full(int(np.prod(shape)), std, np.float32)
                           for shape, std in cfg.param_blocks()])


def mutate(genome: np.ndarray, cfg: NetConfig, sigma: float, rng: np.random.Generator,
           scales: np.ndarray | None = None) -> np.ndarray:
    """Gaussian mutation; noise std = sigma * (init std of that layer). No crossover."""
    if scales is None:
        scales = layer_scales(cfg)
    noise = rng.standard_normal(genome.size, dtype=np.float32)
    return (genome + sigma * scales * noise).astype(np.float32)


class RandomPlayer:
    """Baseline: uniform random over the allowed moves (same rules as the networks)."""

    def __init__(self, name: str = "random"):
        self.name = name

    def select_move(self, game, allowed, legal, max_moves, temperature=0.0, rng=None):
        rng = rng or np.random.default_rng()
        return int(rng.choice(np.flatnonzero(allowed)))


def save_population(path: str, genomes: list[np.ndarray], cfg: NetConfig, names: list[str],
                    extra: dict | None = None) -> None:
    torch.save({"cfg": asdict(cfg), "names": names,
                "genomes": [torch.from_numpy(g) for g in genomes], "extra": extra or {}}, path)


def load_population(path: str):
    d = torch.load(path, weights_only=False)
    cfg = NetConfig(**d["cfg"])
    genomes = [g.numpy() for g in d["genomes"]]
    return genomes, cfg, d["names"], d.get("extra", {})

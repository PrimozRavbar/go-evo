"""Genome-encoded MLP Go players.

A *genome* is a flat float32 vector holding every weight and bias of a fixed
MLP architecture. The phenotype (``Player``) is a PyTorch ``nn.Sequential``
with ``num_layers`` Linear layers (default 10) whose parameters are loaded from
the genome. This flat representation makes mutation / crossover trivial in the
evolutionary phase.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn

from .engine import EMPTY, GoGame

NUM_FEATURE_PLANES = 4  # own stones, opponent stones, empty, legal-for-me


@dataclass
class NetConfig:
    board_size: int = 9
    num_layers: int = 10       # number of Linear layers (input + hidden + output)
    hidden: int = 128          # width of every hidden layer
    activation: str = "tanh"   # tanh | relu | gelu

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
        return sum(i * o + o for i, o in self.layer_dims())


_ACT = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}
_GAIN = {"tanh": 5.0 / 3.0, "relu": 2 ** 0.5, "gelu": 2 ** 0.5}


def build_net(cfg: NetConfig) -> nn.Sequential:
    layers: list[nn.Module] = []
    dims = cfg.layer_dims()
    for k, (i, o) in enumerate(dims):
        layers.append(nn.Linear(i, o))
        if k < len(dims) - 1:
            layers.append(_ACT[cfg.activation]())
    return nn.Sequential(*layers)


def random_genome(cfg: NetConfig, rng: np.random.Generator) -> np.ndarray:
    """Variance-preserving random init (so signals survive 10 layers)."""
    parts = []
    gain = _GAIN[cfg.activation]
    for i, o in cfg.layer_dims():
        w = rng.normal(0.0, gain / np.sqrt(i), size=(o, i))
        b = rng.normal(0.0, 0.1, size=o)
        parts += [w.ravel(), b]
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


class Player:
    """Phenotype: a genome loaded into an MLP that picks moves."""

    def __init__(self, genome: np.ndarray, cfg: NetConfig, name: str = "player",
                 device: str = "cpu"):
        assert genome.size == cfg.genome_size, (genome.size, cfg.genome_size)
        self.genome = genome
        self.cfg = cfg
        self.name = name
        self.device = device
        self.net = build_net(cfg).to(device).eval()
        torch.nn.utils.vector_to_parameters(
            torch.as_tensor(genome, dtype=torch.float32, device=device),
            self.net.parameters())

    @torch.no_grad()
    def logits(self, features: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(features, device=self.device).unsqueeze(0)
        return self.net(x)[0].cpu().numpy()

    def select_move(self, game: GoGame, allowed: np.ndarray, legal: np.ndarray,
                    max_moves: int, temperature: float = 0.0,
                    rng: np.random.Generator | None = None) -> int:
        """Masked argmax (temperature 0) or softmax sampling over ``allowed``."""
        z = self.logits(encode_state(game, legal, max_moves)).astype(np.float64)
        z[~allowed] = -np.inf
        if temperature <= 0:
            return int(np.argmax(z))
        rng = rng or np.random.default_rng()
        p = np.exp((z - z[allowed].max()) / temperature)
        p /= p.sum()
        return int(rng.choice(len(p), p=p))

    def __repr__(self):
        return f"Player({self.name!r}, {self.cfg.num_layers} layers, {self.genome.size} params)"


def save_population(path: str, genomes: list[np.ndarray], cfg: NetConfig, names: list[str],
                    extra: dict | None = None) -> None:
    torch.save({"cfg": asdict(cfg), "names": names,
                "genomes": [torch.from_numpy(g) for g in genomes], "extra": extra or {}}, path)


def load_population(path: str):
    d = torch.load(path, weights_only=False)
    cfg = NetConfig(**d["cfg"])
    genomes = [g.numpy() for g in d["genomes"]]
    return genomes, cfg, d["names"], d.get("extra", {})

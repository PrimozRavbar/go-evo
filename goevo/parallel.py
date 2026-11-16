"""Run many games in parallel worker processes.

A *player spec* is either ``("genome", name, genome_array)`` or ``("random", name)``;
specs are cheap to pickle and are turned back into players inside the worker.
"""
from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np

from .match import GameConfig, GameRecord, play_game
from .player import NetConfig, Player, RandomPlayer


def spec_of(p) -> tuple:
    if isinstance(p, RandomPlayer):
        return ("random", p.name)
    return ("genome", p.name, p.genome)


def _build(spec, net_cfg):
    if spec[0] == "random":
        return RandomPlayer(spec[1])
    return Player(spec[2], spec[3] if len(spec) > 3 else net_cfg, name=spec[1])  # optional own cfg


def _worker_init():
    import torch
    torch.set_num_threads(1)


def _play(job):
    black, white, net_cfg, game_cfg, seed, keep_boards = job
    rec = play_game(_build(black, net_cfg), _build(white, net_cfg), game_cfg,
                    np.random.default_rng(seed))
    if not keep_boards:
        rec.boards = []  # save memory/IPC; moves are enough to replay
    return rec


def default_workers() -> int:
    return max(1, (os.cpu_count() or 1))


class GameRunner:
    """Persistent process pool (``workers=1`` runs in-process)."""

    def __init__(self, workers: int | None = None):
        self.workers = workers or default_workers()
        self.pool = None
        _worker_init()  # batch-1 MLPs: extra torch threads only cause contention
        if self.workers > 1:
            ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
            self.pool = ctx.Pool(self.workers, initializer=_worker_init)

    def play(self, pairs, net_cfg: NetConfig, game_cfg: GameConfig, seed: int = 0,
             keep_boards: bool = False) -> list[GameRecord]:
        """pairs: list of (black_spec, white_spec)."""
        jobs = [(b, w, net_cfg, game_cfg, seed * 100003 + k, keep_boards)
                for k, (b, w) in enumerate(pairs)]
        if self.pool is None:
            return [_play(j) for j in jobs]
        return self.pool.map(_play, jobs, chunksize=1)

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

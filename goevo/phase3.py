"""Phase 3 CLI: parameter-efficient evolution.

20-layer MLP (width 128, tanh) where every weight matrix is a pure low-rank product
W = A @ B (rank 32 by default, ~216k params). Selection as in phase 2d: 32 genomes,
knockout + top-4 qualifier vs the hall of fame, 4 parents, sigma 0.03. No benchmark
during the run; parents checkpointed every 1000 generations; resumable state.pt.

    python -m goevo.phase3 --out work/p3_r32_s0
    python -m goevo.phase3 --out work/p3_r32_s0 --resume --generations 20000   # continue

Any phase-2 flag can be given and overrides these defaults (e.g. --rank 8).
"""
import sys

from .phase2 import main as phase2_main

DEFAULTS = ["--arch", "lowrank", "--layers", "20", "--rank", "32", "--generations", "10000",
            "--bench-every", "0", "--checkpoint-every", "1000", "--out", "work/p3_r32_s0"]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    return phase2_main(DEFAULTS + list(argv))  # later flags override earlier ones


if __name__ == "__main__":
    main()

"""Phase 4 CLI: residual CNN with normalization.

Conv stem (11 input planes -> 16 channels), 4 pre-activation residual blocks
(GroupNorm(1) -> tanh -> 3x3 conv -> GroupNorm(1) -> tanh -> 3x3 conv, added to the input),
final norm + tanh, point / pass heads; ~20.5k params. Selection as in phase 2d (32 genomes,
knockout + top-4 qualifier vs the hall of fame, 4 parents, sigma 0.03). Light tracking:
random-mover check every 500 generations, parents every 1000, resumable state.pt every 25.

    python -m goevo.phase4 --out work/p4_s0
    python -m goevo.phase4 --out work/p4_s0 --resume      # continue after an interruption

Any phase-2 flag can be given and overrides these defaults.
"""
import sys

from .phase2 import main as phase2_main

DEFAULTS = ["--arch", "resnet", "--channels", "16", "--blocks", "4", "--generations", "4000",
            "--bench-every", "500", "--bench-rm-games", "24", "--checkpoint-every", "1000",
            "--out", "work/p4_s0"]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    return phase2_main(DEFAULTS + list(argv))


if __name__ == "__main__":
    main()

# Phase 2 results: is evolution happening?

**Short answer: yes.** Selection produces clearly stronger players than the no-selection control.
But the improvement levels off after about 100 generations, and the evolved players still lose to a random-move player.

## Setup
* Board: 19×19, komi 7.5, move cap 1083 (3·N²).
* Players: 10-layer MLP, tanh, 128 hidden units, 363,882 parameters.
* Generations: population of 10, each genome plays 5 games, 300 generations.
* Reproduction: mutation only, σ = 0.1 × each layer's init std, no crossover.
* **Selection runs** (`sel_s0`, `sel_s1`): only *winners* (more wins than losses in their 5 games) reproduce.
  Winners survive unchanged, and mutated children of the winners fill the rest of the population.
* **Control runs** (`ctl_s0`, `ctl_s1`): exactly the same, but the "winners" are picked at random (the same number as the real selection would pick).
  Any difference from the selection runs is therefore due to selection.
* Checkpoints every 50 generations (`runs/*/gen*.pt`). Full per-generation logs in `runs/*/history.jsonl`.

## Evidence

![curves](curves.png)

### 1. Fixed benchmark
Every 10 generations, each genome plays both colours against 5 fixed random genomes and a uniform-random mover.

| run | gen 0 | mean gens 100–300 | vs random genomes | vs random mover |
|---|---|---|---|---|
| sel_s0 | 0.475 | **0.734** | 0.807 | 0.367 |
| sel_s1 | 0.475 | **0.819** | 0.898 | 0.429 |
| ctl_s0 | 0.475 | 0.602 | 0.670 | 0.260 |
| ctl_s1 | 0.475 | 0.549 | 0.617 | 0.210 |

### 2. Head-to-head
All 10×10 pairings, both colours, 200 games each (`head_to_head.txt`):

| match | win rate of first |
|---|---|
| sel_s0 gen300 vs ctl_s0 gen300 | **0.680** |
| sel_s0 gen300 vs ctl_s1 gen300 | **0.735** |
| sel_s1 gen300 vs ctl_s0 gen300 | **0.830** |
| sel_s1 gen300 vs ctl_s1 gen300 | **0.735** |
| sel_s0 gen300 vs own gen0 | **0.805** |
| sel_s1 gen300 vs own gen0 | **0.780** |
| ctl_s0 gen300 vs own gen0 | 0.605 |
| ctl_s1 gen300 vs own gen0 | 0.540 |
| sel_s0 gen300 vs own gen150 | 0.555 |
| sel_s1 gen300 vs own gen150 | 0.435 |
| sel_s0 gen300 vs sel_s1 gen300 | 0.485 |

With 200 games, one standard error is about ±0.035. The selection-vs-control and selection-vs-gen0 differences are far outside that.

### 3. Behaviour changed
Games between selected players got longer (about 450 moves, against about 350 in the controls).
The players seem to have learned to avoid playing into self-atari or capture, so they keep more stones on the board.

## Caveats and what to do next
* **Plateau.** The real gain happens in the first ~50–100 generations. Generation 300 is not better than generation 150 (0.555 and 0.435).
  This is typical for a population of 10, 5 noisy games per genome and pure relative fitness.
* **Weak survivor advantage.** Last generation's winners beat this generation's newborn children only slightly more often (+0.01 to +0.02, against about 0 in the controls).
  Five games per genome is a very noisy measure of fitness.
* **Still worse than random play.** The evolved players lose to the uniform-random mover (37–43% win rate).
  Better than the controls (21–26%), but they are still very weak Go players.
* **Controls drift above 0.5 against the random-genome panel too.** Possibly because elitism plus mutation favours certain weight scales.
  That's why the controls, not 0.5, are the right baseline.

Ideas for phase 3:
* bigger population or more games per genome (less noisy fitness);
* hall-of-fame / ancestor opponents to avoid cycling;
* include the random mover in the fitness;
* smaller or adaptive σ;
* convolutional or weight-shared input layer;
* save checkpoints in float16 to keep the repo small.

---

# Phase 2b: stronger selection pressure

## What changed (compared with phase 2a)
| | phase 2a | phase 2b |
|---|---|---|
| population | 10 | **32** |
| tournament | 5 round-robin games per genome (25 games) | **knockout bracket**, 1 game per match (31 games) |
| parents | everyone with a winning record (~5 of 10, ~50%) | **best 4 of 32 (12.5%)** |
| fixed opponents in fitness | none | the bracket's top 8 play 2 games vs the random mover + 2 vs a **hall of fame** of past champions; fitness = rounds won + 1.5 × qualifier win rate |
| mutation σ | 0.1 | **0.03** |
| checkpoints | whole population every 50 generations | **only the 4 final parents** |

Runs: `ko_sel_s0` (selection) and `ko_ctl_s0` (control: same, but the 4 parents are picked at random and no qualifier is played).
Both ran for 300 generations: 87 min and 48 min on 2 CPUs each.

## Results
Head-to-head (`phase2b_eval.txt`). Play is deterministic, so every pairing is played once with each colour:

| match | win rate of first | n |
|---|---|---|
| **2b selection vs 2b control** | **0.906 ± 0.05** | 32 |
| **2b selection vs generation 0** | **0.898 ± 0.02** | 256 |
| 2b control vs generation 0 | 0.621 ± 0.03 | 256 |
| 2a selection vs generation 0 | 0.794 ± 0.01 | 1280 |
| **2b selection vs 2a selection (final)** | **0.569 ± 0.04** | 160 |
| 2b selection vs 2a control (final) | 0.775 ± 0.03 | 160 |
| 2b control vs 2a selection (final) | 0.306 ± 0.04 | 160 |

Against the uniform-random mover (a stochastic opponent; 50 games per genome):

| players | win rate |
|---|---|
| generation 0 | 0.105 ± 0.015 |
| 2a control | 0.202 ± 0.013 |
| 2b control | 0.220 ± 0.029 |
| 2a selection | 0.342 ± 0.015 |
| **2b selection** | **0.450 ± 0.035** |

![curves](curves_2b.png)

## Interpretation
* **Stronger selection pressure works.** At the same number of generations, the 2b players:
  * beat generation 0 90% of the time (2a: 79%);
  * beat the phase-2a champions 57% of the time;
  * win 45% against the random mover (2a: 34%; generation 0: 10%).
* **No plateau yet.** The benchmark of the 2b selection run is still rising at generation 300, while 2a levelled off after about 100 generations.
  The qualifier win rate stays flat at about 0.52 even though the hall-of-fame opponents keep getting stronger, which also points to steady improvement.
* **The control drifts too.** The 2b control beats generation 0 62% of the time: mutation plus random selection still drifts towards "less suicidal" play.
  The selection-vs-control gap is the real effect.
* **Still not better than random play.** The players are almost at 50% against the random mover, but not above it.
* **The "survivor − newborn" panel doesn't apply to the knockout runs.** One elimination game is too coarse for it. Use the head-to-head numbers instead.

## Next steps
* Keep running the 2b selection run, which is still improving. Continue from `runs/ko_sel_s0/gen0300.pt`.
* A second seed, to confirm the result.
* A larger population and bracket (64 genomes), or a double-elimination bracket.

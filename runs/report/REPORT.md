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

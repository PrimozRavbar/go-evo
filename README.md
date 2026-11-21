# go-evo

Evolving Go players. Each player is a genome: the flat weight vector of a 10-layer MLP (PyTorch).
The players play each other, and the results are used as fitness.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PrimozRavbar/go-evo/blob/main/notebooks/phase1_colab.ipynb)

## Phase 1 (done)

* **Go API:** [PettingZoo `classic.go_v5`](https://pettingzoo.farama.org/environments/classic/go/), driven through its standard AEC API (`goevo.engine.PettingZooGo`).
* **Own engine as fallback:** `goevo.engine.NativeGo`, in pure numpy. It is used automatically when PettingZoo can't be imported.
  It has the same rules: area scoring with komi, suicide illegal, simple ko, game ends after two passes.
  `tests/` plays 20 random games on both engines and checks that they agree on every legal-move mask, every board and the final score.
* **Players:** `NetConfig(num_layers=10, hidden=128)` gives 10 Linear layers with tanh: about 364k parameters per genome on 19×19 (about 184k on 9×9).
  * Input (from the side to move): own stones, opponent stones, empty points, legal points, plus game progress.
  * Output: 82 logits (81 points + pass). The move is a masked argmax, or softmax sampling if `temperature > 0`.
* **Tournament:** 10 random genomes, each plays 5 headless games.
  That is 25 games against distinct opponents (round-robin pairings), with colours balanced.
  Fitness = wins + 0.5·draws, with the average score margin as tiebreaker.
* **Watching:** in Colab/Jupyter there are three ways (`goevo.viz`):
  * `animate_game(record)`: animation with play/step controls.
  * `slider_viewer(record)`: scrub through a game with a slider.
  * `watch_live(p1, p2)`: play a new game and redraw the board after every move.

  In a terminal, use `ascii_replay(record)`. Every game can also be exported as SGF.

### Game-play helpers (configurable in `GameConfig`)
Random networks would otherwise end games instantly by passing, or ruin their own groups. So by default:
* passing is only allowed after `N*N//2` moves, or when there is no other move;
* players may not fill their own single-point eyes (`forbid_own_eye_fill`);
* games stop at `3*N*N` moves and are then scored as they stand.

## Run

Colab: open a notebook in `notebooks/` and run all cells. The notebooks clone with `--depth 1`, so old checkpoints in the git history are not downloaded. Do the same when cloning by hand:
```bash
git clone --depth 1 https://github.com/PrimozRavbar/go-evo.git
```

Locally:
```bash
pip install -r requirements.txt
```
```bash
python -m goevo.phase1 --pop 10 --games 5 --watch 0
```
```bash
python -m pytest -q tests
```

## Layout
```
goevo/engine.py      Go engines (PettingZoo wrapper + own NativeGo), scoring
goevo/player.py      NetConfig, genome <-> 10-layer MLP, move selection, save/load
goevo/match.py       play_game -> GameRecord (moves, boards, score, SGF)
goevo/tournament.py  population, pairing schedule, standings/fitness
goevo/viz.py         board drawing, animation, slider, live view, ASCII replay
goevo/phase1.py      CLI for phase 1
goevo/evolve.py      phase 2 evolution loop (winners-only reproduction, mutation)
goevo/parallel.py    multiprocess game runner
goevo/analysis.py    selection-signal analysis (benchmark, survivor advantage, head-to-head)
goevo/phase2.py      CLI for phase 2
runs/                phase 2 runs: checkpoints, histories, report
notebooks/phase1_colab.ipynb
notebooks/phase2_colab.ipynb
tests/test_goevo.py
```

## Phase 2 (done): evolution on 19×19

* The default board is now **19×19** (move cap 3·N² = 1083).
* Each generation, every genome plays 5 games as in phase 1.
* **Only winners** (more wins than losses) have children: Gaussian mutation, **no crossover**. Winners survive (elitism).
* Control runs (`--selection random`) choose parents at random, to check whether selection is really doing anything.
* Fixed benchmark panel (random genomes + uniform-random mover) every 10 generations. Checkpoints every 50 generations.

**Result: there is a selection signal.**
* After 300 generations, the selected populations beat the control populations 68–83% of the time.
* They beat their own generation-0 ancestors about 80% of the time (controls: 54–61%).
* The gain levels off after ~100 generations, and the players still lose to a random mover.

### Phase 2b: stronger selection
* 32 genomes per generation, in a **knockout bracket** (one game per match).
* The top 8 also play a qualifier against the random mover and a **hall of fame** of past champions.
* **Only the best 4 reproduce.** σ = 0.03.
* Only the final 4 parents are checkpointed (about 6 MB per run).

Results after 300 generations:
* the 2b players beat their generation 0 90% of the time;
* they beat the control 91% of the time;
* they beat the phase 2a champions 57% of the time;
* they win 45% against a random mover (2a: 34%);
* they are still improving.

### Phase 2c: faster, bigger, longer
* A compiled (numba) Go engine and numpy inference make games about 7× faster. Tests confirm the moves are identical to before.
* **A player passes only when it has no other legal move.**
* 64 genomes per generation, 2000 generations, knockout bracket, best 4 reproduce, σ = 0.03.
* The qualifier is played against the hall of fame of past champions only. The random mover is used only as a yardstick.
* Runs save `state.pt` every 25 generations and resume with `--resume`. Keep these in a git-ignored `work/` folder.

Results after 2000 generations (selection run vs control run):
* Win rate vs the random mover:
  * selection run: **0.50 ± 0.03**, up from 0.28 for generation 0;
  * control run: 0.39 ± 0.02;
  * 2b champions: 0.44 ± 0.03.
* The selection run's champions beat:
  * the control 75% of the time;
  * generation 0 76% of the time;
  * the 2b champions 63% of the time.
* Still improving slowly, about +0.1 vs the random mover per 1000 generations.

### Phase 2d: convolutional network (current defaults)
* **Network:** 10 3×3 conv layers, 16 channels, about 22.5k weights, independent of board size. Use `--arch mlp` for the old network.
* **Inputs:** stones, empty points, legal moves, liberty counts (1, 2, 3+ for each side), and a plane marking the board edge.
* **Selection:** 32 genomes, knockout; the top 4 play 2 qualifier games each against the hall of fame; the best 4 reproduce.
* **Result after 1000 generations:**
  * **88.5%** win rate vs the random mover (control run 62%, generation 0 38%);
  * 81% vs the phase 2c MLP champions;
  * 94% vs the control run.

### Phase 3: parameter-efficient evolution (branch `claude/phase3-lowrank`)
* **Network:** a 20-layer MLP where every weight matrix is a pure low-rank product W = A·B (rank 32), 216k weights. Use `--arch lowrank --rank r`.
* **Selection:** as in phase 2d. 10,000 generations: `python -m goevo.phase3`.
* **Result:**
  * reaches phase 2c strength (50% vs the random mover) in about 1000 generations;
  * beats the 2c MLP champions 69% of the time;
  * levels off after that, and loses 78% of its games against the 2d conv champions.
* Report: [`runs/report/PHASE3.md`](runs/report/PHASE3.md).

### Phase 4: residual CNN with normalization (branch `claude/phase4-cnn`)
* **Network:** a conv stem, then 4 pre-activation residual blocks with GroupNorm(1) (LayerNorm over C, H, W), 16 channels, 20.5k weights. Use `--arch resnet`.
* **Selection:** as in phase 2d. 4000 generations (about 8.3 h on 4 cores): `python -m goevo.phase4`.
* **Result:**
  * beats the random mover in 400 of 400 games;
  * beats the 2d conv champions 81% of the time and the phase 3 champions 100%.
* Report: [`runs/report/PHASE4.md`](runs/report/PHASE4.md).

### Phase 5: PPO self-play baseline (branch `claude/phase5-rl`)
* **Network:** the same phase 4 network (20.5k weights), plus a tiny value head used only in training. Pytorch on CPU.
* **Training:** self-play against the current policy and a pool of past snapshots. Same budget: 8 h on 4 cores (72.6k games). Command: `python -m goevo.ppo --out work/p5_s0 --hours 8`.
* **Result:**
  * after 1 h it already beats the final phase 4 evolved parents;
  * the final policies win 100% against the phase 4, 2d and 3 champions and the random mover.
* Report: [`runs/report/PHASE5.md`](runs/report/PHASE5.md).

### Phase 5b: bigger PPO network (branch `claude/phase5b-ppo-big`)
* **Network:** 32 channels, 6 blocks (115k weights), with random board rotations and mirrors. 8 h, 16.4k games. Command: `python -m goevo.ppo --channels 32 --blocks 6 --sym`.
* **Result:**
  * beats every evolved player;
  * loses all games to the small PPO, and also learns less per game;
  * needs roughly 20–30 h to catch up.
* Report: [`runs/report/PHASE5B.md`](runs/report/PHASE5B.md).

### Phase 5c: the bigger PPO network, continued at a lower learning rate (branch `claude/phase5c-ppo-lr`)
* Resumes phase 5b for 12 more hours (20 h, 42.4k games in total), with lr 1e-4 instead of 3e-4; nothing else changed. Command: `python -m goevo.ppo --channels 32 --blocks 6 --sym --hours 20 --lr 1e-4 --resume`.
* **Result:**
  * beats the small PPO's final policies 22 of 32 (69%);
  * about even with it per game, but 2.5× more CPU time.
* Report: [`runs/report/PHASE5C.md`](runs/report/PHASE5C.md).

### Phase 5d: 12 more hours for the bigger PPO network (branch `claude/phase5d-ppo-more`)
* Continues phase 5c unchanged to 32 h (66.0k games). Command: `python -m goevo.ppo --channels 32 --blocks 6 --sym --hours 32 --lr 1e-4 --resume`.
* **Result:**
  * beats the 5c final 21 of 32 and the small PPO final 19 of 32 (59%);
  * mostly a plateau, only slightly stronger than the small network at a similar number of games.
* Report: [`runs/report/PHASE5D.md`](runs/report/PHASE5D.md).

### Phase 5e: bigger batches and a lower learning rate for the bigger PPO network (branch `claude/phase5e-ppo-bigbatch`)
* Continues phase 5d to 48 h (95.3k games) with 64 games per iteration, lr 5e-5 and minibatch 2048. Command: `python -m goevo.ppo --channels 32 --blocks 6 --sym --hours 48 --games 64 --lr 5e-5 --minibatch 2048 --resume`.
* **Result:**
  * beats the 5d final 7 of 8 at first; all 4 x 4 final policies, 18 of 32;
  * against the small PPO final, still only 19 of 32 (59%); plateau again after about 6 h.
* Report: [`runs/report/PHASE5E.md`](runs/report/PHASE5E.md).

### Phase 6: Chinese rules (branch `claude/phase6-chinese-rules`)
* **New rules:** pass any time, two passes in a row end the game, own-eye filling allowed, area scoring, komi 7.5, move cap 1083 (`goevo.match.CHINESE`). Phases 1–5 used forced passing.
* **Network:** the small phase 5 network (16 channels, 4 blocks) with 2 extra input planes (opponent just passed; mover is Black). Trained from scratch, 16 h, 96.6k games. Command: `python -m goevo.ppo --out work/p6_s0 --rules chinese --sym --hours 16`.
* **Result:**
  * the final policies beat the hour 0–6 snapshots 8 of 8, hours 8–10 7 of 8, hours 12–14 5 of 8, hour 15 4 of 8: gains slowing down at the end;
  * games stay long (about 600 moves): they have not learned to end games early by passing.
* Report: [`runs/report/PHASE6.md`](runs/report/PHASE6.md).

### Phase 6b: phase 6 continued at a lower learning rate (branch `claude/phase6b-chinese-lr`)
* Resumes phase 6 for 20 more hours (36 h, 175.8k games in total) with lr 1e-4, 64 games per iteration and minibatch 2048. Command: `python -m goevo.ppo --rules chinese --sym --hours 36 --games 64 --lr 1e-4 --minibatch 2048 --resume`.
* **Result:**
  * beats the phase 6 final 25 of 32 (78%);
  * most of the gain by about 28 h, then a plateau; games are still about 630 moves long.
* Report: [`runs/report/PHASE6B.md`](runs/report/PHASE6B.md).
* Watch and play games (PPO, evolved players, random, you): `notebooks/games_colab.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PrimozRavbar/go-evo/blob/main/notebooks/games_colab.ipynb)
* Play against them by clicking on the board: `notebooks/games_click_colab.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PrimozRavbar/go-evo/blob/main/notebooks/games_click_colab.ipynb)
* Train PPO yourself, every step written out in the notebook: `notebooks/ppo_training_colab.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PrimozRavbar/go-evo/blob/main/notebooks/ppo_training_colab.ipynb)

Details: [`runs/report/REPORT.md`](runs/report/REPORT.md). Notebook: `notebooks/phase2_colab.ipynb`.

**Checkpoints and repo size:** `*.pt` files are git-ignored. Only each run's final parents checkpoint is committed (force-added).
Generation 0 is never stored: `goevo.evolve.initial_population` regenerates it from the seed.

```bash
python -m goevo.phase2 --generations 2000 --seed 0 --workers 2 --out work/c_sel_s0   # add --resume to continue
```
```bash
python -m goevo.phase2 --generations 300 --seed 0 --out runs/ko_sel_s0
```
```bash
python -m goevo.phase2 --generations 300 --seed 0 --selection random --out runs/ko_ctl_s0
```

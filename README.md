# go-evo

Evolving Go players. Each player is a genome: the flat weight vector of a 10-layer MLP (PyTorch).
The players play each other, and the results are used as fitness.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PrimozRavbar/go-evo/blob/main/notebooks/phase1_colab.ipynb)

## Phase 1 (done)

* **Go API:** [PettingZoo `classic.go_v5`](https://pettingzoo.farama.org/environments/classic/go/), driven through its standard AEC API (`goevo.engine.PettingZooGo`).
* **Own engine as fallback:** `goevo.engine.NativeGo`, in pure numpy. It is used automatically when PettingZoo can't be imported.
  It has the same rules: area scoring with komi, suicide illegal, simple ko, game ends after two passes.
  `tests/` plays 20 random games on both engines and checks that they agree on every legal-move mask, every board and the final score.
* **Players:** `NetConfig(board_size=9, num_layers=10, hidden=128)` gives 10 Linear layers with tanh, about 184k parameters per genome.
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

Colab: open `notebooks/phase1_colab.ipynb` and run all cells.

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
notebooks/phase1_colab.ipynb
tests/test_goevo.py
```

## Next (phase 2)
Selection plus mutation/crossover on the flat genomes. A tournament per generation gives the fitness.

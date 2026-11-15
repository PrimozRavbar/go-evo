import numpy as np
import pytest

from goevo import (NativeGo, NetConfig, GameConfig, make_schedule, pettingzoo_available,
                   play_game, random_population)
from goevo.engine import area_score, BLACK, WHITE


def test_capture_suicide_ko():
    g2 = NativeGo(5, 0.5)
    g2._board[:] = np.array([[0, 1, -1, 0, 0],
                             [1, -1, 0, -1, 0],
                             [0, 1, -1, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]], np.int8)
    g2._to_play = BLACK
    g2.play(1 * 5 + 2)        # black captures white at (1,1)
    assert g2.board[1, 1] == 0
    assert not g2.legal_mask()[1 * 5 + 1]  # white may not retake ko immediately
    # suicide: white at (0,0) would have no liberties and capture nothing
    assert not g2.legal_mask()[0]


def test_area_score():
    b = np.zeros((3, 3), np.int8)
    b[:, 1] = BLACK
    assert area_score(b, 0.5) == 9 - 0.5
    b[:, 2] = WHITE
    assert area_score(b, 0.5) == (6 - 3) - 0.5


def test_schedule_balanced():
    s = make_schedule(10, 5, seed=3)
    assert len(s) == 25
    counts = np.bincount(np.array(s).ravel(), minlength=10)
    assert (counts == 5).all()
    assert len({frozenset(p) for p in s}) == 25  # no repeated pairing
    blacks = np.bincount([b for b, _ in s], minlength=10)
    assert blacks.max() - blacks.min() <= 2


def test_genome_size_and_layers():
    cfg = NetConfig(board_size=9, num_layers=10, hidden=128)
    p = random_population(1, cfg)[0]
    import torch.nn as nn
    assert sum(isinstance(m, nn.Linear) for m in p.net) == 10
    assert p.genome.size == cfg.genome_size == sum(x.numel() for x in p.net.parameters())


@pytest.mark.parametrize("backend", ["native", "pettingzoo"])
def test_phase1_tournament_runs(backend):
    if backend == "pettingzoo" and not pettingzoo_available():
        pytest.skip("pettingzoo not installed")
    from goevo import run_tournament
    cfg = NetConfig(board_size=9, num_layers=10, hidden=32)
    players = random_population(10, cfg, seed=1)
    res = run_tournament(players, 5, GameConfig(board_size=9, backend=backend), verbose=False)
    assert len(res.records) == 25
    assert all(s.games == 5 for s in res.standings)


@pytest.mark.skipif(not pettingzoo_available(), reason="pettingzoo not installed")
@pytest.mark.parametrize("seed", range(20))
def test_native_matches_pettingzoo(seed):
    """Random games: both engines must agree on legal moves, boards and final score."""
    from goevo import PettingZooGo
    rng = np.random.default_rng(seed)
    size = [5, 7, 9, 19][seed % 4]
    a, b = NativeGo(size, 7.5), PettingZooGo(size, 7.5)
    for _ in range(4 * size * size):
        if a.is_over():
            break
        ma, mb = a.legal_mask(), b.legal_mask()
        np.testing.assert_array_equal(ma, mb)
        stones = np.flatnonzero(ma[:-1])
        mv = a.pass_action if (len(stones) == 0 or rng.random() < 0.03) else int(rng.choice(stones))
        a.play(mv)
        b.play(mv)
        np.testing.assert_array_equal(a.board, b.board)
        assert a.is_over() == b.is_over()
    assert a.score() == b.score()


def test_game_record_and_sgf():
    cfg = NetConfig(board_size=9, num_layers=10, hidden=32)
    p, q = random_population(2, cfg, seed=2)
    rec = play_game(p, q, GameConfig(board_size=9, backend="native"))
    assert len(rec.boards) == len(rec.moves) + 1
    assert rec.to_sgf().startswith("(;GM[1]")


def test_defaults_are_19x19():
    assert NetConfig().board_size == 19 and GameConfig().board_size == 19
    assert GameConfig().resolved().max_moves == 3 * 19 * 19


def test_mutation_no_crossover():
    from goevo.player import mutate, random_genome
    cfg = NetConfig(board_size=9, hidden=16)
    rng = np.random.default_rng(0)
    g = random_genome(cfg, rng)
    c = mutate(g, cfg, 0.1, rng)
    assert c.shape == g.shape and c.dtype == np.float32 and not np.array_equal(c, g)
    assert np.abs(c - g).mean() < 0.1 * np.abs(g).mean()


@pytest.mark.parametrize("selection", ["winners", "random"])
def test_evolution_roundrobin_winners_reproduce(tmp_path, selection):
    from goevo.evolve import EvoConfig, run_evolution
    net = NetConfig(board_size=9, hidden=16)
    evo = EvoConfig(pop_size=10, tournament="roundrobin", n_parents=0, generations=3, bench_every=2,
                    bench_panel=1, checkpoint_every=2, checkpoint_parents_only=False,
                    out_dir=str(tmp_path), workers=1, selection=selection)
    hist = run_evolution(evo, net, GameConfig(board_size=9, backend="native"), verbose=False)
    assert len(hist) == 4
    for prev, cur in zip(hist, hist[1:]):
        assert len(cur["ids"]) == 10
        for gid, par in zip(cur["ids"], cur["parents"]):
            if gid not in prev["ids"]:
                assert par in prev["winners"]   # only winners have children
            else:
                assert gid in prev["winners"]   # survivors are winners
    assert (tmp_path / "gen0000.pt").exists() and (tmp_path / "gen0003.pt").exists()
    assert "bench_mean" in hist[2]


@pytest.mark.parametrize("selection", ["fitness", "random"])
def test_evolution_knockout(tmp_path, selection):
    from goevo import load_population
    from goevo.evolve import EvoConfig, run_evolution
    net = NetConfig(board_size=9, hidden=16)
    evo = EvoConfig(pop_size=16, n_parents=4, generations=3, bench_every=3, bench_panel=1,
                    external_top=4, out_dir=str(tmp_path), workers=1, selection=selection)
    hist = run_evolution(evo, net, GameConfig(board_size=9, backend="native"), verbose=False)
    for h in hist:
        assert h["games"] == 15                       # single elimination: n - 1 games
        assert sorted(h["wins"]) == [0] * 8 + [1] * 4 + [2] * 2 + [3, 4]
        assert h["n_winners"] == 4
        assert bool(h["qualifier"]) == (selection == "fitness")
    for prev, cur in zip(hist, hist[1:]):
        assert len(cur["ids"]) == 16
        for gid, par in zip(cur["ids"], cur["parents"]):
            assert (par in prev["winners"]) if gid not in prev["ids"] else (gid in prev["winners"])
        born = [par for gid, par in zip(cur["ids"], cur["parents"]) if gid not in prev["ids"]]
        assert sorted(born.count(p) for p in prev["winners"]) == [3, 3, 3, 3]
    if selection == "fitness":  # parents are the 4 best by fitness
        h = hist[0]
        best = sorted(range(16), key=lambda i: -h["fitness"][i])[:4]
        assert {h["ids"][i] for i in best} == set(h["winners"])
    files = sorted(p.name for p in tmp_path.glob("*.pt"))
    assert files == ["gen0003.pt"]                    # only the final checkpoint
    genomes, *_ = load_population(str(tmp_path / "gen0003.pt"))
    assert len(genomes) == 4                          # only the parents

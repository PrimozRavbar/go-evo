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
                    bench_panel=1, bench_rm_games=2, checkpoint_every=2, checkpoint_parents_only=False,
                    state_every=0,
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
                    bench_rm_games=2, external_top=4, checkpoint_every=0, state_every=0,
                    out_dir=str(tmp_path), workers=1, selection=selection)
    hist = run_evolution(evo, net, GameConfig(board_size=9, backend="native"), verbose=False)
    for h in hist:
        assert h["games"] == 15                       # single elimination: n - 1 games
        assert sorted(h["wins"]) == [0] * 8 + [1] * 4 + [2] * 2 + [3, 4]
        assert h["n_winners"] == 4
        # qualifier = games vs hall of fame, which is empty in generation 0
        assert bool(h["qualifier"]) == (selection == "fitness" and h["gen"] > 0)
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


@pytest.mark.parametrize("seed", range(16))
def test_fast_engine_matches_native(seed):
    from goevo.fastgo import FastGo
    from goevo.match import own_eye_mask
    rng = np.random.default_rng(100 + seed)
    size = [5, 9, 13, 19][seed % 4]
    a, b = NativeGo(size, 7.5), FastGo(size, 7.5)
    for _ in range(5 * size * size):
        if a.is_over():
            break
        ma, mb = a.legal_mask(), b.legal_mask()
        np.testing.assert_array_equal(ma, mb)
        np.testing.assert_array_equal(own_eye_mask(a), b.own_eye_mask())
        stones = np.flatnonzero(ma[:-1])
        mv = a.pass_action if (len(stones) == 0 or rng.random() < 0.02) else int(rng.choice(stones))
        a.play(mv)
        b.play(mv)
        np.testing.assert_array_equal(a.board, b.board)
        assert a.is_over() == b.is_over()
        assert a.score() == b.score()


def test_numpy_forward_matches_torch():
    from goevo.player import encode_state
    from goevo.fastgo import FastGo
    cfg = NetConfig(board_size=19)
    p = random_population(1, cfg, seed=3)[0]
    g = FastGo(19)
    x = encode_state(g, g.legal_mask(), 1083)
    np.testing.assert_allclose(p.logits(x), p.logits_torch(x), rtol=1e-4, atol=1e-4)


def test_forced_pass_rule():
    from goevo.player import RandomPlayer
    rec = play_game(RandomPlayer("a"), RandomPlayer("b"), GameConfig(board_size=9, backend="fast"),
                    np.random.default_rng(0))
    n = 81
    # with pass_rule="forced" the only passes are the final ones (nothing else possible)
    passes = [i for i, m in enumerate(rec.moves) if m == n]
    assert passes and passes[-1] == len(rec.moves) - 1
    assert all(i >= len(rec.moves) - 3 or rec.moves[i + 1] != n for i in passes[:-2])


def test_resume_reproduces_uninterrupted_run(tmp_path):
    from goevo.evolve import EvoConfig, run_evolution
    net = NetConfig(board_size=9, hidden=16)
    gc = GameConfig(board_size=9, backend="fast")
    kw = dict(pop_size=8, n_parents=2, bench_every=0, checkpoint_every=0, state_every=2, workers=1)
    full = run_evolution(EvoConfig(generations=5, out_dir=str(tmp_path / "a"), **kw), net, gc, verbose=False)
    run_evolution(EvoConfig(generations=3, out_dir=str(tmp_path / "b"), **kw), net, gc, verbose=False)
    resumed = run_evolution(EvoConfig(generations=5, out_dir=str(tmp_path / "b"), resume=True, **kw),
                            net, gc, verbose=False)
    assert [h["ids"] for h in resumed] == [h["ids"] for h in full]
    assert [h["wins"] for h in resumed] == [h["wins"] for h in full]


@pytest.mark.parametrize("seed", range(4))
def test_liberty_map_fast_matches_python(seed):
    from goevo.engine import liberty_map
    from goevo.fastgo import FastGo
    rng = np.random.default_rng(seed)
    g = FastGo([5, 9, 13, 19][seed])
    for _ in range(3 * g.size ** 2):
        stones = np.flatnonzero(g.legal_mask()[:-1])
        if not len(stones):
            break
        g.play(int(rng.choice(stones)))
        np.testing.assert_array_equal(g.liberty_map(), liberty_map(g.board))


def test_conv_net_numpy_matches_torch_and_size():
    from goevo.player import encode_planes, random_genome, Player
    from goevo.fastgo import FastGo
    cfg = NetConfig(board_size=19, arch="conv", channels=16)
    assert cfg.genome_size == 22514
    assert NetConfig(board_size=9, arch="conv").genome_size == cfg.genome_size  # board-size independent
    p = Player(random_genome(cfg, np.random.default_rng(1)), cfg)
    g = FastGo(19)
    rng = np.random.default_rng(0)
    for _ in range(80):
        g.play(int(rng.choice(np.flatnonzero(g.legal_mask()[:-1]))))
    x = encode_planes(g, g.legal_mask())
    assert x.shape == (11, 19, 19)
    np.testing.assert_allclose(p.logits(x), p.logits_torch(x), rtol=1e-4, atol=1e-4)


def test_conv_evolution_smoke(tmp_path):
    from goevo import load_population
    from goevo.evolve import EvoConfig, run_evolution
    net = NetConfig(board_size=9, arch="conv", num_layers=3, channels=4)
    evo = EvoConfig(pop_size=8, n_parents=2, generations=2, external_top=2, hof_games=2,
                    bench_every=2, bench_rm_games=2, checkpoint_every=0, state_every=0,
                    out_dir=str(tmp_path), workers=1)
    hist = run_evolution(evo, net, GameConfig(board_size=9, backend="fast"), verbose=False)
    assert len(hist) == 3 and hist[-1]["n_winners"] == 2
    genomes, cfg, *_ = load_population(str(tmp_path / "gen0002.pt"))
    assert cfg.arch == "conv" and genomes[0].size == net.genome_size


def test_lowrank_net_matches_torch_and_size():
    from goevo.player import encode_state, random_genome, Player, layer_scales
    from goevo.fastgo import FastGo
    cfg = NetConfig(board_size=19, arch="lowrank", num_layers=20, rank=32)
    i, h, o = 4 * 361 + 1, 128, 362
    assert cfg.genome_size == 32 * (h + i) + 18 * 32 * 2 * h + 32 * (o + h) + 19 * h + o
    g = random_genome(cfg, np.random.default_rng(1))
    assert g.size == layer_scales(cfg).size == cfg.genome_size
    p = Player(g, cfg)
    x = encode_state(FastGo(19), FastGo(19).legal_mask(), 1083)
    np.testing.assert_allclose(p.logits(x), p.logits_torch(x), rtol=1e-4, atol=1e-4)
    # each W = A @ B has rank <= r
    A = g[:h * 32].reshape(h, 32)
    B = g[h * 32:h * 32 + 32 * i].reshape(32, i)
    assert np.linalg.matrix_rank(A @ B) == 32


def test_lowrank_evolution_smoke(tmp_path):
    from goevo import load_population
    from goevo.evolve import EvoConfig, run_evolution
    net = NetConfig(board_size=9, arch="lowrank", num_layers=4, hidden=16, rank=4)
    evo = EvoConfig(pop_size=8, n_parents=2, generations=2, external_top=2, hof_games=2,
                    bench_every=0, checkpoint_every=0, state_every=0, out_dir=str(tmp_path), workers=1)
    hist = run_evolution(evo, net, GameConfig(board_size=9, backend="fast"), verbose=False)
    assert len(hist) == 3
    genomes, cfg, *_ = load_population(str(tmp_path / "gen0002.pt"))
    assert cfg.arch == "lowrank" and cfg.rank == 4 and genomes[0].size == net.genome_size

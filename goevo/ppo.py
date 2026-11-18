"""Phase 5: PPO self-play baseline with the phase-4 network.

The policy is exactly the phase-4 residual CNN (``arch="resnet"``, 16 channels, 4 blocks,
~20.5k params): its weights are kept in genome order, so a trained policy is an ordinary
genome and is evaluated with the same ``Player`` (masked argmax) as the evolved players.
A tiny value head (global mean of the final features -> 16 tanh -> 1 tanh, ~290 params) is
used only for training.

Everything is pytorch. Each iteration plays ``games`` games with 4 worker processes
(batch-1 torch inference, sampling at temperature 1). With probability ``p_self`` the opponent is the current policy (both sides
are training data), otherwise a past snapshot from the pool (only the learner's side).
Reward is +1 / -1 at the end of the game; advantages by GAE (gamma 1, lambda 0.95).
Then a few PPO epochs (clipped objective, value loss, entropy bonus) in torch on CPU.

The budget is wall-clock training time (``--hours``), resumable via ``state.pt``:

    python -m goevo.ppo --out work/p5_s0 --hours 8
    python -m goevo.ppo --out work/p5_s0 --hours 8 --resume
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .engine import make_game
from .match import GameConfig, own_eye_mask
from .parallel import _play
from .player import NetConfig, ResNet, encode_planes, random_genome, save_population

VH = 16  # value head hidden width


# ----------------------------------------------------------------------------- model
class PPONet(nn.Module):
    """Phase-4 ResNet (``self.pi``, genome order) + small value head on the shared trunk."""

    def __init__(self, cfg: NetConfig):
        super().__init__()
        self.cfg = cfg
        self.pi = ResNet(cfg)
        self.v1 = nn.Linear(cfg.channels, VH)
        self.v2 = nn.Linear(VH, 1)
        nn.init.normal_(self.v1.weight, 0.0, 1.0 / np.sqrt(cfg.channels))
        nn.init.zeros_(self.v1.bias)
        nn.init.normal_(self.v2.weight, 0.0, 0.1 / np.sqrt(VH))
        nn.init.zeros_(self.v2.bias)

    def forward(self, x):
        r = self.pi
        h = r.stem(x)
        for b in r.blocks:
            h = b(h)
        h = r.act(r.norm(h))
        g = h.mean((2, 3))
        logits = torch.cat([r.point(h).flatten(1), r.pass_(g)], 1)
        v = torch.tanh(self.v2(torch.tanh(self.v1(g)))).squeeze(1)
        return logits, v

    def genome(self) -> np.ndarray:
        return nn.utils.parameters_to_vector(self.pi.parameters()).detach().numpy().astype(np.float32)

    def set_genome(self, g: np.ndarray):
        nn.utils.vector_to_parameters(torch.as_tensor(g, dtype=torch.float32), self.pi.parameters())

    def value_head(self):
        return [p.detach().numpy().copy() for p in (self.v1.weight, self.v1.bias, self.v2.weight, self.v2.bias)]


def load_value_head(model: PPONet, vh):
    with torch.no_grad():
        for p, w in zip((model.v1.weight, model.v1.bias, model.v2.weight, model.v2.bias), vh):
            p.copy_(torch.as_tensor(w))


def save_full(path: str, model: PPONet, extra: dict | None = None) -> None:
    """Whole trained network (policy + value head) as a torch state_dict, for analysis."""
    torch.save({"cfg": asdict(model.cfg), "state_dict": model.state_dict(), "extra": extra or {}}, path)


def load_full(path: str) -> PPONet:
    """Load a ``save_full`` file: ``net(x) -> (move logits (B, N*N+1), value (B,))``, eval mode.

    ``net.pi`` is the policy alone (same module as ``Player(genome, cfg).net``)."""
    d = torch.load(path, weights_only=False)
    net = PPONet(NetConfig(**d["cfg"]))
    net.load_state_dict(d["state_dict"])
    net.extra = d.get("extra", {})
    return net.eval()


# ----------------------------------------------------------------------------- rollouts
def allowed_moves(game, legal, gc: GameConfig):
    """Same move rules as match.play_game (gc must be resolved)."""
    allowed = legal.copy()
    if gc.forbid_own_eye_fill:
        eyes = game.own_eye_mask() if hasattr(game, "own_eye_mask") else own_eye_mask(game)
        allowed[:-1] &= ~eyes
    stone = allowed[:-1].any()
    if gc.pass_rule == "forced":
        allowed[-1] = not stone
    else:
        allowed[-1] = (game.move_number >= gc.min_moves_before_pass) or not stone
    if not allowed.any():
        allowed[-1] = True
    return allowed


def _sample(z, allowed, rng):
    z = np.where(allowed, z.astype(np.float64), -np.inf)
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    a = int(rng.choice(len(p), p=p))
    return a, float(np.log(p[a]))


def _sym(x, k):
    """k-th of the 8 board symmetries (k//4: mirror, k%4: quarter turns) on the last two axes."""
    if k >= 4:
        x = x[..., ::-1]
    return np.rot90(x, k % 4, axes=(-2, -1))


_PERMS = {}


def sym_perm(n, k):
    """perm[j] = original point shown at point j of the transformed board."""
    if (n, k) not in _PERMS:
        _PERMS[n, k] = np.ascontiguousarray(_sym(np.arange(n * n).reshape(n, n), k)).ravel()
    return _PERMS[n, k]


def _rollout(job):
    """One game. Returns the learner-side trajectories (one, or two when opponent is self).

    With ``sym``, every move is decided on a randomly rotated / mirrored board and stored
    in that orientation (exact PPO data, all 8 orientations seen in training)."""
    pol, vh, opp, lcol, seed, net_cfg, gc, sym = job
    rng = np.random.default_rng(seed)
    me = PPONet(net_cfg).eval()
    me.set_genome(pol)
    load_value_head(me, vh)
    other = None
    if opp is not None:
        other = PPONet(net_cfg).eval()
        other.set_genome(opp)
    game = make_game(gc.board_size, gc.komi, gc.backend)
    traj = {1: [], -1: []}
    with torch.no_grad():
        while not game.is_over() and game.move_number < gc.max_moves:
            legal = game.legal_mask()
            allowed = allowed_moves(game, legal, gc)
            c = game.to_play
            x = encode_planes(game, legal)
            n = gc.board_size
            if sym:
                k = int(rng.integers(8))
                perm = sym_perm(n, k)
                x = np.ascontiguousarray(_sym(x, k))
                allowed = np.append(allowed[:-1][perm], allowed[-1])
            learner = other is None or c == lcol
            z, v = (me if learner else other)(torch.from_numpy(x)[None])
            a, lp = _sample(z[0].numpy(), allowed, rng)
            if learner:
                traj[c].append((x.astype(np.int8), allowed, a, lp, float(v[0])))
            game.play(int(perm[a]) if sym and a < n * n else a)
    w = game.winner()
    out = []
    for c, steps in traj.items():
        if steps:
            out.append(dict(planes=np.stack([s[0] for s in steps]), allowed=np.stack([s[1] for s in steps]),
                            actions=np.array([s[2] for s in steps], np.int64),
                            logp=np.array([s[3] for s in steps], np.float32),
                            values=np.array([s[4] for s in steps], np.float32),
                            reward=float(w * c), vs_self=other is None))
    return out, game.move_number


def gae(values, reward, lam=0.95):
    """gamma = 1, reward only at the end; returns (advantages, returns)."""
    T = len(values)
    nxt = np.append(values[1:], 0.0)
    delta = nxt - values
    delta[-1] += reward
    adv = np.zeros(T, np.float32)
    acc = 0.0
    for t in range(T - 1, -1, -1):
        acc = delta[t] + lam * acc
        adv[t] = acc
    return adv, adv + values


def _worker_init():
    torch.set_num_threads(1)


# ----------------------------------------------------------------------------- training
def ppo_update(model, opt, batch, a):
    X, M, A, LP, ADV, RET = batch
    N = len(A)
    ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
    stats = []
    for ep in range(a.epochs):
        perm = np.random.permutation(N)
        kls = []
        for s in range(0, N, a.minibatch):
            idx = perm[s:s + a.minibatch]
            x = torch.from_numpy(X[idx].astype(np.float32))
            m = torch.from_numpy(M[idx])
            act = torch.from_numpy(A[idx])
            old = torch.from_numpy(LP[idx])
            adv = torch.from_numpy(ADV[idx])
            ret = torch.from_numpy(RET[idx])
            logits, v = model(x)
            logits = logits.masked_fill(~m, -1e9)
            logp_all = F.log_softmax(logits, 1)
            logp = logp_all.gather(1, act[:, None])[:, 0]
            ratio = torch.exp(logp - old)
            pg = -torch.min(ratio * adv, torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * adv).mean()
            vl = F.mse_loss(v, ret)
            p = logp_all.exp()
            ent = -(p * logp_all.masked_fill(~m, 0.0)).sum(1).mean()
            loss = pg + a.vf_coef * vl - a.ent_coef * ent
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), a.max_grad)
            opt.step()
            kl = (old - logp).mean().item()
            kls.append(kl)
            stats.append((pg.item(), vl.item(), ent.item(), kl))
        if np.mean(kls) > a.target_kl:
            break
    s = np.array(stats)
    return dict(pg=float(s[:, 0].mean()), vl=float(s[:, 1].mean()), ent=float(s[:, 2].mean()),
                kl=float(s[:, 3].mean()), epochs=ep + 1)


def bench_random(pool, genome, net_cfg, gc, games, seed):
    s, rm = ("genome", "P", genome), ("random", "rm")
    pairs = [(s, rm) if k % 2 == 0 else (rm, s) for k in range(games)]
    jobs = [(b, w, net_cfg, gc, seed * 100003 + k, False) for k, (b, w) in enumerate(pairs)]
    recs = pool.map(_play, jobs, chunksize=1)
    return float(np.mean([r.winner == (1 if k % 2 == 0 else -1) for k, r in enumerate(recs)]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work/p5_s0")
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--sym", action="store_true", help="random board symmetry for every move (augmentation)")
    ap.add_argument("--games", type=int, default=32, help="games per iteration")
    ap.add_argument("--p-self", type=float, default=0.5, help="prob. opponent = current policy")
    ap.add_argument("--pool-size", type=int, default=50)
    ap.add_argument("--snapshot-every", type=int, default=10, help="iterations between pool snapshots")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--max-grad", type=float, default=0.5)
    ap.add_argument("--target-kl", type=float, default=0.03)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--bench-every", type=int, default=50)
    ap.add_argument("--bench-games", type=int, default=24)
    ap.add_argument("--checkpoint-every-hours", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=5, help="iterations between state.pt saves")
    ap.add_argument("--max-iters", type=int, default=0, help="stop after this many iterations (0 = no limit)")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    net_cfg = NetConfig(board_size=19, arch="resnet", channels=a.channels, blocks=a.blocks)
    gc = GameConfig(board_size=19).resolved()
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    model = PPONet(net_cfg)
    state_path = f"{a.out}/state.pt"
    if a.resume and os.path.exists(state_path):
        st = torch.load(state_path, weights_only=False)
        model.load_state_dict(st["model"])
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        opt.load_state_dict(st["opt"])
        for g in opt.param_groups:  # the command-line learning rate wins (e.g. lower it on resume)
            g["lr"] = a.lr
        it, elapsed, games_total, steps_total = st["it"], st["elapsed"], st["games"], st["steps"]
        opp_pool, next_ckpt = st["pool"], st["next_ckpt"]
        rng.bit_generator.state = st["rng"]
        hist = [json.loads(l) for l in open(f"{a.out}/history.jsonl")] if os.path.exists(f"{a.out}/history.jsonl") else []
        hist = [h for h in hist if h["it"] < it]
        with open(f"{a.out}/history.jsonl", "w") as f:
            f.writelines(json.dumps(h) + "\n" for h in hist)
        print(f"resumed at iteration {it}, {elapsed / 3600:.2f} h used, lr {a.lr}", flush=True)
    else:
        model.set_genome(random_genome(net_cfg, rng))  # same init distribution as evolution
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        it, elapsed, games_total, steps_total, next_ckpt = 0, 0.0, 0, 0, 0.0
        opp_pool = [model.genome()]
        json.dump({"args": vars(a), "net": asdict(net_cfg), "game": asdict(gc),
                   "policy_params": net_cfg.genome_size,
                   "value_params": sum(p.numel() for p in (*model.v1.parameters(), *model.v2.parameters()))},
                  open(f"{a.out}/config.json", "w"), indent=1)
        open(f"{a.out}/history.jsonl", "w").close()

    ctx = mp.get_context("fork")
    pool = ctx.Pool(a.workers, initializer=_worker_init)
    torch.set_num_threads(a.workers)
    budget = a.hours * 3600
    print(f"PPO: policy {net_cfg.genome_size} params, {a.games} games/iter, budget {a.hours} h", flush=True)

    def save_state():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "it": it, "elapsed": elapsed,
                    "games": games_total, "steps": steps_total, "pool": opp_pool,
                    "next_ckpt": next_ckpt, "rng": rng.bit_generator.state}, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)

    while elapsed < budget and (a.max_iters <= 0 or it < a.max_iters):
        if elapsed >= next_ckpt:  # hourly policy checkpoint (for ancestor comparisons)
            save_population(f"{a.out}/h{round(next_ckpt / 3600):02d}.pt", [model.genome()], net_cfg,
                            [f"it{it}"], {"it": it, "games": games_total, "hours": elapsed / 3600})
            next_ckpt += a.checkpoint_every_hours * 3600
        t0 = time.time()
        pol, vh = model.genome(), model.value_head()
        jobs = []
        for k in range(a.games):
            opp = None if rng.random() < a.p_self else opp_pool[int(rng.integers(len(opp_pool)))]
            jobs.append((pol, vh, opp, 1 if k % 2 == 0 else -1, int(rng.integers(2 ** 31)), net_cfg, gc, a.sym))
        res = pool.map(_rollout, jobs, chunksize=1)
        t1 = time.time()
        trajs = [t for r in res for t in r[0]]
        lens = [r[1] for r in res]
        X = np.concatenate([t["planes"] for t in trajs])
        M = np.concatenate([t["allowed"] for t in trajs])
        A = np.concatenate([t["actions"] for t in trajs])
        LP = np.concatenate([t["logp"] for t in trajs])
        advs, rets = zip(*[gae(t["values"], t["reward"], a.lam) for t in trajs])
        stats = ppo_update(model, opt, (X, M, A, LP, np.concatenate(advs), np.concatenate(rets)), a)
        pool_w = [t["reward"] > 0 for t in trajs if not t["vs_self"]]
        it += 1
        games_total += a.games
        steps_total += len(A)
        if it % a.snapshot_every == 0:
            opp_pool.append(model.genome())
            opp_pool = opp_pool[-a.pool_size:]
        dt = time.time() - t0
        elapsed += dt
        rec = dict(it=it, hours=round(elapsed / 3600, 4), games=games_total, steps=len(A),
                   moves=float(np.mean(lens)), win_vs_pool=float(np.mean(pool_w)) if pool_w else None,
                   t_play=round(t1 - t0, 1), t_train=round(time.time() - t1, 1), **stats)
        msg = (f"[ppo] it {it:5d} {elapsed / 3600:5.2f}h games={games_total} steps={len(A)} "
               f"moves={rec['moves']:.0f} pool_wr={rec['win_vs_pool'] if pool_w else float('nan'):.2f} "
               f"ent={stats['ent']:.2f} kl={stats['kl']:.3f} vl={stats['vl']:.3f} ep={stats['epochs']} "
               f"({t1 - t0:.1f}+{time.time() - t1:.1f}s)")
        if a.bench_every and it % a.bench_every == 0:
            rec["bench_rm"] = bench_random(pool, model.genome(), net_cfg, gc, a.bench_games, it)
            msg += f" rm={rec['bench_rm']:.2f}"
        print(msg, flush=True)
        with open(f"{a.out}/history.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        if it % a.save_every == 0:
            save_state()
    save_state()
    # final "parents": the last 4 policies, for head-to-head evaluation like the evolved runs
    finals = (opp_pool[-3:] + [model.genome()])[-4:]
    save_population(f"{a.out}/final.pt", finals, net_cfg, [f"final{k}" for k in range(len(finals))],
                    {"it": it, "games": games_total, "hours": elapsed / 3600})
    save_full(f"{a.out}/full.pt", model, {"it": it, "games": games_total, "hours": elapsed / 3600})
    pool.close()
    pool.join()
    print(f"done in {elapsed / 3600:.2f} h, {it} iterations, {games_total} games", flush=True)


if __name__ == "__main__":
    main()

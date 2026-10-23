"""Lockstep leaf rollouts on ``VecNLHE`` for single-net neural blueprints.

The same estimator as :func:`pokerbot.search.leaf._rollout` (see the
:mod:`~pokerbot.search.leaf` docstring), run for every rollout of a solve at
once: each rollout is one slot of a ``VecNLHE`` dealt with its runout board
and replayed to its leaf, and every step queries the blueprint for all 1326
combos of every live slot in a few large network calls. Hand-strength columns
come from a per-board cache (a rollout's board changes only when a card is
dealt, and many rollouts share turn boards).

Supports blueprints whose policy is one net without own-reach weighting
(``NeuralBlueprint`` of a distilled run); :func:`supports` says whether the
fast path applies, and :mod:`~pokerbot.search.leaf` falls back otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..blueprint.deepcfr.features import features_from_obs
from ..blueprint.deepcfr.range_policy import ALL_COMBOS
from ..blueprint.deepcfr.strength import load_strength
from ..blueprint.deepcfr.traversal import deal_env
from ..env import actions as am
from ..env.actions import CHECK_CALL, FOLD, RAISE
from ..env.cards import NO_CARD
from .combos import NUM_COMBOS

BOARD_LEN = (0, 3, 4, 5)
CLASSES = {"fold": FOLD, "call": CHECK_CALL, "raise": RAISE}


def supports(bp: Any) -> bool:
    """Whether ``bp`` can use the lockstep path (one net, no reach weighting)."""
    agent = getattr(bp, "agent", None)
    pols = getattr(agent, "policies", None)
    return bool(pols) and all(len(p) == 1 and not p.reach_weighted for p in pols)


@dataclass
class Rollout:
    state: Any  # engine state at the leaf
    board: list[int]  # the full 5-card runout
    chooser: int  # the leaf chooser (gets the k continuation strategies)


@dataclass
class RolloutResult:
    w_ch: torch.Tensor  # [R, k, 1326] chooser weights per continuation
    w_nc: torch.Tensor  # [R, 1326] other-player weights
    kind: list[int]  # 0 fold, 1 showdown
    folder: list[int]
    amount: list[int]


class StrengthCache:
    """``[1326, 11]`` strength columns of every combo, per board prefix (GPU)."""

    def __init__(self, tables: Any, device: torch.device) -> None:
        self.tables = tables
        self.device = device
        self._cache: dict[tuple, torch.Tensor] = {}
        self._holes = ALL_COMBOS.numpy()

    def get(self, board: tuple[int, ...], street: int) -> torch.Tensor:
        key = (street, board)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        holes = self._holes.copy()
        clash = np.isin(holes, np.asarray(board, dtype=np.int64)).any(1)
        free = [c for c in range(52) if c not in board][:2]
        holes[clash] = free  # placeholder rows (zero reach anyway): lookups reject duplicates
        b = np.full((NUM_COMBOS, 5), NO_CARD, dtype=np.int64)
        b[:, : len(board)] = board
        cols = self.tables.lookup(holes, b, np.full(NUM_COMBOS, street))
        out = torch.from_numpy(cols).to(self.device, torch.float32)
        self._cache[key] = out
        return out


def _replay(env: Any, histories: list[list[tuple[int, int]]]) -> None:
    """Apply each slot's concrete history in lockstep (shorter ones wait),
    recording off-tree sizes with the deterministic pseudo-harmonic mapping."""
    L = max((len(h) for h in histories), default=0)
    dev = env.device
    for j in range(L):
        sel = torch.tensor([i for i, h in enumerate(histories) if len(h) > j], device=dev)
        sub = env.select(sel)
        kind = torch.tensor([histories[i][j][0] for i in sel.tolist()], device=dev)
        amount = torch.tensor([histories[i][j][1] for i in sel.tolist()], device=dev)
        info = sub.legal_info()
        te = sub._targets(info)
        me = sub._mask(info, te)
        rec = am.harmonic_abstract(
            sub.tab,
            info.street,
            kind,
            amount,
            te,
            me,
            info.pot,
            info.max_bet,
            info.to_call,
            info.max_raise_to,
            torch.full((sub.n,), 0.5, device=dev),
        )
        sub.step_concrete(kind, amount, validate=True, record=rec)
        for name in env._STATE:
            getattr(env, name)[sel] = getattr(sub, name)


def _history(state: Any) -> list[tuple[int, int]]:
    out = []
    for _st, _p, a in state.history:
        k = int(a.kind)
        out.append((k, int(a.amount) if k == RAISE else 0))
    return out


def make_env(rollouts: list[Rollout], spec: Any, game_config: Any, device: Any) -> Any:
    """A ``VecNLHE`` with one slot per rollout at its leaf: placeholder hole
    cards for both seats, the rollout's runout board, the leaf's history."""
    from ..env.config import GameConfig

    decks = []
    for ro in rollouts:
        free = [c for c in range(52) if c not in set(ro.board)]
        decks.append(free[:4] + list(ro.board) + free[4:])
    gc = GameConfig(
        num_players=2,
        stacks=[int(x) for x in game_config.stacks],
        small_blind=int(game_config.small_blind),
        big_blind=int(game_config.big_blind),
        ante=int(getattr(game_config, "ante", 0)),
    )
    env = deal_env(decks, [int(ro.state.button) for ro in rollouts], gc, spec, device)
    _replay(env, [_history(ro.state) for ro in rollouts])
    return env


def slot_policies(
    agent: Any,
    env: Any,
    idx: torch.Tensor,
    feats: dict[str, torch.Tensor],
    boards: torch.Tensor,
    cache: StrengthCache | None,
    bf16: bool = False,
) -> torch.Tensor:
    """``[m, 1326, A]`` blueprint policy of every combo for the actor of each
    slot ``idx``, normalised over the legal actions as ``policy_matrix`` does
    (uniform where a row has no legal mass)."""
    device = env.device
    m = idx.numel()
    A = env.num_actions
    street = env.street.clamp(0, 3)
    f = {key: v[idx].repeat_interleave(NUM_COMBOS, 0) for key, v in feats.items()}
    f["cards"][:, :2] = ALL_COMBOS.to(device).repeat(m, 1)
    if cache is not None:
        cols = [
            cache.get(
                tuple(int(c) for c in boards[i, : BOARD_LEN[int(street[i])]].tolist()),
                int(street[i]),
            )
            for i in idx.tolist()
        ]
        f["scalars"] = torch.cat([f["scalars"], torch.cat(cols, 0).to(f["scalars"].dtype)], 1)
    seat = env.actor[idx]
    P = torch.empty(m * NUM_COMBOS, A, device=device)
    for s in (0, 1):
        rows = (seat == s).repeat_interleave(NUM_COMBOS).nonzero().squeeze(1)
        if rows.numel():
            sub = {key: v[rows] for key, v in f.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                P[rows] = agent.policies[s].net_policies(sub)[0].float().to(device)
    P = P.view(m, NUM_COMBOS, A)
    legal = env.legal_mask()[idx].float()
    P = P.clamp(min=0) * legal[:, None]
    tot = P.sum(-1, keepdim=True)
    uni = (legal / legal.sum(-1, keepdim=True))[:, None]
    return torch.where(tot > 0, P / tot.clamp(min=1e-30), uni)


@torch.no_grad()
def run_rollouts(
    rollouts: list[Rollout],
    bp: Any,
    game_config: Any,
    strategies: list[str],
    bias: float,
    explore: float,
    device: torch.device,
    seed: int = 0,
    slots_per_call: int = 48,
    bf16: bool | None = None,
) -> RolloutResult:
    """Play every rollout to the end of the hand; weights as in ``leaf._rollout``."""
    agent = bp.agent
    spec = agent.spec
    if bf16 is None:  # bf16 network calls on CUDA: rollout noise dwarfs the rounding
        bf16 = torch.device(device).type == "cuda"
    R = len(rollouts)
    k = len(strategies)
    tables = (
        load_strength(agent.features.strength_tables) if agent.features.strength_tables else None
    )
    cache = StrengthCache(tables, device) if tables is not None else None
    combos = ALL_COMBOS.to(device)
    env = make_env(rollouts, spec, game_config, device)
    boards = torch.tensor([ro.board for ro in rollouts], dtype=torch.long, device=device)
    chooser = torch.tensor([ro.chooser for ro in rollouts], dtype=torch.long, device=device)
    # combos that hit each rollout's full board have zero weight everywhere
    valid = ~(combos[None, :, :, None] == boards[:, None, None, :]).any(-1).any(-1)  # [R, 1326]
    w_ch = torch.ones(R, k, NUM_COMBOS, device=device)
    w_nc = torch.ones(R, NUM_COMBOS, device=device)
    bias_cls = [CLASSES.get(s) for s in strategies]  # None = the unbiased blueprint
    gen = torch.Generator(device=device).manual_seed(int(seed))
    concrete = env.tab.concrete  # [4, A] concrete kind of every abstract index
    while not bool(env.done.all()):
        live = (~env.done).nonzero().squeeze(1)
        obs = env.obs(**agent.features.obs_kwargs())
        feats = features_from_obs(obs)
        legal_all = env.legal_mask().float()
        street_all = env.street.clamp(0, 3)
        actions = env.tab.call_index[street_all].clone()
        for lo in range(0, live.numel(), slots_per_call):
            idx = live[lo : lo + slots_per_call]
            m = idx.numel()
            P = slot_policies(agent, env, idx, feats, boards, cache, bf16)
            legal = legal_all[idx]  # [m, A]
            seat = env.actor[idx]
            is_ch = seat == chooser[idx]
            cls = torch.where(legal > 0, concrete[street_all[idx]], -1)  # [m, A]
            Pk = []
            for c in bias_cls:
                if c is None:
                    Pk.append(P)
                    continue
                mult = torch.where(cls == c, bias, 1.0)[:, None]
                Q = P * mult
                Pk.append(Q / Q.sum(-1, keepdim=True).clamp(min=1e-30))
            Pk = torch.stack(Pk, 1)  # [m, k, 1326, A]
            mix = torch.where(is_ch[:, None, None], Pk.mean(1), P)
            vm = valid[idx].float()[..., None]
            avg = (mix * vm).sum(1) / vm.sum(1).clamp(min=1)
            q = ((1 - explore) * avg + explore * legal / legal.sum(-1, keepdim=True)) * legal
            q = q / q.sum(-1, keepdim=True)
            a = torch.multinomial(q, 1, generator=gen).squeeze(1)  # [m]
            qa = q.gather(1, a[:, None]).squeeze(1)  # [m]
            ar = a[:, None, None, None]
            pk_a = Pk.gather(3, ar.expand(m, k, NUM_COMBOS, 1)).squeeze(3)  # [m, k, 1326]
            p_a = P.gather(2, a[:, None, None].expand(m, NUM_COMBOS, 1)).squeeze(2)  # [m, 1326]
            ch = is_ch.nonzero().squeeze(1)
            nc = (~is_ch).nonzero().squeeze(1)
            if ch.numel():
                w_ch[idx[ch]] *= pk_a[ch] / qa[ch][:, None, None]
            if nc.numel():
                w_nc[idx[nc]] *= p_a[nc] / qa[nc][:, None]
            actions[idx] = a
        env.step(actions)
    folded = env.folded.cpu().tolist()
    contrib = env.contrib.cpu().tolist()
    kind, folder, amount = [], [], []
    for fl, c in zip(folded, contrib, strict=True):
        if any(fl):
            f_ = fl.index(True)
            kind.append(0)
            folder.append(f_)
            amount.append(int(c[f_]))
        else:
            kind.append(1)
            folder.append(-1)
            amount.append(int(min(c)))
    return RolloutResult(w_ch, w_nc, kind, folder, amount)

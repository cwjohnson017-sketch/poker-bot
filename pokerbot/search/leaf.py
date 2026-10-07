"""Depth-limit leaf values from blueprint rollouts (Pluribus-style).

At a depth-limit leaf the leaf chooser (the searcher's opponent) picks one of
``k`` continuation strategies for the rest of the hand: the blueprint, or the
blueprint with the probability of fold, call or raise multiplied by ``bias``
and renormalised. The other player continues with the unbiased blueprint.
The solver treats the choice as a decision node, so in the limit the chooser
takes, for every combo, the best of the ``k`` continuations.

The value of continuation ``j`` is a linear operator on the opponent's reach
and is estimated by importance-weighted rollouts over public action
sequences:

* sample the remaining board uniformly from the cards not on the board;
* at every rollout decision sample an abstract action ``a`` from a
  combo-independent proposal ``q`` (the combo-averaged policy of the actor,
  mixed with ``explore`` uniform), and multiply each combo's weight by
  ``sigma(c, a) / q(a)`` (for the chooser separately per continuation);
* at the end, the fold or showdown payoff on the sampled board is evaluated
  range-vs-range with the vectorised kernels.

For a pair of hands ``(c, c')`` the sampled board is disjoint from both with
the same probability ``N_k / |All|`` for every pair, so each rollout is
scaled by ``|All| / (R * N_k)`` and the estimate is unbiased:

    v_i(c) = scale * sum_r w_i,r(c) * K_r(w_-i,r * pi_-i)(c)

where ``K_r`` is the fold or showdown kernel of rollout ``r``. With a
card-independent blueprint (uniform) the sampled betting paths are shared by
all leaves that come from the same skeleton leaf.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from . import vec_rollouts
from .abstract import CHECK_CALL, FOLD, RAISE, CardView, contributions, legal_options, to_action
from .blueprint import normalise_combos, policy_matrix
from .combos import NUM_COMBOS, valid_mask
from .showdown import ShowdownTables, fold_values
from .tree import CONTINUATION, LEAF

STRATEGY_CLASSES = {"blueprint": None, "fold": FOLD, "call": CHECK_CALL, "raise": RAISE}


@dataclass
class LeafConfig:
    strategies: list[str] = field(default_factory=lambda: ["blueprint", "fold", "call", "raise"])
    bias: float = 5.0
    rollouts: int = 4  # per leaf
    max_total_rollouts: int = 8000  # per solve; rollouts per leaf shrink (to >= 1) above this
    explore: float = 0.25  # uniform mixing in the action proposal
    seed: int = 0
    # "rollouts" (above) or "value_net": a value net at turn-end leaves
    # (pokerbot.search.value_leaf); flop solves need depth_streets >= 1
    mode: str = "rollouts"
    # value_net: checkpoint path; a river net (48 rows per leaf) or a turn-end net
    # (one row per leaf), by the checkpoint's meta kind (turn_net.load_leaf_predictor)
    net: str | None = None
    net_every: int = 1  # value_net: run the net every n regret updates per player (1 = exact)
    # value_net: a second net for trees rooted on the turn (a turn-end or river net), loaded
    # lazily; needed when leaf.net is a turn-start net (flop solves with depth_streets 0)
    # and turn solves have leaves (tree.depth_streets_turn 0). None: leaf.net for every tree
    turn_net: str | None = None


def runout_scale(board_len: int) -> float:
    """``|All| / N_k`` for completing a board of ``board_len`` cards."""
    k = 5 - board_len
    return math.comb(52 - board_len, k) / math.comb(52 - board_len - 4, k)


class RolloutSet:
    """Rollout rows grouped by *target* (a continuation node, or a gadget root).

    ``values(player, opp_reach)`` returns the ``[T, 1326]`` counterfactual
    values of ``player`` at every target given the opponent's reach at the
    target, ``opp_reach[T, 1326]``.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.num_targets = 0
        self._rows: dict[str, list] = {
            k: [] for k in ("target", "kind", "folder", "amount", "board", "w0", "w1", "scale")
        }
        self._weights: list[torch.Tensor] = []
        self._board_ids: dict[tuple, int] = {}
        self._finalised = False

    def add_weight(self, w: torch.Tensor) -> int:
        self._weights.append(w)
        return len(self._weights) - 1

    def add_row(self, target, kind, folder, amount, board, w0, w1, scale) -> None:
        b = tuple(board)
        bid = self._board_ids.setdefault(b, len(self._board_ids))
        for k, v in zip(
            ("target", "kind", "folder", "amount", "board", "w0", "w1", "scale"),
            (target, kind, folder, amount, bid, w0, w1, scale),
            strict=True,
        ):
            self._rows[k].append(v)

    def finalise(self) -> RolloutSet:
        dev = self.device
        r = self._rows
        self.target = torch.tensor(r["target"], dtype=torch.long, device=dev)
        self.kind = torch.tensor(r["kind"], dtype=torch.long, device=dev)
        self.folder = torch.tensor(r["folder"], dtype=torch.long, device=dev)
        self.amount = torch.tensor(r["amount"], dtype=torch.float32, device=dev)
        self.board = torch.tensor(r["board"], dtype=torch.long, device=dev)
        self.w = torch.stack([self.target.new_tensor(r["w0"]), self.target.new_tensor(r["w1"])])
        self.scale = torch.tensor(r["scale"], dtype=torch.float32, device=dev)
        boards = [None] * len(self._board_ids)
        for b, i in self._board_ids.items():
            boards[i] = b
        self.tables = ShowdownTables(boards, dev)
        if self._weights:
            self.W = torch.stack(
                [w.to(dev, torch.float32).expand(NUM_COMBOS) for w in self._weights]
            )
        else:
            self.W = torch.ones(1, NUM_COMBOS, device=dev)
        self._weights = []
        self._finalised = True
        return self

    @property
    def num_rows(self) -> int:
        return int(self.target.shape[0])

    def values(self, player: int, opp_reach: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        out = torch.zeros(self.num_targets, NUM_COMBOS, device=self.device, dtype=opp_reach.dtype)
        n = self.num_rows
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            tgt = self.target[s:e]
            inp = opp_reach[tgt] * self.W[self.w[1 - player, s:e]]
            bid = self.board[s:e]
            kind = self.kind[s:e]
            res = torch.empty_like(inp)
            fold = kind == 0
            if bool(fold.any()):
                fi = fold.nonzero().flatten()
                res[fi] = fold_values(inp[fi], self.tables.valid[bid[fi]])
            if bool((~fold).any()):
                si = (~fold).nonzero().flatten()
                res[si] = self.tables.showdown(inp[si], bid[si])
            coef = torch.where(
                fold & (self.folder[s:e] == player), -self.amount[s:e], self.amount[s:e]
            )
            res = res * (self.W[self.w[player, s:e]] * (coef * self.scale[s:e])[:, None])
            out.index_add_(0, tgt, res)
        return out


def _biased(P: torch.Tensor, classes: torch.Tensor, strategies: list[str], bias: float):
    """``[k, C|1, A]`` biased versions of ``P`` (``classes[A]`` = concrete kind)."""
    out = []
    for name in strategies:
        cls = STRATEGY_CLASSES[name]
        if cls is None:
            out.append(P)
            continue
        m = torch.where(classes == cls, bias, 1.0).to(P.dtype)
        Q = P * m
        out.append(Q / Q.sum(-1, keepdim=True).clamp(min=1e-30))
    return torch.stack(out)


def _rollout(
    state: Any,
    full_board: list[int],
    bp: Any,
    chooser: int,
    strategies: list[str],
    cfg: LeafConfig,
    game_config: Any,
    engine: Any,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """One rollout from ``state``. Returns (chooser weights ``[k, C|1]``,
    other-player weights ``[C|1]``, kind 0 fold / 1 showdown, folder, amount)."""
    k = len(strategies)
    w_ch = torch.ones(k, 1)
    w_nc = torch.ones(1)
    s = state.clone()
    ok = valid_mask(full_board)
    A = bp.spec.num_actions
    # Blueprints with per-combo own reach (neural SD-CFR) can carry it along the
    # rollout instead of replaying the history at every decision: same values.
    incremental = hasattr(bp, "policy_combos_nets")
    log_reach: dict[int, Any] = {}
    nets = None
    while not s.is_terminal:
        p = int(s.current_player)
        opts = legal_options(s, bp.spec)
        classes = torch.full((A,), -1, dtype=torch.long)
        legal = torch.zeros(A)
        for o in opts:
            classes[o.index] = o.kind
            legal[o.index] = 1.0
        view = CardView(s, full_board)
        if incremental:
            if p not in log_reach:  # the first rollout decision of p: reach of the leaf
                log_reach[p] = bp.log_reach_combos(view, p)
            avg, nets = bp.policy_combos_nets(view, p, log_reach[p])
            P = normalise_combos(avg, legal)
        else:
            P = policy_matrix(bp, view, p)  # [C|1, A]
        if p == chooser:
            Pk = _biased(P, classes, strategies, cfg.bias)  # [k, C|1, A]
            mix = Pk.mean(0)
        else:
            Pk = None
            mix = P
        rows = mix if mix.shape[0] == 1 else mix[ok]
        q = (1 - cfg.explore) * rows.mean(0) + cfg.explore * legal / legal.sum()
        q = q * legal
        q = q / q.sum()
        qq = q.double().numpy()
        a = int(rng.choice(A, p=qq / qq.sum()))
        if Pk is not None:
            w_ch = w_ch * (Pk[:, :, a] / q[a])
        else:
            w_nc = w_nc * (P[:, a] / q[a])
        if incremental and log_reach[p] is not None:
            log_reach[p] = log_reach[p] + torch.log(nets[:, :, a].clamp(min=0))
        opt = next(o for o in opts if o.index == a)
        s.apply(to_action(engine, opt.kind, opt.amount))
    c = contributions(s, game_config)
    folded = list(s.folded)
    if any(folded):
        f = folded.index(True)
        return w_ch, w_nc, 0, f, c[f]
    return w_ch, w_nc, 1, -1, min(c)


def build_leaf_rollouts(
    tree: Any,
    bp: Any,
    game_config: Any,
    cfg: LeafConfig,
    engine: Any,
    device: torch.device,
) -> RolloutSet:
    """Rollouts for every CONTINUATION node of ``tree`` (targets are indexed in
    the order of ``tree`` CONTINUATION node ids)."""

    rs = RolloutSet(device)
    leaves = (tree.kind == LEAF).nonzero().flatten().tolist()
    conts = (tree.kind == CONTINUATION).nonzero().flatten().tolist()
    cont_target = {c: i for i, c in enumerate(conts)}
    rs.num_targets = len(conts)
    rs.target_nodes = torch.tensor(conts, dtype=torch.long, device=device)
    if not leaves:
        return rs.finalise()
    k = len(cfg.strategies)
    R = max(1, min(int(cfg.rollouts), int(cfg.max_total_rollouts) // max(1, len(leaves))))
    rng = np.random.default_rng(cfg.seed)
    shared = bool(getattr(bp, "card_independent", False))
    path_cache: dict[tuple, tuple] = {}
    if vec_rollouts.supports(bp):  # every rollout at once on VecNLHE
        specs, meta = [], []
        for leaf in leaves:
            state = tree.states[leaf]
            board = list(tree.boards[int(tree.board_id[leaf])])
            chooser = int(tree.actor[leaf])
            first = int(tree.first_child[leaf])
            if int(tree.num_children[leaf]) != k:
                raise ValueError("tree continuations do not match the leaf config")
            scale = runout_scale(len(board)) / R
            avail = [c for c in range(52) if c not in board]
            for _ in range(R):
                runout = rng.choice(avail, size=5 - len(board), replace=False).tolist()
                full = board + [int(c) for c in runout]
                specs.append(vec_rollouts.Rollout(state, full, chooser))
                meta.append((first, chooser, full, scale))
        out = vec_rollouts.run_rollouts(
            specs, bp, game_config, cfg.strategies, cfg.bias, cfg.explore, device, cfg.seed
        )
        for r, (first, chooser, full, scale) in enumerate(meta):
            nc_id = rs.add_weight(out.w_nc[r])
            ch_ids = [rs.add_weight(out.w_ch[r, j]) for j in range(k)]
            for j in range(k):
                w0, w1 = (ch_ids[j], nc_id) if chooser == 0 else (nc_id, ch_ids[j])
                rs.add_row(
                    cont_target[first + j],
                    out.kind[r],
                    out.folder[r],
                    out.amount[r],
                    full,
                    w0,
                    w1,
                    scale,
                )
        return rs.finalise()
    for leaf in leaves:
        state = tree.states[leaf]
        board = list(tree.boards[int(tree.board_id[leaf])])
        chooser = int(tree.actor[leaf])
        first = int(tree.first_child[leaf])
        nk = int(tree.num_children[leaf])
        if nk != k:
            raise ValueError("tree continuations do not match the leaf config")
        scale = runout_scale(len(board)) / R
        avail = [c for c in range(52) if c not in board]
        for r in range(R):
            runout = rng.choice(avail, size=5 - len(board), replace=False).tolist()
            full = board + [int(c) for c in runout]
            key = (id(state), r)
            if shared and key in path_cache:
                res, nc_id, ch_ids = path_cache[key]
            else:
                res = _rollout(
                    state, full, bp, chooser, cfg.strategies, cfg, game_config, engine, rng
                )
                nc_id = rs.add_weight(res[1])
                ch_ids = [rs.add_weight(res[0][j]) for j in range(k)]
                if shared:
                    path_cache[key] = (res, nc_id, ch_ids)
            _w_ch, _w_nc, kind, folder, amount = res
            for j in range(k):
                w0, w1 = (ch_ids[j], nc_id) if chooser == 0 else (nc_id, ch_ids[j])
                rs.add_row(cont_target[first + j], kind, folder, amount, full, w0, w1, scale)
    return rs.finalise()


def build_root_rollouts(
    state: Any,
    board: list[int],
    bp: Any,
    game_config: Any,
    rollouts: int,
    engine: Any,
    device: torch.device,
    seed: int = 0,
) -> RolloutSet:
    """Blueprint-vs-blueprint rollouts from one public state (single target);
    used for the gadget's terminate values when no earlier solve exists."""
    cfg = LeafConfig(strategies=["blueprint"], rollouts=rollouts, seed=seed)
    rs = RolloutSet(device)
    rs.num_targets = 1
    rng = np.random.default_rng(seed)
    avail = [c for c in range(52) if c not in board]
    scale = runout_scale(len(board)) / rollouts
    if vec_rollouts.supports(bp):
        fulls = [
            list(board) + [int(c) for c in rng.choice(avail, size=5 - len(board), replace=False)]
            for _ in range(rollouts)
        ]
        out = vec_rollouts.run_rollouts(
            [vec_rollouts.Rollout(state, f, 0) for f in fulls],
            bp,
            game_config,
            cfg.strategies,
            cfg.bias,
            cfg.explore,
            device,
            seed,
        )
        for r, full in enumerate(fulls):
            i0 = rs.add_weight(out.w_ch[r, 0])
            i1 = rs.add_weight(out.w_nc[r])
            rs.add_row(0, out.kind[r], out.folder[r], out.amount[r], full, i0, i1, scale)
        return rs.finalise()
    for _ in range(rollouts):
        runout = rng.choice(avail, size=5 - len(board), replace=False).tolist()
        full = list(board) + [int(c) for c in runout]
        w_ch, w_nc, kind, folder, amount = _rollout(
            state, full, bp, 0, cfg.strategies, cfg, game_config, engine, rng
        )
        i0 = rs.add_weight(w_ch[0])
        i1 = rs.add_weight(w_nc)
        rs.add_row(0, kind, folder, amount, full, i0, i1, scale)
    return rs.finalise()

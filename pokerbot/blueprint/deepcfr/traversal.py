"""Batched external-sampling traversal on ``VecNLHE`` (DESIGN.md 5.5).

**Frontier scheme.** ``K`` root hands are dealt; the frontier is the set of
live env slots. One frontier step looks at every live slot at once:

* the traverser acts and the slot may branch: it becomes a *node* and is
  replicated (``env.select``) once per legal abstract action, each copy then
  steps with its action. Every copy remembers its *owner edge*
  ``node_id * A + action``;
* the opponent acts: one action is sampled from the opponent's current
  policy (regret matching on its advantage network);
* the traverser acts but may not branch (see the caps below): the action is
  sampled from the traverser's own current policy (outcome sampling for the
  rest of that hand).

Chance is sampled once per root (the whole deal is fixed at the root and
shared by every copy), which keeps the estimator unbiased and correlates the
branches of a node. The only Python loop is over frontier steps; every step
does one observation encode, at most one network batch per player, one
``select`` for the fan-out, one ``step`` and one ``select`` to drop finished
slots.

**Value backup.** A finished slot's traverser payoff (chips / ``value_scale``)
is the value of its owner edge. A slot that reaches a new branching node
hands that node's value to its owner edge instead. Each edge therefore gets
exactly one value. Nodes are created in frontier-step order and every child
node is created strictly after its parent, so one pass over the steps in
reverse computes ``v(node) = sum_a pi(a) * v(node, a)`` and writes it into
the parent edge. The instantaneous regret sample of a node is
``r(a) = v(node, a) - v(node)`` for legal ``a`` and 0 otherwise.

**Caps.** ``max_depth`` bounds the number of branching nodes on any path;
``max_frontier_nodes`` bounds the live frontier (when a step's fan-out would
exceed it, the branching slots in slot order take the budget and the rest are
cut). A cut slot, and a slot past ``max_depth``, is *rolled out*: both
players sample from their current policies until the hand ends, and the
payoff is used as the leaf value of its owner edge. That leaf is an unbiased
sample of the edge's counterfactual value under the current strategies, so
regrets stay unbiased (with more variance); rolled-out slots record no
samples. After ``max_steps`` frontier steps every remaining slot is cut.
Memory: live slots <= ``max_frontier_nodes``; recorded nodes <=
``max_steps * max_frontier_nodes`` (in practice far fewer).

**Variance reduction** (both off by default, both unbiased). The deal of a
root is fixed, so the traverser's equity against the opponent's actual hand
``E_s`` on each street ``s`` is computed once per root (exact on the flop,
turn and river, Monte Carlo preflop; :func:`~pokerbot.env.equity.street_equities`).

* ``allin_equity``: a hand that ends in a called all-in before the river is
  scored ``stake * (2 E_s - 1)`` (the expectation over the undealt board)
  instead of the pre-dealt runout.
* ``chance_cv = beta``: when a slot moves to a new street with ``c`` chips in
  per player, its value gets ``-beta * 2c * (E_new - E_old)``, a check-down
  control variate with zero mean given the history. A slot carries these
  corrections until it branches or ends, and they are added to the value of
  the edge it reports to.

:func:`rollout` (play env slots to the end with a policy per seat) and
:func:`actor_probs` are the reusable pieces for the search component.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from ...env.actions import DEFAULT_SPEC, ActionSpec
from ...env.cards import make_generator
from ...env.config import GameConfig
from ...env.equity import street_equities
from ...env.vec_env import VecNLHE
from .features import FeatureConfig, features_from_obs, index_features
from .networks import regret_matching

PolicyFn = Callable[[dict[str, torch.Tensor]], torch.Tensor]
"""Maps a feature dict ``[n, ...]`` to action probabilities ``[n, A]`` (0 on illegal)."""

_ROLLOUT_STEPS = 256  # safety bound on the steps after max_steps (hands are far shorter)


def uniform_policy(feats: dict[str, torch.Tensor]) -> torch.Tensor:
    lf = feats["legal"].float()
    return lf / lf.sum(1, keepdim=True).clamp(min=1)


class NetPolicy:
    """Regret-matching policy of an advantage network (uniform when ``net`` is None).

    Evaluates in chunks of ``chunk`` rows under autocast ``amp_dtype`` (None =
    fp32) so a 256k-slot frontier does not materialize one huge activation.
    """

    def __init__(
        self,
        net: torch.nn.Module | None,
        fallback: str = "uniform",
        amp_dtype: torch.dtype | None = None,
        chunk: int = 65536,
    ) -> None:
        self.net = net
        self.fallback = fallback
        self.amp_dtype = amp_dtype
        self.chunk = int(chunk)

    @torch.no_grad()
    def advantages(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        legal = feats["legal"]
        n = legal.shape[0]
        if self.net is None or n == 0:
            return torch.zeros(legal.shape, dtype=torch.float32, device=legal.device)
        dev = legal.device
        outs = []
        for lo in range(0, n, self.chunk):
            sub = {k: v[lo : lo + self.chunk] for k, v in feats.items()}
            with torch.autocast(
                device_type=dev.type,
                dtype=self.amp_dtype or torch.float32,
                enabled=self.amp_dtype is not None,
            ):
                outs.append(self.net(sub).float())
        return torch.cat(outs, 0)

    @torch.no_grad()
    def __call__(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.net is None:
            return uniform_policy(feats)
        return regret_matching(self.advantages(feats), feats["legal"], self.fallback)


@torch.no_grad()
def actor_probs(
    feats: dict[str, torch.Tensor], actor: torch.Tensor, policies: Sequence[PolicyFn]
) -> torch.Tensor:
    """``[n, A]`` probabilities: each row from the policy of the seat to act.

    One batched call per seat. Rows with no actor (finished slots) get the
    uniform legal policy (check/call only).
    """
    probs = uniform_policy(feats)
    for seat in (0, 1):
        idx = (actor == seat).nonzero().squeeze(1)
        if idx.numel():
            probs[idx] = policies[seat](index_features(feats, idx)).float()
    return probs


def sample_actions(probs: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    return torch.multinomial(probs, 1, generator=generator).squeeze(1)


@torch.no_grad()
def rollout(
    env: VecNLHE,
    policies: Sequence[PolicyFn],
    generator: torch.Generator | None = None,
    obs_kwargs: dict[str, Any] | None = None,
    max_steps: int = 256,
) -> torch.Tensor:
    """Play every live slot of ``env`` to the end (in place), each seat sampling
    from its policy. Returns ``payoffs [n, 2]`` (long chips)."""
    obs_kwargs = obs_kwargs or {}
    for _ in range(max_steps):
        live = (~env.done).nonzero().squeeze(1)
        if live.numel() == 0:
            break
        feats = features_from_obs(env.obs(generator=generator, **obs_kwargs))
        sub = index_features(feats, live)
        probs = actor_probs(sub, env.actor[live], policies)
        a = env.tab.call_index[env.street.clamp(0, 3)].clone()
        a[live] = sample_actions(probs, generator)
        env.step(a)
    else:
        if not bool(env.done.all()):
            raise RuntimeError("rollout did not finish within max_steps")
    return env.payoffs.clone()


def deal_env(
    decks: torch.Tensor | Sequence[Sequence[int]],
    buttons: torch.Tensor | Sequence[int],
    game_config: Any = None,
    spec: ActionSpec = DEFAULT_SPEC,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> VecNLHE:
    """An env whose slots are dealt from the given 52-card decks and buttons
    (root hands for tests and for search). Uses the env's internal deal path."""
    decks = torch.as_tensor(decks, dtype=torch.long, device=device)
    buttons = torch.as_tensor(buttons, dtype=torch.long, device=device)
    k = decks.shape[0]
    env = VecNLHE(k, game_config, device, seed, spec, validate=False, auto_deal=False)
    env._deal(torch.arange(k, device=env.device), decks, buttons)
    return env


@dataclass
class TraversalConfig:
    max_frontier_nodes: int = 131072
    max_depth: int = 64  # branching (traverser) nodes per path
    max_steps: int = 64  # frontier steps before the remaining slots are rolled out
    value_scale: float | None = None  # chips per value unit; default: big blind
    record_strategy: bool = False  # opponent (infoset, policy) samples for a strategy memory
    allin_equity: bool = False  # score all-ins called before the river by their equity
    chance_cv: float = 0.0  # beta of the street-change control variate (0 = off)
    preflop_equity_samples: int = 1024  # Monte Carlo runouts for the preflop equity
    features: FeatureConfig = field(default_factory=FeatureConfig)

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> TraversalConfig:
        d = dict(d or {})
        if "features" in d and not isinstance(d["features"], FeatureConfig):
            d["features"] = FeatureConfig.from_dict(d["features"])
        return TraversalConfig(**d)


@dataclass
class TraversalResult:
    """Output of one batch of traversals for one traverser.

    ``samples``: advantage-memory rows (compact dtypes, on the env device):
    ``cards, hist, hist_amt, scalars, legal, target`` (regrets in value
    units) and ``iteration``. ``strategy``: opponent rows with ``target`` =
    the opponent's policy, or None. ``node_*`` describe every branching node
    in creation order (for tests and diagnostics); ``root_value [K]`` is the
    traverser's backed-up value of each root hand. ``root_equity [K, 4]`` is
    the traverser's equity per street against the opponent's hand when
    ``allin_equity`` or ``chance_cv`` is on, else None.
    """

    samples: dict[str, torch.Tensor]
    strategy: dict[str, torch.Tensor] | None
    node_value: torch.Tensor
    node_parent: torch.Tensor
    node_root: torch.Tensor
    edge_value: torch.Tensor
    root_value: torch.Tensor
    stats: dict[str, float]
    root_equity: torch.Tensor | None = None


def _compact(feats: dict[str, torch.Tensor], vocab: int) -> dict[str, torch.Tensor]:
    return {
        "cards": feats["cards"].to(torch.uint8),
        "hist": feats["hist"].to(torch.uint8 if vocab <= 256 else torch.int16),
        "hist_amt": feats["hist_amt"].half(),
        "scalars": feats["scalars"].half(),
        "legal": feats["legal"].bool(),
    }


class FrontierTraverser:
    """External-sampling traversals for one player at a time, batched on ``VecNLHE``."""

    def __init__(
        self,
        game_config: Any = None,
        spec: ActionSpec = DEFAULT_SPEC,
        cfg: TraversalConfig | None = None,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        self.game_config = game_config if game_config is not None else GameConfig()
        self.spec = spec
        self.cfg = cfg or TraversalConfig()
        self.device = torch.device(device)
        self.seed = int(seed)
        self.generator = make_generator(seed * 2 + 1, self.device)
        self._roots: VecNLHE | None = None
        self.value_scale = float(self.cfg.value_scale or self.game_config.big_blind)

    def new_roots(self, k: int) -> VecNLHE:
        """``k`` freshly dealt hands (buttons alternate per slot and per call)."""
        if self._roots is None or self._roots.n != k:
            self._roots = VecNLHE(
                k, self.game_config, self.device, self.seed * 2, self.spec, validate=False
            )
        else:
            self._roots.reset()
        return self._roots

    @torch.no_grad()
    def traverse(
        self,
        traverser: int,
        policies: Sequence[PolicyFn],
        iteration: int,
        roots: int | VecNLHE,
    ) -> TraversalResult:
        """Run one batch of traversals for ``traverser`` (seat 0 or 1).

        ``policies[s]`` is seat ``s``'s current policy: the traverser's own
        policy weighs its node values and drives its rollouts, the opponent's
        is sampled. ``roots`` is a number of hands to deal or an env whose
        slots are the root hands (it is not modified).
        """
        cfg = self.cfg
        p = int(traverser)
        gen = self.generator
        env = self.new_roots(roots) if isinstance(roots, int) else roots
        dev = env.device
        K, A = env.n, env.num_actions
        vocab = env.vocab_size
        scale = self.value_scale
        obs_kwargs = cfg.features.obs_kwargs()

        use_cv = cfg.chance_cv != 0.0
        use_eq = cfg.allin_equity or use_cv
        root_eq = None  # [K, 4] traverser equity vs the opponent's hand per street
        if use_eq:
            c9 = env.cards
            hero, opp = (c9[:, 0:2], c9[:, 2:4]) if p == 0 else (c9[:, 2:4], c9[:, 0:2])
            root_eq = street_equities(hero, opp, c9[:, 4:9], cfg.preflop_equity_samples, gen)

        root_value = torch.zeros(K, dtype=torch.float32, device=dev)
        pre_done = env.done.clone()
        root_value[pre_done] = env.payoffs[pre_done, p].float() / scale
        if cfg.allin_equity and bool(pre_done.any()):
            # both all-in from the blinds: nothing to decide, score the runout by equity
            i = pre_done.nonzero().squeeze(1)
            stake = env.contrib[i].min(1).values.float()
            root_value[i] = stake * (2 * root_eq[i, 0] - 1) / scale
        live0 = (~pre_done).nonzero().squeeze(1)
        env = env.select(live0)
        n = env.n
        owner = torch.full((n,), -1, dtype=torch.long, device=dev)
        root = live0
        cut = torch.zeros(n, dtype=torch.bool, device=dev)
        depth = torch.zeros(n, dtype=torch.long, device=dev)
        # chance control-variate corrections since the slot's last branching node
        corr = torch.zeros(n, dtype=torch.float32, device=dev)

        node_feats: list[dict[str, torch.Tensor]] = []
        node_pol: list[torch.Tensor] = []
        node_parent: list[torch.Tensor] = []
        node_root: list[torch.Tensor] = []
        node_corr: list[torch.Tensor] = []
        counts: list[int] = []
        strat: list[dict[str, torch.Tensor]] = []
        leaf_edge: list[torch.Tensor] = []
        leaf_val: list[torch.Tensor] = []
        rleaf_idx: list[torch.Tensor] = []
        rleaf_val: list[torch.Tensor] = []
        num_nodes = 0
        slot_steps = 0
        max_front = n
        num_cut = 0
        allin_leaves = 0
        steps = 0

        def record_leaves(own: torch.Tensor, rt: torch.Tensor, val: torch.Tensor) -> None:
            e = own >= 0
            leaf_edge.append(own[e])
            leaf_val.append(val[e])
            rleaf_idx.append(rt[~e])
            rleaf_val.append(val[~e])

        while env.n > 0:
            if steps >= cfg.max_steps:  # past max_steps: roll every slot out
                if steps >= cfg.max_steps + _ROLLOUT_STEPS:
                    raise RuntimeError("traversal did not finish its rollouts")
                cut = torch.ones_like(cut)
            n = env.n
            feats = features_from_obs(env.obs(generator=gen, **obs_kwargs))
            actor = env.actor
            probs = actor_probs(feats, actor, policies)
            is_trav = actor == p
            legal = feats["legal"]
            want = is_trav & ~cut & (depth < cfg.max_depth)
            extra = torch.where(want, legal.sum(1) - 1, 0)
            ok = want & (torch.cumsum(extra, 0) <= cfg.max_frontier_nodes - n)
            newly_cut = want & ~ok
            cut = cut | newly_cut
            e_idx = ok.nonzero().squeeze(1)
            m = e_idx.numel()
            num_cut += int(newly_cut.sum())
            if m:
                node_feats.append(_compact(index_features(feats, e_idx), vocab))
                node_pol.append(probs[e_idx])
                node_parent.append(owner[e_idx])
                node_root.append(root[e_idx])
                if use_cv:
                    node_corr.append(corr[e_idx])
            counts.append(m)
            if cfg.record_strategy:
                o_idx = (~is_trav & ~cut).nonzero().squeeze(1)
                if o_idx.numel():
                    row = _compact(index_features(feats, o_idx), vocab)
                    row["target"] = probs[o_idx].half()
                    strat.append(row)

            sampled = sample_actions(probs, gen)
            s_idx = (~ok).nonzero().squeeze(1)
            r, a = legal[e_idx].nonzero(as_tuple=True)
            child_slot = e_idx[r]
            new_idx = torch.cat([s_idx, child_slot])
            new_act = torch.cat([sampled[s_idx], a])
            owner = torch.cat([owner[s_idx], (num_nodes + r) * A + a])
            root = torch.cat([root[s_idx], root[child_slot]])
            cut = torch.cat([cut[s_idx], torch.zeros_like(r, dtype=torch.bool)])
            depth = torch.cat([depth[s_idx], depth[child_slot] + 1])
            corr = torch.cat([corr[s_idx], torch.zeros(r.numel(), dtype=corr.dtype, device=dev)])
            num_nodes += m

            env = env.select(new_idx)
            pre_street = env.street.clone() if use_eq else None
            pay, done = env.step(new_act)
            slot_steps += n
            max_front = max(max_front, env.n)
            if use_cv:
                # a new street was dealt: check-down control variate
                mv = (~done & (env.street != pre_street)).nonzero().squeeze(1)
                if mv.numel():
                    rt = root[mv]
                    d_eq = root_eq[rt, env.street[mv]] - root_eq[rt, pre_street[mv]]
                    c = env.contrib[mv, p].float()  # equal for both seats once a street closes
                    corr[mv] -= cfg.chance_cv * 2.0 * c * d_eq / scale
            d_idx = done.nonzero().squeeze(1)
            if d_idx.numel():
                val = pay[d_idx, p].float() / scale
                if cfg.allin_equity:
                    ps = pre_street[d_idx]
                    ro = ~env.folded[d_idx].any(1) & (ps < 3)  # all-in called before the river
                    j = d_idx[ro]
                    if j.numel():
                        stake = env.contrib[j].min(1).values.float()
                        val[ro] = stake * (2 * root_eq[root[j], ps[ro]] - 1) / scale
                        allin_leaves += int(j.numel())
                if use_cv:
                    val = val + corr[d_idx]
                record_leaves(owner[d_idx], root[d_idx], val)
            keep = (~done).nonzero().squeeze(1)
            if keep.numel() < env.n:
                env = env.select(keep)
                owner, root, cut, depth = owner[keep], root[keep], cut[keep], depth[keep]
                corr = corr[keep]
            steps += 1

        # ------------------------------------------------------------ backup
        N = num_nodes
        pol = torch.cat(node_pol, 0) if N else torch.zeros(0, A, dtype=torch.float32, device=dev)
        parent = torch.cat(node_parent) if N else torch.zeros(0, dtype=torch.long, device=dev)
        nroot = torch.cat(node_root) if N else torch.zeros(0, dtype=torch.long, device=dev)
        ncorr = torch.cat(node_corr) if (N and use_cv) else None
        ev = torch.zeros(N * A, dtype=torch.float32, device=dev)
        if leaf_edge:
            ev[torch.cat(leaf_edge)] = torch.cat(leaf_val)
            rl = torch.cat(rleaf_idx)
            root_value[rl] = torch.cat(rleaf_val)
        ev2 = ev.view(N, A)
        nv = torch.zeros(N, dtype=torch.float32, device=dev)
        offs = [0]
        for c in counts:
            offs.append(offs[-1] + c)
        for s in range(len(counts) - 1, -1, -1):
            lo, hi = offs[s], offs[s + 1]
            if lo == hi:
                continue
            v = (pol[lo:hi] * ev2[lo:hi]).sum(1)
            nv[lo:hi] = v
            # the parent edge also gets the corrections between it and this node
            up = v + ncorr[lo:hi] if ncorr is not None else v
            par = parent[lo:hi]
            top = par < 0
            ev[par[~top]] = up[~top]
            root_value[nroot[lo:hi][top]] = up[top]

        if N:
            samples = {k: torch.cat([f[k] for f in node_feats], 0) for k in node_feats[0]}
            regret = (ev2 - nv[:, None]) * samples["legal"].float()
            samples["target"] = regret.half()
        else:
            T, S = env.history_len, cfg.features.num_scalars
            samples = {
                "cards": torch.zeros(0, 7, dtype=torch.uint8, device=dev),
                "hist": torch.zeros(0, T, dtype=torch.uint8, device=dev),
                "hist_amt": torch.zeros(0, T, dtype=torch.float16, device=dev),
                "scalars": torch.zeros(0, S, dtype=torch.float16, device=dev),
                "legal": torch.zeros(0, A, dtype=torch.bool, device=dev),
                "target": torch.zeros(0, A, dtype=torch.float16, device=dev),
            }
            regret = samples["target"].float()
        samples["iteration"] = torch.full(
            (N,), int(iteration), dtype=torch.int32, device=samples["target"].device
        )
        strategy = None
        if cfg.record_strategy and strat:
            strategy = {k: torch.cat([f[k] for f in strat], 0) for k in strat[0]}
            strategy["iteration"] = torch.full(
                (strategy["target"].shape[0],), int(iteration), dtype=torch.int32, device=dev
            )
        stats = {
            "roots": K,
            "nodes": N,
            "slot_steps": slot_steps,
            "frontier_steps": steps,
            "max_frontier": max_front,
            "cut_slots": num_cut,
            "allin_leaves": allin_leaves,
            "regret_abs_mean": float(regret.abs().sum() / samples["legal"].sum().clamp(min=1))
            if N
            else 0.0,
        }
        return TraversalResult(
            samples=samples,
            strategy=strategy,
            node_value=nv,
            node_parent=parent,
            node_root=nroot,
            edge_value=ev2,
            root_value=root_value,
            stats=stats,
            root_equity=root_eq,
        )

"""Translate a search strategy onto a richer tree rooted at the same public state.

:func:`~pokerbot.search.exact_eval.map_sigma` needs two trees with the same
decision nodes and child actions. :func:`translate_sigma` does not: it puts the
strategy of a solve on one tree (``src``, e.g. a 6,000-node flop tree with two
bet sizes) onto the decision nodes of another tree with the same root (``dst``,
e.g. the 20,000-node tree with seven), so both can be scored on one trunk.

**Node matching** (:func:`match_nodes`). Each ``dst`` node's history is
translated into ``src``'s tree by walking both trees from the root, one step
per ``dst`` edge:

* chance: the ``src`` child with the same card;
* betting while the histories still agree (every earlier step matched the
  concrete action, so both trees are in the same engine state): the ``src``
  child with the same concrete action, if there is one;
* otherwise (a size ``src`` lacks, or any action after a translated one):

  1. fold and check/call map to themselves (a fold to the check/call when
     there is no fold);
  2. a raise maps to the ``src`` child with the same abstract action, i.e. the
     same entry of the shared action spec such as ``("raise", 0.75, "open")``:
     the same kind and pot fraction, whatever the amount;
  3. an all-in maps to the all-in;
  4. any other raise uses the pseudo-harmonic rule (Ganzfried & Sandholm 2013,
     the formula of :func:`~pokerbot.search.abstract.map_concrete`) on pot
     fractions: the action's fraction in its own state against the target
     children's fractions in theirs. It is deterministic: the more likely side
     of the split (``u = 0.5``; the larger size on an exact tie, as the
     engine's ``translate``). Below the smallest size it takes the smallest,
     above the largest the largest. With no raise in ``src`` it takes the call.

Translated amounts change the pot, so later ``dst`` actions are matched in the
``src`` subtree reached so far by rules 1-4 (abstract action, then pot
fraction), never by amount. When the histories agree the walk is exact: the
same concrete actions, so ``src == dst`` gives the identity. A ``dst`` node is
**unmatched** when the walk reaches a ``src`` node of another kind or actor:
for example ``src`` maps a bet to all-in, the call of it ends the hand there,
and ``dst`` goes on to the turn.

**Strategy** (:func:`translate_sigma`). At a matched ``dst`` decision node,
each ``src`` child's probability goes to one ``dst`` child:

* histories agree: the ``dst`` child with the same concrete action, and ``dst``-only
  children get zero. ``src`` mass on an action ``dst`` lacks is **lost**:
  ``strict`` raises at the searcher's nodes. Otherwise that node's rows are
  renormalised and the lost mass is reported (always, at the opponent's nodes);
* after a translation: the ``src`` child translated onto ``dst``'s children by
  rules 1-4 (the reverse direction), summed when two land on one child.

Unmatched nodes play ``fallback`` (default: uniform). Only the betting part of
``tree.states`` (pot, street bets, legal raise range) is read, which does not
depend on the cards; chance children are matched by ``tree.deal_card``.

For re-searching as the agent does in play (:mod:`pokerbot.search.size_eval`),
:func:`offtree_edges` lists one player's first off-tree actions in ``dst``
(after a history ``src`` has exactly), :func:`subtree_nodes` the nodes below
one, and ``translate_sigma(..., nodes=...)`` translates only those.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any

import torch

from .abstract import CHECK_CALL, FOLD, RAISE, action_key, pot_fraction
from .combos import NUM_COMBOS
from .tree import CHANCE, DECISION, LEAF, SubgameTree

C = NUM_COMBOS
_WALK_KINDS = (DECISION, CHANCE, LEAF)  # dst nodes that can lead to decisions


# -- trees as Python lists ---------------------------------------------------------


@dataclass(frozen=True)
class _Kid:
    node: int
    kind: int  # FOLD / CHECK_CALL / RAISE
    amount: int  # raise-to for RAISE, else 0
    label: tuple | None  # the spec entry, e.g. ("raise", 0.75, "open"); None when off-tree


class _Info:
    """``SubgameTree`` fields as lists, with per-node child tables built lazily."""

    def __init__(self, tree: SubgameTree) -> None:
        self.tree = tree
        self.kind = tree.kind.tolist()
        self.parent = tree.parent.tolist()
        self.first = tree.first_child.tolist()
        self.nch = tree.num_children.tolist()
        self.akind = tree.action_kind.tolist()
        self.aamt = tree.action_amount.tolist()
        self.aabs = tree.action_abstract.tolist()
        self.street = tree.street.tolist()
        self.deal = tree.deal_card.tolist()
        self.actor = tree.actor.tolist()
        self.board = tree.board_id.tolist()
        self._kids: dict[int, list[_Kid]] = {}
        self._cards: dict[int, dict[int, int]] = {}

    def label(self, child: int) -> tuple | None:
        i = self.aabs[child]
        if i < 0:
            return None
        return tuple(self.tree.street_actions[self.street[self.parent[child]]][i])

    def kids(self, node: int) -> list[_Kid]:
        out = self._kids.get(node)
        if out is None:
            s = self.first[node]
            out = [
                _Kid(c, self.akind[c], self.aamt[c] if self.akind[c] == RAISE else 0, self.label(c))
                for c in range(s, s + self.nch[node])
            ]
            self._kids[node] = out
        return out

    def card_child(self, node: int, card: int) -> int | None:
        table = self._cards.get(node)
        if table is None:
            s = self.first[node]
            table = {self.deal[c]: c for c in range(s, s + self.nch[node])}
            self._cards[node] = table
        return table.get(card)


# -- one action onto another node's children ------------------------------------------


def _max_raise(state: Any) -> int:
    la = state.legal_actions()
    return int(la.max_raise_to) if int(la.min_raise_to) > 0 else -1


def harmonic_choice(x: float, fractions: list[float]) -> int:
    """Index into ``fractions`` (ascending pot fractions of the target raises)
    that the deterministic pseudo-harmonic rule picks for a raise of pot
    fraction ``x``: the side :func:`~pokerbot.search.abstract.map_concrete`
    gives more probability (``u = 0.5``; the larger one on an exact tie)."""
    if x <= fractions[0]:
        return 0
    if x >= fractions[-1]:
        return len(fractions) - 1
    k = next(i for i, f in enumerate(fractions) if f >= x)
    fa, fb = fractions[k - 1], fractions[k]
    if fb == x:
        return k
    if fb <= fa:
        return k - 1
    pa = (fb - x) * (1 + fa) / ((fb - fa) * (1 + x))
    return k - 1 if pa > 0.5 else k


def translate_action(
    kind: int,
    amount: int,
    label: tuple | None,
    from_state: Any,
    to_state: Any,
    to_kids: list[_Kid],
    aligned: bool,
) -> tuple[int | None, str]:
    """Index into ``to_kids`` of the child that the action ``(kind, amount)``
    (spec entry ``label``) taken in ``from_state`` maps to, and the rule used
    (``exact``, ``kind``, ``label``, ``allin``, ``harmonic``, ``to_call``; see the
    module docstring). ``aligned``: both states are the same, so concrete
    amounts are comparable. ``(None, "none")`` when nothing fits."""
    if aligned:
        for j, k in enumerate(to_kids):
            if k.kind == kind and k.amount == amount:
                return j, "exact"
    if kind in (FOLD, CHECK_CALL):
        for want in (FOLD, CHECK_CALL) if kind == FOLD else (CHECK_CALL,):
            for j, k in enumerate(to_kids):
                if k.kind == want:
                    return j, "kind"
        return None, "none"
    raises = [j for j, k in enumerate(to_kids) if k.kind == RAISE]
    if not raises:
        for j, k in enumerate(to_kids):
            if k.kind == CHECK_CALL:
                return j, "to_call"
        return None, "none"
    if label is not None:
        for j in raises:
            if to_kids[j].label == label:
                return j, "label"
    hi = _max_raise(from_state)
    if hi > 0 and amount >= hi:
        top = _max_raise(to_state)
        for j in raises:
            if to_kids[j].amount >= top:
                return j, "allin"
    x = pot_fraction(from_state, amount)
    fr = sorted((pot_fraction(to_state, to_kids[j].amount), to_kids[j].amount, j) for j in raises)
    return fr[harmonic_choice(x, [f for f, _, _ in fr])][2], "harmonic"


# -- node matching --------------------------------------------------------------------


@dataclass
class NodeMatch:
    """Per ``dst`` node: the matched ``src`` node (``-1``: unmatched, or a
    terminal ``dst`` node, which is never matched) and whether its history went
    through a translated (not exactly matched) action."""

    src_of: list[int]
    translated: list[bool]
    steps: dict[str, int] = field(default_factory=dict)  # walk steps per rule


def _history_keys(history: Any) -> list[tuple]:
    return [(int(s), int(p), *action_key(a)) for s, p, a in history]


def _same_root(st: SubgameTree, dt: SubgameTree) -> None:
    same = tuple(st.boards[0]) == tuple(dt.boards[0]) and _history_keys(
        st.root_history
    ) == _history_keys(dt.root_history)
    if not same:
        raise ValueError("src and dst trees are not rooted at the same public state")
    if int(st.kind[0]) != int(dt.kind[0]) or int(st.actor[0]) != int(dt.actor[0]):
        raise ValueError("src and dst roots differ (kind or actor)")


def match_nodes(src_tree: SubgameTree, dst_tree: SubgameTree) -> NodeMatch:
    """Match every ``dst`` decision, chance and leaf node to the ``src`` node of
    its translated history (see the module docstring)."""
    _same_root(src_tree, dst_tree)
    si, di = _Info(src_tree), _Info(dst_tree)
    N = dst_tree.num_nodes
    src_of = [-1] * N
    translated = [False] * N
    steps: Counter = Counter()
    src_of[0] = 0
    for n in range(1, N):
        kn = di.kind[n]
        if kn not in _WALK_KINDS:
            continue
        p = di.parent[n]
        s = src_of[p]
        if s < 0:
            continue
        pk = di.kind[p]
        rule = "exact"
        if pk == DECISION:
            j, rule = translate_action(
                di.akind[n],
                di.aamt[n] if di.akind[n] == RAISE else 0,
                di.label(n),
                dst_tree.states[p],
                src_tree.states[s],
                si.kids(s),
                not translated[p],
            )
            sc = None if j is None else si.kids(s)[j].node
        elif pk == CHANCE:
            sc = si.card_child(s, di.deal[n])
            rule = "chance"
        else:
            continue
        ok = sc is not None and si.kind[sc] == kn and si.actor[sc] == di.actor[n]
        if not ok:
            steps["unmatched"] += 1
            continue
        steps[rule] += 1
        src_of[n] = sc
        translated[n] = translated[p] or rule not in ("exact", "chance")
    return NodeMatch(src_of, translated, dict(steps))


# -- strategies -----------------------------------------------------------------------


@dataclass
class StrategySnapshot:
    """What :func:`translate_sigma` reads from a source solver (its tree, the
    decision-node order and a strategy), without the solver's tables."""

    tree: SubgameTree
    dec_nodes: torch.Tensor
    sigma: torch.Tensor  # [D, A, C]
    ranges: torch.Tensor  # [2, C] root ranges

    @classmethod
    def of(cls, solver: Any, device: torch.device | str = "cpu") -> StrategySnapshot:
        """The average strategy of ``solver``, moved to ``device``."""
        return cls(
            solver.tree,
            solver.dec_nodes.cpu(),
            solver.average_strategy().to(device),
            solver.ranges.to(device),
        )

    def average_strategy(self) -> torch.Tensor:
        return self.sigma


def _role(actor: int, searcher: int) -> str:
    return "searcher" if actor == searcher else "opponent"


def translate_sigma(
    src: Any,
    dst: Any,
    searcher: int,
    src_sigma: torch.Tensor | None = None,
    *,
    strict: bool = True,
    fallback: torch.Tensor | None = None,
    match: NodeMatch | None = None,
    nodes: Collection[int] | None = None,
    chunk: int = 8192,
) -> tuple[torch.Tensor, dict]:
    """``src``'s strategy (its average strategy unless ``src_sigma``) on
    ``dst``'s decision nodes, ``[D, A, C]`` in ``dst``'s order, and a report.

    ``src`` and ``dst`` are :class:`~pokerbot.search.solver.RangeSolver` s (or
    anything with ``tree`` and ``dec_nodes``; ``dst`` also needs ``uniform``, and
    ``src`` an ``average_strategy()`` when ``src_sigma`` is ``None``, e.g. a
    :class:`StrategySnapshot`) on trees rooted at the same public state.
    ``searcher`` is the player whose lost mass ``strict`` refuses. ``fallback``
    (``[D, A, C]`` on ``dst``, default uniform) is played at unmatched nodes and
    where a renormalised row has no mass left. With ``nodes`` (``dst`` node ids)
    only those decision nodes are translated and counted; the others play
    ``fallback``. See the module docstring.

    The report (JSON-ready): ``decision_nodes``, ``translated_nodes`` (matched
    nodes reached through at least one translated action), ``unmatched_nodes``
    (each split by role, ``searcher`` / ``opponent``), ``lost_mass`` per role
    (``nodes`` with any, ``max`` probability lost, ``mean`` over those nodes and
    all combos), ``fill`` (``src`` children placed per rule) and ``steps`` (the
    walk's steps per rule)."""
    sig = src.average_strategy() if src_sigma is None else src_sigma
    st, dt = src.tree, dst.tree
    m = match_nodes(st, dt) if match is None else match
    si, di = _Info(st), _Info(dt)
    s_index = {n: d for d, n in enumerate(src.dec_nodes.tolist())}
    uni = dst.uniform
    Dn, A = int(uni.shape[0]), int(uni.shape[1])
    dev, dtype = uni.device, sig.dtype
    fb = uni if fallback is None else fallback
    roles = ("searcher", "opponent")
    count = {k: dict.fromkeys(roles, 0) for k in ("decision", "translated", "unmatched")}
    fill: Counter = Counter()
    put_d: list[int] = []
    put_j: list[int] = []
    put_s: list[int] = []
    put_i: list[int] = []
    lost_d: list[int] = []
    lost_s: list[int] = []
    lost_i: list[int] = []
    unmatched: list[int] = []
    skipped: list[int] = []
    only = None if nodes is None else set(nodes)
    for d, n in enumerate(dst.dec_nodes.tolist()):
        if only is not None and n not in only:
            skipped.append(d)
            continue
        role = _role(di.actor[n], searcher)
        count["decision"][role] += 1
        s = m.src_of[n]
        if s < 0:
            count["unmatched"][role] += 1
            unmatched.append(d)
            continue
        ds = s_index[s]
        if m.translated[n]:
            count["translated"][role] += 1
        if di.kind[n] == LEAF:  # continuation choices: copy by slot
            for i in range(min(si.nch[s], di.nch[n])):
                put_d.append(d)
                put_j.append(i)
                put_s.append(ds)
                put_i.append(i)
            continue
        aligned = not m.translated[n]
        dkids, skids = di.kids(n), si.kids(s)
        for i, k in enumerate(skids):
            j, rule = translate_action(
                k.kind, k.amount, k.label, st.states[s], dt.states[n], dkids, aligned
            )
            if j is None or (aligned and rule != "exact"):
                fill["lost"] += 1
                lost_d.append(d)
                lost_s.append(ds)
                lost_i.append(i)
                continue
            fill[rule] += 1
            put_d.append(d)
            put_j.append(j)
            put_s.append(ds)
            put_i.append(i)
    out = torch.zeros(Dn, A, C, device=dev, dtype=dtype)

    def rows(ss: list[int], ii: list[int], a: int, b: int) -> torch.Tensor:
        s_t = torch.tensor(ss[a:b], device=sig.device)
        i_t = torch.tensor(ii[a:b], device=sig.device)
        return sig[s_t, i_t].to(dev, dtype)

    for a in range(0, len(put_d), chunk):
        b = a + chunk
        idx = (torch.tensor(put_d[a:b], device=dev), torch.tensor(put_j[a:b], device=dev))
        out.index_put_(idx, rows(put_s, put_i, a, b), accumulate=True)
    lost_report = {r: {"nodes": 0, "max": 0.0, "mean": 0.0} for r in roles}
    if lost_d:
        lost = torch.zeros(Dn, C, device=dev, dtype=dtype)
        for a in range(0, len(lost_d), chunk):
            b = a + chunk
            lost.index_add_(0, torch.tensor(lost_d[a:b], device=dev), rows(lost_s, lost_i, a, b))
        nodes = sorted(set(lost_d))
        dec = dst.dec_nodes.tolist()
        for d in nodes:
            row = lost[d]
            mx = float(row.max())
            if mx <= 0:
                continue
            role = _role(di.actor[dec[d]], searcher)
            if strict and role == "searcher":
                n = dec[d]
                raise ValueError(
                    f"src puts up to {mx:.3g} probability on actions dst lacks at the searcher's "
                    f"node {n} (history {dt.histories[n]}, dst actions {dt.child_actions(n)}); "
                    "pass strict=False to renormalise"
                )
            rep = lost_report[role]
            rep["nodes"] += 1
            rep["max"] = max(rep["max"], mx)
            rep["mean"] += float(row.mean())
        for r in roles:
            if lost_report[r]["nodes"]:
                lost_report[r]["mean"] /= lost_report[r]["nodes"]
        ids = torch.tensor(nodes, device=dev)
        part = out[ids]
        tot = part.sum(1, keepdim=True)
        out[ids] = torch.where(tot > 0, part / tot.clamp(min=1e-30), fb[ids].to(dtype))
    if unmatched or skipped:
        ids = torch.tensor(unmatched + skipped, device=dev)
        out[ids] = fb[ids].to(dtype)
    report = {
        "searcher": int(searcher),
        "decision_nodes": count["decision"],
        "translated_nodes": count["translated"],
        "unmatched_nodes": count["unmatched"],
        "lost_mass": lost_report,
        "fill": dict(fill),
        "steps": dict(m.steps),
    }
    return out, report


def offtree_edges(
    match: NodeMatch, dst_tree: SubgameTree, actor: int, street: int | None = None
) -> list[int]:
    """The first off-tree ``actor`` edges of ``dst_tree`` on ``street`` (default:
    the root street), as the ``dst`` child nodes they lead to: ``actor`` acts at
    a matched ``dst`` decision node whose whole history ``src`` has exactly, with
    an action ``src`` lacks there. No two of them are nested."""
    st = dst_tree.root_street if street is None else int(street)
    di = _Info(dst_tree)
    out = []
    for n in range(1, dst_tree.num_nodes):
        p = di.parent[n]
        if di.kind[p] != DECISION or di.actor[p] != actor or di.street[p] != st:
            continue
        if match.src_of[p] < 0 or match.translated[p] or di.kind[n] not in _WALK_KINDS:
            continue
        if match.src_of[n] < 0 or match.translated[n]:
            out.append(n)
    return out


def subtree_nodes(tree: SubgameTree, root: int) -> list[int]:
    """``root`` and every node below it (ids ascending; BFS order puts
    descendants after their ancestors)."""
    parent = tree.parent.tolist()
    inside = {root}
    out = [root]
    for n in range(root + 1, tree.num_nodes):
        if parent[n] in inside:
            inside.add(n)
            out.append(n)
    return out


def path_nodes(tree: SubgameTree, node: int) -> list[int]:
    """The strict ancestors of ``node``, root first."""
    out = []
    p = int(tree.parent[node])
    while p >= 0:
        out.append(p)
        p = int(tree.parent[p])
    return out[::-1]


def action_set_differences(src_tree: SubgameTree, dst_tree: SubgameTree) -> list[str]:
    """Why ``src``'s abstraction is not a subset of ``dst``'s: per street, spec
    entries only ``src`` keeps, higher raise caps, or other chance cards.
    Empty when every ``src`` action set is contained in ``dst``'s."""
    out = []
    for s, (a, b) in enumerate(zip(src_tree.street_actions, dst_tree.street_actions, strict=True)):
        extra = [tuple(x) for x in a if tuple(x) not in {tuple(y) for y in b}]
        if extra:
            out.append(f"street {s}: src-only actions {extra}")
    for s, (a, b) in enumerate(zip(src_tree.raise_caps, dst_tree.raise_caps, strict=True)):
        if a > b:
            out.append(f"street {s}: src raise cap {a} > dst {b}")
    if src_tree.chance_cards_used != dst_tree.chance_cards_used:
        out.append(
            f"chance cards differ: {src_tree.chance_cards_used} vs {dst_tree.chance_cards_used}"
        )
    return out


__all__ = [
    "NodeMatch",
    "StrategySnapshot",
    "action_set_differences",
    "harmonic_choice",
    "match_nodes",
    "offtree_edges",
    "path_nodes",
    "subtree_nodes",
    "translate_action",
    "translate_sigma",
]

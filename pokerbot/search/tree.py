"""Depth-limited public subgame tree as flat tensors.

The tree is built in two steps.

1. **Skeleton.** A recursive walk over scalar-engine ``GameState``s from the
   subgame root (the start of the current street) using the search action
   abstraction. Betting never depends on the cards, so after a chance event
   the skeleton keeps a single *template* subtree, built on whatever card the
   engine dealt. The observed actions of the current street (``path``) are
   forced into the tree: an observed size that is not an abstract action (an
   off-tree opponent bet, or one of our own sizes that the node budget
   dropped) becomes an extra branch at the node where it happened.
2. **Expansion.** A breadth-first walk that replicates every chance template
   once per possible card (card conflicts with the hole combos are handled by
   the solver's masks) and writes the flat node arrays. BFS order makes node
   ids depth-sorted and the children of a node contiguous.

Node kinds: ``DECISION``, ``CHANCE`` (deals the next street's card),
``FOLD`` and ``SHOWDOWN`` terminals (a showdown before the river is an all-in
run-out), ``LEAF`` (depth limit: the start of a street beyond the solved
horizon) and ``CONTINUATION``. A ``LEAF`` is a decision of the leaf chooser
(the searcher's opponent by default) among ``k`` continuation strategies
(Pluribus): each ``CONTINUATION`` child is valued by blueprint rollouts
(see :mod:`pokerbot.search.leaf`). With ``leaf_mode="value_net"`` a depth
limit is instead a ``VALUE`` terminal (no children), valued by a leaf value
network on both players' current reaches (see
:mod:`pokerbot.search.value_leaf`).

Node budget: when the expanded tree is larger than ``max_nodes``, bet sizes
are removed deepest street first (the size farthest from pot-sized first),
then raise caps are lowered deepest-first, then remaining sized bets are
dropped (keeping all-in), and finally chance cards are subsampled.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import torch

from ..engine_select import get_engine
from ..env.actions import DEFAULT_SPEC, ActionSpec
from .abstract import (
    BOARD_LEN,
    CHECK_CALL,
    FOLD,
    RAISE,
    action_key,
    contributions,
    legal_options,
    make_state,
    to_action,
)

DECISION, CHANCE, FOLD_NODE, SHOWDOWN, LEAF, CONTINUATION, VALUE = range(7)
KIND_NAMES = ("decision", "chance", "fold", "showdown", "leaf", "continuation", "value")
LEAF_MODES = ("rollouts", "value_net")


@dataclass
class TreeConfig:
    spec: ActionSpec = DEFAULT_SPEC
    depth_streets: int = 1  # streets beyond the current one; the river is always solved fully
    max_nodes: int = 40000
    chance_cards: int | None = None  # None: every card; int: subsample per chance node
    min_chance_cards: int = 4  # floor when the budget forces subsampling
    num_continuations: int = 4  # k continuation strategies at depth-limit leaves
    seed: int = 0
    # "rollouts": LEAF + k CONTINUATION children; "value_net": one VALUE terminal
    leaf_mode: str = "rollouts"
    # nodes a depth-limit leaf counts for in the budget (None: 1 + k for rollouts,
    # 1 for value_net; 1 + k gives a value-net tree the rollout tree's abstraction)
    leaf_budget_cost: int | None = None
    # depth_streets of trees rooted on the turn (None: depth_streets). 0 gives turn
    # solves VALUE (or LEAF) nodes at the end of turn betting; >= 1 solves to showdown
    depth_streets_turn: int | None = None


@dataclass
class _Skel:
    kind: int
    street: int  # street of the betting (for LEAF: the street about to be dealt)
    board_len: int  # public board length at this node
    contrib: tuple[int, int]
    bets: tuple[int, int]
    actor: int = -1
    state: Any = None
    history: tuple = ()  # concrete actions from the subgame root: (street, player, kind, amount)
    children: list = field(default_factory=list)  # [(kind, amount, abstract_idx, _Skel)]
    template: _Skel | None = None
    folder: int = -1
    path_pos: int | None = None  # index of the observed action taken here (None: off path)
    path_slot: int | None = None  # child slot of that observed action
    is_current: bool = False
    count: int = 1


@dataclass
class SubgameTree:
    """Flat public tree. All per-node tensors have length ``num_nodes``."""

    device: torch.device
    parent: torch.Tensor
    slot: torch.Tensor  # position among the parent's children
    depth: torch.Tensor
    kind: torch.Tensor
    actor: torch.Tensor  # acting player (DECISION, LEAF), else -1 (VALUE too)
    street: torch.Tensor
    contrib: torch.Tensor  # [N, 2] chips committed this hand
    bets: torch.Tensor  # [N, 2] chips committed on the node's street
    board_id: torch.Tensor  # index into ``boards``
    deal_card: torch.Tensor  # card dealt by the chance parent, else -1
    chance_weight: torch.Tensor  # float, weight of a chance child in its parent's value
    action_kind: torch.Tensor  # concrete action leading here (-1 at root / chance children)
    action_amount: torch.Tensor
    action_abstract: torch.Tensor  # abstract index of that action, -1 when off-tree
    cont: torch.Tensor  # continuation strategy index (CONTINUATION), else -1
    folder: torch.Tensor  # FOLD: player who folded, else -1
    first_child: torch.Tensor
    num_children: torch.Tensor
    children: torch.Tensor  # [N, max_children] child ids, -1 padded
    boards: list[tuple[int, ...]]
    level_start: list[int]  # node ids of depth d are level_start[d] .. level_start[d+1]-1
    states: list[Any]  # engine state per node (DECISION / LEAF / VALUE), else None
    histories: list[tuple]  # concrete actions from the subgame root per node
    current_node: int  # the observed decision node (end of the path)
    path_nodes: list[tuple[int, int]]  # (node, child slot taken) along the observed path
    root_street: int
    last_street: int
    street_actions: list[list]  # abstract actions per street after budget reduction
    raise_caps: list[int]
    chance_cards_used: int | None
    leaf_chooser: int
    root_history: tuple = ()

    @property
    def num_nodes(self) -> int:
        return int(self.parent.shape[0])

    @property
    def max_children(self) -> int:
        return int(self.children.shape[1])

    def count(self, kind: int) -> int:
        return int((self.kind == kind).sum())

    def summary(self) -> str:
        parts = [f"{name}={self.count(k)}" for k, name in enumerate(KIND_NAMES)]
        return f"{self.num_nodes} nodes ({', '.join(parts)}), depth {len(self.level_start) - 1}"

    def child_actions(self, node: int) -> list[tuple[int, int]]:
        """Concrete ``(kind, amount)`` of each child of a decision node."""
        s, n = int(self.first_child[node]), int(self.num_children[node])
        return [(int(self.action_kind[c]), int(self.action_amount[c])) for c in range(s, s + n)]


def _size_rank(a: tuple) -> float:
    """Drop order for sized raises: farthest from pot-sized first, larger on ties."""
    f = float(a[1])
    return abs(math.log(f)) + 1e-6 * f


class TreeBuilder:
    """Builds a :class:`SubgameTree` rooted at the start of the current street."""

    def __init__(
        self,
        config: Any,
        button: int,
        board: Sequence[int],
        history_before: Sequence,
        path: Sequence = (),
        tree_config: TreeConfig | None = None,
        searcher: int | None = None,
        engine: ModuleType | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.engine = engine or get_engine()
        self.game_config = config
        self.button = int(button)
        self.board = [int(c) for c in board]
        self.history_before = list(history_before)
        self.path = [(int(s), int(p), *action_key(a)) for s, p, a in path]
        self.tc = tree_config or TreeConfig()
        if self.tc.leaf_mode not in LEAF_MODES:
            raise ValueError(f"unknown leaf_mode {self.tc.leaf_mode!r} (expected {LEAF_MODES})")
        self.value_leaves = self.tc.leaf_mode == "value_net"
        cost = self.tc.leaf_budget_cost
        if cost is None:
            cost = 1 if self.value_leaves else 1 + int(self.tc.num_continuations)
        self.leaf_cost = int(cost)
        self.device = torch.device(device)
        self.root_state = make_state(self.engine, config, button, self.board, self.history_before)
        if self.root_state.is_terminal:
            raise ValueError("subgame root is terminal")
        self.root_street = int(self.root_state.street)
        if self.root_street == 0:
            raise ValueError("search subgames start on the flop; preflop uses the blueprint")
        if len(self.board) != BOARD_LEN[self.root_street]:
            raise ValueError("board does not match the street of the root")
        depth = int(self.tc.depth_streets)
        if self.root_street == 2 and self.tc.depth_streets_turn is not None:
            depth = int(self.tc.depth_streets_turn)
        self.last_street = min(3, self.root_street + max(0, depth))
        if self.root_street == 2 and depth >= 1:
            self.last_street = 3
        if searcher is None:
            searcher = int(self.root_state.current_player)
        self.searcher = searcher
        self.leaf_chooser = 1 - searcher
        self.street_actions = [list(s) for s in self.tc.spec.streets]
        self.raise_caps = [self.tc.spec.max_raises] * 4
        self.chance_k = self.tc.chance_cards

    # -- skeleton -----------------------------------------------------------

    def _skeleton(self) -> _Skel:
        return self._node(self.root_state, (), 0)

    def _node(self, state: Any, hist: tuple, path_pos: int | None) -> _Skel:
        c = contributions(state, self.game_config)
        bets = tuple(int(b) for b in state.street_bets)
        node = _Skel(
            DECISION,
            int(state.street),
            BOARD_LEN[int(state.street)],
            (c[0], c[1]),
            bets,
            actor=int(state.current_player),
            state=state,
            history=hist,
        )
        st = node.street
        opts = legal_options(
            state,
            self.tc.spec,
            max_raises=self.raise_caps[st],
            street_actions=self.street_actions[st],
        )
        acts = [(o.kind, o.amount, o.index) for o in opts]
        forced = None
        if path_pos is not None:
            if path_pos < len(self.path):
                ps, pp, pk, pa = self.path[path_pos]
                if pp != node.actor or ps != st:
                    raise ValueError("observed path does not match the tree")
                forced = (pk, pa)
                if all((k, a) != forced for k, a, _ in acts):
                    acts.append((pk, pa, -1))
                    acts.sort(key=lambda t: (t[0], t[1]))
                node.path_pos = path_pos
            else:
                node.is_current = True
        total = 1
        for kind, amount, idx in acts:
            child_state = state.child(to_action(self.engine, kind, amount))
            h = (*hist, (st, node.actor, kind, amount))
            on_path = path_pos + 1 if forced is not None and (kind, amount) == forced else None
            child = self._after(state, child_state, h, on_path)
            if on_path is not None:
                node.path_slot = len(node.children)
            node.children.append((kind, amount, idx, child))
            total += child.count
        node.count = total
        return node

    def _after(self, parent: Any, state: Any, hist: tuple, path_pos: int | None) -> _Skel:
        st = int(parent.street)
        c = contributions(state, self.game_config)
        if state.is_terminal:
            folded = list(state.folded)
            if any(folded):
                n = _Skel(FOLD_NODE, st, BOARD_LEN[st], (c[0], c[1]), (0, 0), history=hist)
                n.folder = folded.index(True)
            else:
                n = _Skel(SHOWDOWN, st, BOARD_LEN[st], (c[0], c[1]), (0, 0), history=hist)
            return n
        new_st = int(state.street)
        if new_st == st:
            return self._node(state, hist, path_pos)
        if path_pos is not None and path_pos < len(self.path):
            raise ValueError("observed path continues past the end of the street")
        if new_st > self.last_street:
            kind = VALUE if self.value_leaves else LEAF
            n = _Skel(kind, new_st, BOARD_LEN[st], (c[0], c[1]), (0, 0), state=state, history=hist)
            if kind == LEAF:
                n.actor = self.leaf_chooser
            n.count = self.leaf_cost
            return n
        n = _Skel(CHANCE, st, BOARD_LEN[st], (c[0], c[1]), (0, 0), history=hist)
        n.template = self._node(state, hist, None)
        n.count = 1 + self._num_cards(BOARD_LEN[st]) * n.template.count
        return n

    def _num_cards(self, board_len: int) -> int:
        avail = 52 - board_len
        return avail if self.chance_k is None else min(avail, int(self.chance_k))

    # -- budget -------------------------------------------------------------

    def build_skeleton(self) -> _Skel:
        """Skeleton within the node budget (see module docstring)."""
        budget = int(self.tc.max_nodes)
        sk = self._skeleton()
        streets = list(range(self.last_street, self.root_street - 1, -1))

        def sized(s: int) -> list:
            return [a for a in self.street_actions[s] if a[0] in ("raise", "raise_x")]

        for s in streets:  # 1: drop bet sizes deepest-first, keep one
            while sk.count > budget and len(sized(s)) > 1:
                drop = max(sized(s), key=_size_rank)
                self.street_actions[s].remove(drop)
                sk = self._skeleton()
        for s in streets:  # 2: lower raise caps deepest-first
            while sk.count > budget and self.raise_caps[s] > 1:
                self.raise_caps[s] -= 1
                sk = self._skeleton()
        for s in streets:  # 3: all-in only
            if sk.count > budget and sized(s):
                for a in sized(s):
                    self.street_actions[s].remove(a)
                sk = self._skeleton()
        while sk.count > budget and self.last_street > self.root_street:  # 4: fewer cards
            cur = 52 - BOARD_LEN[self.root_street] if self.chance_k is None else self.chance_k
            nxt = max(int(self.tc.min_chance_cards), cur // 2)
            if nxt >= cur:
                break
            self.chance_k = nxt
            sk = self._skeleton()
        return sk

    # -- expansion ----------------------------------------------------------

    def build(self) -> SubgameTree:
        sk = self.build_skeleton()
        k = int(self.tc.num_continuations)
        rows: dict[str, list] = {
            n: []
            for n in [
                "parent",
                "slot",
                "depth",
                "kind",
                "actor",
                "street",
                "c0",
                "c1",
                "b0",
                "b1",
                "board",
                "deal",
                "cw",
                "akind",
                "aamt",
                "aabs",
                "cont",
                "folder",
                "first",
                "nchild",
            ]
        }
        states: list[Any] = []
        hists: list[tuple] = []
        boards: dict[tuple, int] = {}
        current = -1
        path_nodes: list[tuple[int, int]] = []
        # queue items: (skel, board, parent, slot, depth, deal, cw, akind, aamt, aabs, cont)
        queue = [(sk, tuple(self.board), -1, 0, 0, -1, 1.0, -1, 0, -1, -1)]
        head = 0
        while head < len(queue):
            s, board, par, slot, depth, deal, cw, ak, aa, ab, cont = queue[head]
            nid = head
            head += 1
            bid = boards.setdefault(board, len(boards))
            r = rows
            r["parent"].append(par)
            r["slot"].append(slot)
            r["depth"].append(depth)
            r["kind"].append(s.kind if cont < 0 else CONTINUATION)
            r["actor"].append(s.actor if cont < 0 else -1)
            r["street"].append(s.street)
            r["c0"].append(s.contrib[0])
            r["c1"].append(s.contrib[1])
            r["b0"].append(s.bets[0])
            r["b1"].append(s.bets[1])
            r["board"].append(bid)
            r["deal"].append(deal)
            r["cw"].append(cw)
            r["akind"].append(ak)
            r["aamt"].append(aa)
            r["aabs"].append(ab)
            r["cont"].append(cont)
            r["folder"].append(s.folder if cont < 0 else -1)
            states.append(s.state if s.kind in (DECISION, LEAF, VALUE) and cont < 0 else None)
            hists.append(s.history)
            kids: list[tuple] = []
            if cont >= 0:
                pass
            elif s.kind == DECISION:
                if s.is_current:
                    current = nid
                for i, (kind, amount, idx, ch) in enumerate(s.children):
                    kids.append((ch, board, nid, i, depth + 1, -1, 1.0, kind, amount, idx, -1))
                if s.path_slot is not None:
                    path_nodes.append((nid, s.path_slot))
            elif s.kind == CHANCE:
                cards, w = self._chance_cards(board)
                for i, x in enumerate(cards):
                    kids.append((s.template, (*board, x), nid, i, depth + 1, x, w, -1, 0, -1, -1))
            elif s.kind == LEAF:
                for j in range(k):
                    kids.append((s, board, nid, j, depth + 1, -1, 1.0, -1, 0, -1, j))
            r["first"].append(len(queue) if kids else -1)
            r["nchild"].append(len(kids))
            queue.extend(kids)
        if current < 0:
            raise ValueError("observed path does not end at a decision node")
        return self._finish(rows, states, hists, boards, current, path_nodes)

    def _chance_cards(self, board: tuple) -> tuple[list[int], float]:
        avail = [c for c in range(52) if c not in board]
        D = 52 - len(board) - 4
        if self.chance_k is None or self.chance_k >= len(avail):
            return avail, 1.0 / D
        rng = random.Random(hash((self.tc.seed, board)) & 0xFFFFFFFF)
        pick = sorted(rng.sample(avail, int(self.chance_k)))
        return pick, len(avail) / (len(pick) * D)

    def _finish(self, rows, states, hists, boards, current, path_nodes) -> SubgameTree:
        dev = self.device

        def L(name: str) -> torch.Tensor:
            return torch.tensor(rows[name], dtype=torch.long, device=dev)

        N = len(rows["parent"])
        nchild = L("nchild")
        first = L("first")
        A = max(1, int(nchild.max()))
        children = torch.full((N, A), -1, dtype=torch.long, device=dev)
        ar = torch.arange(A, device=dev)
        has = ar[None, :] < nchild[:, None]
        children[has] = (first[:, None] + ar[None, :])[has]
        depth = L("depth")
        D = int(depth.max()) + 1
        level_start = torch.searchsorted(depth, torch.arange(D + 1, device=dev)).tolist()
        board_list = [None] * len(boards)
        for b, i in boards.items():
            board_list[i] = b
        return SubgameTree(
            device=dev,
            parent=L("parent"),
            slot=L("slot"),
            depth=depth,
            kind=L("kind"),
            actor=L("actor"),
            street=L("street"),
            contrib=torch.stack([L("c0"), L("c1")], 1),
            bets=torch.stack([L("b0"), L("b1")], 1),
            board_id=L("board"),
            deal_card=L("deal"),
            chance_weight=torch.tensor(rows["cw"], dtype=torch.float32, device=dev),
            action_kind=L("akind"),
            action_amount=L("aamt"),
            action_abstract=L("aabs"),
            cont=L("cont"),
            folder=L("folder"),
            first_child=first,
            num_children=nchild,
            children=children,
            boards=board_list,
            level_start=level_start,
            states=states,
            histories=hists,
            current_node=current,
            path_nodes=path_nodes,
            root_street=self.root_street,
            last_street=self.last_street,
            street_actions=[list(a) for a in self.street_actions],
            raise_caps=list(self.raise_caps),
            chance_cards_used=self.chance_k,
            leaf_chooser=self.leaf_chooser,
            root_history=tuple(self.history_before),
        )


def build_tree(
    config: Any,
    button: int,
    board: Sequence[int],
    history: Sequence,
    tree_config: TreeConfig | None = None,
    searcher: int | None = None,
    engine: ModuleType | None = None,
    device: torch.device | str = "cpu",
) -> SubgameTree:
    """Tree for the public state after ``history`` (full hand history, as in
    ``GameState.history``) on ``board``, rooted at the start of the current
    street, with the current street's actions forced in as the observed path."""
    board = list(board)
    street = {0: 0, 3: 1, 4: 2, 5: 3}[len(board)]
    before = [h for h in history if int(h[0]) < street]
    path = [h for h in history if int(h[0]) == street]
    return TreeBuilder(
        config, button, board, before, path, tree_config, searcher, engine, device
    ).build()


__all__ = [
    "CHANCE",
    "CHECK_CALL",
    "CONTINUATION",
    "DECISION",
    "FOLD",
    "FOLD_NODE",
    "KIND_NAMES",
    "LEAF",
    "LEAF_MODES",
    "RAISE",
    "SHOWDOWN",
    "SubgameTree",
    "TreeBuilder",
    "TreeConfig",
    "VALUE",
    "build_tree",
]

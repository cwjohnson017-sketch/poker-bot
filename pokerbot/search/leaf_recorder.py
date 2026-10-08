"""Record the leaf states a value-net search queries (on-policy training data).

The leaf net of a flop search is evaluated at the end of turn betting on the
solver's *current* reaches of both players, every update. Those reach pairs
(wide ranges early in a solve, sharper ones later, on every line of the tree)
are the inputs the net actually sees, and the training mix of blueprint
self-play, perturbed and random ranges only approximates them (ReBeL trains on
exactly these).

:class:`RecordingLeafProvider` wraps a leaf-value provider
(:class:`~.value_leaf.ValueLeafEvaluator`, :class:`~.value_leaf.TurnEndLeafEvaluator`)
and, on every ``every``-th regret update, stores up to ``per_call`` randomly
chosen turn-end leaves whose reach is non-negligible for both players: the
4-card board, ``c``, the chips behind and both ranges, normalised, in
``(OOP, IP)`` order. :func:`river_states` turns such turn-end states into
river-root states (one or more random river cards each) for exact solving.
"""

from __future__ import annotations

from typing import Any

import torch

from .combos import NUM_COMBOS, avoids_card, valid_masks
from .tree import VALUE, SubgameTree

C = NUM_COMBOS


class LeafStateStore:
    """Turn-end leaf states on the CPU: ``boards [n, 4]``, ``c``, ``stack`` and
    ``ranges [n, 2, 1326]`` (float16, normalised, ``(OOP, IP)``)."""

    def __init__(self) -> None:
        self._parts: dict[str, list[torch.Tensor]] = {k: [] for k in ("boards", "c", "stack")}
        self._parts["ranges"] = []
        self.n = 0

    def __len__(self) -> int:
        return self.n

    def add(self, boards: torch.Tensor, c: torch.Tensor, stack: torch.Tensor, ranges: torch.Tensor):
        self._parts["boards"].append(boards.to("cpu", torch.uint8))
        self._parts["c"].append(c.to("cpu", torch.int32))
        self._parts["stack"].append(stack.to("cpu", torch.int32))
        self._parts["ranges"].append(ranges.to("cpu", torch.float16))
        self.n += int(boards.shape[0])

    def tensors(self) -> dict[str, torch.Tensor]:
        if self.n == 0:
            return {
                "boards": torch.zeros(0, 4, dtype=torch.uint8),
                "c": torch.zeros(0, dtype=torch.int32),
                "stack": torch.zeros(0, dtype=torch.int32),
                "ranges": torch.zeros(0, 2, C, dtype=torch.float16),
            }
        return {k: torch.cat(v) for k, v in self._parts.items()}


class RecordingLeafProvider:
    """A leaf-value provider that forwards to ``inner`` and records leaf states
    into ``store`` on every ``every``-th regret update (``cached=True`` call)."""

    def __init__(
        self,
        inner: Any,
        tree: SubgameTree,
        store: LeafStateStore,
        every: int = 10,
        per_call: int = 8,
        min_mass: float = 1e-5,
        seed: int = 0,
    ) -> None:
        self.inner = inner
        self.store = store
        self.every = max(1, int(every))
        self.per_call = int(per_call)
        self.min_mass = float(min_mass)
        self.gen = torch.Generator().manual_seed(int(seed))
        ids = getattr(inner, "ids", None)
        if ids is None:
            ids = (tree.kind == VALUE).nonzero().flatten()
        self.ids = torch.as_tensor(ids, dtype=torch.long, device=tree.device)
        nodes = self.ids.tolist()
        self.boards4 = torch.tensor(
            [list(tree.boards[int(tree.board_id[n])]) for n in nodes], dtype=torch.long
        ).view(-1, 4)
        self.c = tree.contrib[self.ids, 0].cpu()
        self.stack = torch.tensor(
            [min(int(s) for s in tree.states[n].stacks) for n in nodes], dtype=torch.long
        )
        buttons = {int(tree.states[n].button) for n in nodes}
        if len(buttons) > 1:
            raise ValueError("value leaves of one tree must share the button")
        self.oop = 1 - buttons.pop() if buttons else 1
        self.calls = 0
        self.recorded = 0

    def __getattr__(self, name: str) -> Any:  # num_leaves, net_rows, ... of the inner provider
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def values(self, player: int, reach: torch.Tensor, cached: bool = False) -> torch.Tensor:
        if cached and self.per_call > 0 and len(self.ids):
            self.calls += 1
            if self.calls % self.every == 0:
                self._record(reach)
        return self.inner.values(player, reach, cached)

    @torch.no_grad()
    def _record(self, reach: torch.Tensor) -> None:
        r = reach[[self.oop, 1 - self.oop]]  # [2, L, C], (OOP, IP)
        mass = r.sum(-1)  # [2, L]
        ok = ((mass > self.min_mass).all(0)).nonzero().flatten().cpu()
        if ok.numel() == 0:
            return
        pick = ok[torch.randperm(ok.numel(), generator=self.gen)[: self.per_call]]
        dev_pick = pick.to(r.device)
        sel = r[:, dev_pick] / mass[:, dev_pick, None]
        self.store.add(self.boards4[pick], self.c[pick], self.stack[pick], sel.transpose(0, 1))
        self.recorded += int(pick.numel())


def river_states(
    turn: dict[str, torch.Tensor],
    rivers_per_state: int = 1,
    generator: torch.Generator | None = None,
    min_mass: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """River-root states from turn-end states: ``rivers_per_state`` distinct
    random river cards per state; ranges masked by the card and renormalised.
    States where a player's masked range is (nearly) empty are dropped. Returns
    ``boards [m, 5]`` long, ``c``, ``stack`` long, ``ranges [m, 2, C]`` float."""
    b4 = turn["boards"].long()
    n = b4.shape[0]
    k = max(1, int(rivers_per_state))
    if n == 0:
        return {
            "boards": torch.zeros(0, 5, dtype=torch.long),
            "c": torch.zeros(0, dtype=torch.long),
            "stack": torch.zeros(0, dtype=torch.long),
            "ranges": torch.zeros(0, 2, C),
        }
    on_board = torch.zeros(n, 52, dtype=torch.bool)
    on_board.scatter_(1, b4, True)
    scores = torch.rand(n, 52, generator=generator).masked_fill(on_board, -1.0)
    cards = scores.topk(k, dim=1).indices  # [n, k] distinct, never on the board
    rows = torch.arange(n).repeat_interleave(k)
    x = cards.flatten()
    boards = torch.cat([b4[rows], x[:, None]], 1)
    avoid = avoids_card("cpu").float()
    r = turn["ranges"].float()[rows] * avoid[x][:, None, :]
    r = r * valid_masks(boards)[:, None, :].float()
    mass = r.sum(-1)  # [m, 2]
    keep = (mass > min_mass).all(1)
    r = r / mass.clamp(min=1e-30)[..., None]
    return {
        "boards": boards[keep],
        "c": turn["c"].long()[rows][keep],
        "stack": turn["stack"].long()[rows][keep],
        "ranges": r[keep],
    }


__all__ = ["LeafStateStore", "RecordingLeafProvider", "river_states"]

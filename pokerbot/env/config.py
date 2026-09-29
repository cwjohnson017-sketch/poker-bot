"""Minimal ``GameConfig`` matching the interface contract.

``VecNLHE`` duck-types its config (``stacks``, ``small_blind``,
``big_blind``, ``ante``), so the package-wide ``GameConfig`` can be passed
instead once it exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GameConfig:
    num_players: int = 2
    stacks: list[int] = field(default_factory=lambda: [20000, 20000])
    small_blind: int = 50
    big_blind: int = 100
    ante: int = 0

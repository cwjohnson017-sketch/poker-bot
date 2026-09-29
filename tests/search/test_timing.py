"""(f) Solve time per decision on CPU for a flop subgame at the default config.

Informational only (nothing is asserted about speed). Run with ``-s`` to see
the numbers. The default config targets a GPU; on CPU the iteration loop is
stopped by the time budget after ``min_iterations`` (lowered to 1 here so the
test stays short), so the useful numbers are the setup time and the time per
iteration.
"""

from __future__ import annotations

import numpy as np

from pokerbot.engine_select import get_engine
from pokerbot.eval.masking import MaskedState
from pokerbot.search import SearchAgent, UniformBlueprint, search_config


def test_flop_decision_timing_default_config():
    engine = get_engine()
    cfg = engine.GameConfig(num_players=2, stacks=[20000, 20000], small_blind=50, big_blind=100)
    deck = np.random.default_rng(11).permutation(52).tolist()
    s = engine.GameState.new_hand(cfg, 0, deck)
    s.apply(engine.Action.raise_to(250))
    s.apply(engine.Action.check_call())
    seat = s.current_player
    sc = search_config(device="cpu", fallback_on_error=False, min_iterations=1)
    agent = SearchAgent(UniformBlueprint(), sc)
    agent.new_hand(seat, cfg)
    rng = np.random.default_rng(0)
    action = agent.act(MaskedState(s, seat, cfg, rng), seat, rng)
    assert s.legal_actions().is_legal(action)
    st = agent.last_stats
    per_iter = st["solve_seconds"] / max(1, st["iterations"])
    print(
        f"\n[search timing, CPU] flop subgame, default config: {st['nodes']} nodes; "
        f"tree {st['tree_seconds']:.2f}s, setup (rollouts, terminal tables) "
        f"{st['setup_seconds']:.2f}s, {st['iterations']} iterations in "
        f"{st['solve_seconds']:.2f}s ({per_iter:.2f}s/iteration), "
        f"decision total {st['total_seconds']:.2f}s"
    )

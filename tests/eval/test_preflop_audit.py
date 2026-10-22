"""Preflop off-tree audit (pokerbot.eval.preflop_audit) on tiny agents."""

from __future__ import annotations

import pytest
import torch

from pokerbot.engine_select import get_engine
from pokerbot.eval.preflop_audit import (
    ALLIN,
    audit_decision,
    audit_markdown,
    audit_spots,
    flag_rows,
    run_audit,
)
from pokerbot.search import SearchAgent, UniformBlueprint
from pokerbot.search.abstract import FOLD, RAISE, legal_options, to_action
from pokerbot.search.combos import NUM_COMBOS
from pokerbot.search.preflop_solve import PreflopTable

TINY = {"device": "cpu", "time_budget": 0.02, "fallback_on_error": False}


@pytest.fixture(scope="module")
def config():
    engine = get_engine()
    return engine.GameConfig(
        num_players=2, stacks=[10000, 10000], small_blind=50, big_blind=100, ante=0
    )


def test_spots_end_at_a_bot_decision(config):
    engine = get_engine()
    spots = audit_spots(config)
    fams = {s.family for s in spots}
    assert fams == {"open", "3bet", "4bet", "limp_raise", "limp_reraise"}
    series: dict = {}
    for s in spots:
        state = engine.GameState.new_hand(config, 0, list(range(52)))
        for k, a in s.prefix:
            assert state.legal_actions().is_legal(to_action(engine, k, a)), s
            state.apply(to_action(engine, k, a))
        assert not state.is_terminal and state.street == 0
        assert state.current_player == s.bot_seat
        series.setdefault((s.family, s.series), []).append(s.prefix[-1][1])
    for sizes in series.values():
        assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)
        assert sizes[-1] == 10000  # every series ends with an all-in
    opens = series[("open", "vs open")]
    assert opens[0] == 200 and 275 in opens  # min-raise and an off-tree size


def test_uniform_bot_frequencies_and_defence_bar(config):
    agent = SearchAgent(UniformBlueprint(), TINY)
    rows = run_audit(agent, config)
    assert not agent.record  # restored
    assert len(rows) == len(audit_spots(config))
    for r in rows:
        assert r["fold"] + r["call"] + r["raise"] == pytest.approx(1.0, abs=1e-5)
        assert 0 < r["max_fold"] < 1
        assert r["allin"] <= r["raise"] + 1e-9
    r2 = next(r for r in rows if r["family"] == "open" and r["raise_to"] == 200)
    # SB adds 150 to a pot of 150: folding more than half lets any two cards raise
    assert r2["bet"] == 150 and r2["pot_before"] == 150
    assert r2["max_fold"] == pytest.approx(0.5)
    assert r2["range"] == pytest.approx(1.0)  # the big blind has not acted
    jam = next(r for r in rows if r["family"] == "open" and r["size"] == ALLIN)
    assert jam["raise"] == 0 and jam["fold"] == pytest.approx(0.5)  # fold or call
    md = audit_markdown(rows)
    assert "Big blind facing an open" in md and "spots flagged" in md


def test_table_spots_and_flags(config):
    """A table that folds every hand to a 2bb open: that spot is read from the
    table and overfolds; 2.25bb is off the table (the uniform base plays it)."""
    engine = get_engine()
    base = UniformBlueprint()
    width = base.spec.num_actions
    state = engine.GameState.new_hand(config, 0, list(range(52)))
    state.apply(to_action(engine, RAISE, 200))
    fold = next(o.index for o in legal_options(state, base.spec) if o.kind == FOLD)
    rows = torch.zeros(NUM_COMBOS, width)
    rows[:, fold] = 1.0
    bp = PreflopTable(base, {((0, RAISE, 200),): rows})
    agent = SearchAgent(bp, TINY, record=True)
    spots = [s for s in audit_spots(config) if s.family == "open"][:2]
    r = flag_rows([audit_decision(agent, config, s, engine) for s in spots])
    assert r[0]["source"] == "table" and r[0]["fold"] == pytest.approx(1.0)
    assert any(f.startswith("overfold") for f in r[0]["flags"])
    assert r[1]["source"] == "blueprint" and r[1]["fold"] < 0.5
    assert any(f.startswith("non_monotone") for f in r[1]["flags"])
    assert any(f.startswith("jump") for f in r[1]["flags"])
    # pseudo-harmonic translation of 2.25bb between the base's neighbouring opens
    assert sum(r[1]["translation"].values()) == pytest.approx(1.0)


def test_flags_skip_far_apart_sizes():
    def row(size, to, fold):
        return dict(family="open", series="s", size=size, raise_to=to, fold=fold, max_fold=0.99)

    rows = flag_rows([row("a", 200, 0.3), row("b", 250, 0.7), row("c", 1000, 0.1)])
    assert rows[0]["flags"] == []
    assert rows[1]["flags"] == ["jump +40 vs a"]
    assert rows[2]["flags"] == ["non_monotone -60 vs b"]  # 4x apart: no jump flag

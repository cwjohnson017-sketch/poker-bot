"""Archetype opponents (``pokerbot.agents.archetypes``) and the suite runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from pokerbot.agents import (
    ARCHETYPES,
    SUITE,
    Agent,
    Hud,
    ProfileAgent,
    RandomAgent,
    TranslationStressAgent,
    make_agent,
)
from pokerbot.agents.archetypes import preflop_percentiles, read_spot
from pokerbot.agents.base import CHECK_CALL, FOLD, RAISE
from pokerbot.engine_select import get_engine
from pokerbot.eval import MaskedState, play_hand
from pokerbot.reference import cards_from_str
from tests.helpers import make_deck

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
ENGINE = get_engine()


def cfg(*stacks: int):
    return ENGINE.GameConfig(num_players=2, stacks=list(stacks), small_blind=50, big_blind=100)


DEEP = cfg(10000, 10000)
# short and uneven stacks: all-in for less, no legal raise, blinds nearly all-in
SHORT = [cfg(10000, 350), cfg(1000, 1000), cfg(150, 10000), cfg(2500, 600)]


class LegalityCheck:
    """Wraps an agent; fails if it folds when checking is free or sends a
    raise outside the legal range (the engine would also reject it)."""

    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name
        self.decisions = [0, 0, 0, 0]

    def new_hand(self, seat, config):
        self.inner.new_hand(seat, config)

    def observe_end(self, state):
        self.inner.observe_end(state)

    def act(self, state, seat, rng):
        legal = state.legal_actions()
        a = self.inner.act(state, seat, rng)
        kind = int(a.kind)
        assert not (kind == FOLD and legal.can_check), f"{self.name} folded with a free check"
        if kind == RAISE:
            assert legal.min_raise_to > 0, f"{self.name} raised with no raise legal"
            ok = legal.min_raise_to <= a.amount <= legal.max_raise_to
            assert ok or a.amount == legal.max_raise_to, f"{self.name} raise {a.amount}"
        self.decisions[state.street] += 1
        return a


def view(state, seat, config=DEEP):
    return MaskedState(state, seat, config, np.random.default_rng(0))


def test_registry_names_and_specs():
    for name in [*ARCHETYPES, *SUITE]:
        a = make_agent(name)
        assert isinstance(a, Agent)
        assert a.name == name
    p = make_agent("population:vpip=0.3,pfr=0.1,agg=0.2")
    assert isinstance(p, ProfileAgent) and (p.vpip, p.pfr, p.agg) == (0.3, 0.1, 0.2)
    assert make_agent("maniac:bet_size=1.5").bet_size == 1.5
    assert make_agent("maniac:name=crazy").name == "crazy"
    t = make_agent("translation_stress:limp=0.9")
    assert isinstance(t, TranslationStressAgent) and isinstance(t, RandomAgent) and t.limp == 0.9
    with pytest.raises(ValueError):
        make_agent("population:sizing=huge")


def test_preflop_percentiles():
    pct = preflop_percentiles()

    def p(s):
        a, b = cards_from_str(s)
        return pct[a, b]

    assert p("AsAh") < 0.01 and p("AhAs") == p("AsAh")
    assert p("AsKs") < p("AsKd") < p("9s8d") < p("7c2d")
    assert p("7c2d") > 0.95
    # "the top x of hands" covers about x of the 1,326 combos
    combos = pct[np.triu_indices(52, 1)]
    for x in (0.05, 0.12, 0.35, 0.6):
        assert abs((combos < x).mean() - x) < 0.02


@pytest.mark.parametrize("name", SUITE)
def test_archetypes_always_legal(name):
    """2,000 hands per archetype at 100bb against a mix of opponents, plus
    1,000 at short and uneven stacks."""
    rng = np.random.default_rng(hash(name) % 2**32)
    agent = LegalityCheck(make_agent(name))
    opponents = [RandomAgent(), make_agent("translation_stress"), make_agent(name)]
    configs = [DEEP] * 4 + SHORT
    for h in range(3000):
        config = configs[h % len(configs)] if h >= 2000 else DEEP
        opp = LegalityCheck(opponents[h % len(opponents)])
        seats = [agent, opp] if h % 2 == 0 else [opp, agent]
        s = play_hand(config, seats, (h // 2) % 2, rng.permutation(52), rng, ENGINE)
        assert s.is_terminal
        assert sum(s.payoffs()) == 0
    assert all(d > 0 for d in agent.decisions), agent.decisions  # reached every street


def test_hud_counts_one_hand():
    s = ENGINE.GameState.new_hand(DEEP, 0, make_deck(["AsKs", "7c2d"], "Qh9d3c4s5h"))
    A = ENGINE.Action
    for a in (A.raise_to(250), A.check_call(), A.check_call(), A.raise_to(300), A.fold()):
        s.apply(a)
    sb, bb = Hud(), Hud()
    sb.record(list(s.history), 0, 0)
    bb.record(list(s.history), 1, 0)
    assert (sb.vpip, sb.pfr, sb.cbet_opp, sb.cbet) == (1, 1, 1, 1)
    assert (sb.three_bet_opp, sb.post_aggr, sb.post_passive) == (0, 1, 0)
    assert sb.summary()["agg"] == 1.0 and np.isnan(sb.summary()["three_bet"])
    assert (bb.vpip, bb.pfr, bb.three_bet_opp, bb.three_bet) == (1, 0, 1, 0)
    assert (bb.fold_cbet_opp, bb.fold_cbet, bb.post_aggr, bb.post_passive) == (1, 1, 0, 1)
    assert sb.opp_actions == [1, 2, 0, 0] and bb.opp_actions == [1, 1, 0, 0]
    merged = Hud()
    merged.merge(sb.counts())
    merged.merge(sb.counts())
    assert merged.hands == 2 and merged.opp_actions == [2, 4, 0, 0]


def test_read_spot_limp_and_aggressor():
    s = ENGINE.GameState.new_hand(DEEP, 0, make_deck(["AsKs", "7c2d"], "Qh9d3c4s5h"))
    A = ENGINE.Action
    s.apply(A.check_call())  # button limps
    s.apply(A.raise_to(400))  # big blind raises
    spot = read_spot(view(s, 0), 0)
    assert spot.limped and spot.raises == 1 and spot.acted and spot.preflop_aggressor == 1
    s.apply(A.check_call())
    spot = read_spot(view(s, 1), 1)
    assert spot.street == 1 and spot.first and spot.preflop_aggressor == 1 and spot.raises == 0


def test_profile_decisions():
    rng = np.random.default_rng(0)
    A = ENGINE.Action
    nit = make_agent("nit")
    station = make_agent("calling_station")
    # the nit folds 72o from the small blind and raises aces
    for agent, hole, kind in ((nit, "7c2d", FOLD), (nit, "AsAh", RAISE)):
        s = ENGINE.GameState.new_hand(DEEP, 0, make_deck([hole, "KsKh"]))
        agent.new_hand(0, DEEP)
        assert int(agent.act(view(s, 0), 0, rng).kind) == kind
    # min-raiser opens to the minimum; maniac opens to its open size
    s = ENGINE.GameState.new_hand(DEEP, 0, make_deck(["AsAh", "KsKh"]))
    mr = make_agent("min_raiser")
    mr.new_hand(0, DEEP)
    assert mr.act(view(s, 0), 0, rng).amount == s.legal_actions().min_raise_to
    # the calling station calls a pot-sized flop bet with a weak pair
    s = ENGINE.GameState.new_hand(DEEP, 0, make_deck(["AsKs", "3h4d"], "Qh9d3c2s5h"))
    for a in (A.raise_to(250), A.check_call(), A.check_call(), A.raise_to(500)):
        s.apply(a)
    station.new_hand(1, DEEP)
    assert int(station.act(view(s, 1), 1, rng).kind) == CHECK_CALL


def test_overbet_bluffer_sizes():
    rng = np.random.default_rng(1)
    sizes = []

    class Recorder(LegalityCheck):
        def act(self, state, seat, rng):
            all_in = state.legal_actions().max_raise_to
            a = super().act(state, seat, rng)
            if state.street > 0 and int(a.kind) == RAISE and max(state.street_bets) == 0:
                sizes.append(9.0 if a.amount == all_in else a.amount / state.pot)
            return a

    agent = Recorder(make_agent("overbet_bluffer"))
    opp = make_agent("calling_station")
    for h in range(400):
        play_hand(DEEP, [agent, opp] if h % 2 else [opp, agent], h % 2, rng.permutation(52), rng)
    sizes = np.array(sizes)
    assert len(sizes) > 30
    # 1.5-2.5x pot unless capped by the stack (all-in recorded as 9)
    assert np.all(sizes >= 1.49)


def test_translation_stress_lines():
    """Limps, odd open sizes, limp-reraises and donk bets all show up."""
    rng = np.random.default_rng(2)
    stress = make_agent("translation_stress")
    opp = make_agent("regular")
    limps = opens = limp_reraises = donks = 0
    open_sizes = set()
    for h in range(600):
        seats = [stress, opp] if h % 2 == 0 else [opp, stress]
        me = h % 2
        button = me if h % 4 < 2 else 1 - me
        s = play_hand(DEEP, seats, button, rng.permutation(52), rng)
        hist = list(s.history)
        pre = [(p, a) for st, p, a in hist if st == 0]
        if button == me and pre:
            first = pre[0][1]
            if int(first.kind) == CHECK_CALL:
                limps += 1
                if len(pre) >= 3 and int(pre[1][1].kind) == RAISE and pre[2][0] == me:
                    limp_reraises += int(pre[2][1].kind) == RAISE
            elif int(first.kind) == RAISE:
                opens += 1
                open_sizes.add(first.amount)
        aggr = [p for p, a in pre if int(a.kind) == RAISE]
        flop = [(p, a) for st, p, a in hist if st == 1]
        if aggr and aggr[-1] != me and flop and flop[0][0] == me:
            donks += int(flop[0][1].kind) == RAISE
    assert limps > 50 and opens > 50 and limp_reraises > 3 and donks > 5
    assert len(open_sizes) > 20  # odd sizes, not a fixed open


def load_runner():
    spec = importlib.util.spec_from_file_location("run_archetypes", SCRIPTS / "run_archetypes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_suite_runner_tiny(tmp_path, capsys):
    runner = load_runner()
    out = tmp_path / "suite.json"
    base = ["--agent", "equity:samples=20", "--reference", "call", "--deals", "6", "--chunk", "3"]
    args = [*base, "--archetypes", "nit,translation_stress", "--out", str(out), "--no-adjust"]
    assert runner.main(args) == 0
    data = json.loads(out.read_text())
    assert set(data["archetypes"]) == {"nit", "translation_stress"}
    for entry in data["archetypes"].values():
        for role in ("agent", "reference"):
            chunks = entry[role]["chunks"]
            assert [c["k"] for c in chunks] == [0, 1]
            assert sum(len(c["a"]) for c in chunks) == 6
    row = data["summary"]["nit"]
    assert row["agent"]["hands"] == 12 and "diff" in row
    assert row["agent"]["hud"]["hands"] == 12
    text = capsys.readouterr().out
    assert "agent - reference" in text and "translation_stress" in text
    # resume plays nothing new; a different agent is refused
    before = out.read_text()
    assert runner.main([*args, "--resume"]) == 0
    assert json.loads(out.read_text())["archetypes"] == json.loads(before)["archetypes"]
    with pytest.raises(SystemExit):
        runner.main(["--agent", "random", *args[2:], "--resume"])
    # luck adjustment on a tiny run
    adj = tmp_path / "adj.json"
    runner.main(["--agent", "call", "--archetypes", "maniac", "--deals", "3", "--out", str(adj)])
    data = json.loads(adj.read_text())
    chunk = data["archetypes"]["maniac"]["agent"]["chunks"][0]
    assert data["luck_adjust"] and len(chunk["adjusted"]) == 3

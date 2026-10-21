"""PlaySession: the hand loop, the log, evaluation-mode views, resuming."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pokerbot.agents import AlwaysCallAgent, RandomAgent
from pokerbot.engine_select import get_engine, to_engine_action
from pokerbot.play import PlaySession, SessionLog, deal, parse_action
from pokerbot.play.session import raise_presets

from .helpers import calling_station, drive, mixed, new_log


@pytest.fixture
def engine():
    return get_engine()


@pytest.fixture
def config(engine):
    return engine.GameConfig(
        num_players=2, stacks=[2000, 2000], small_blind=50, big_blind=100, ante=0
    )


def replay(engine, config, rec):
    """Replay a logged hand from its deck and actions; returns the terminal state."""
    s = engine.GameState.new_hand(config, rec["button"], rec["deck"])
    for a in rec["actions"]:
        assert int(s.current_player) == a["seat"]
        s.apply(to_engine_action(engine, SimpleNamespace(**a)))
    assert s.is_terminal
    return s


def test_session_plays_logs_and_stops_at_its_sample(tmp_path, engine, config):
    log = new_log(tmp_path)
    session = PlaySession(
        RandomAgent(), config, log, seed=11, hands_target=6, engine=engine, luck_samples=64
    )
    session.start()
    last = drive(session, mixed(1))
    session.join(5)
    assert last["status"] == "finished"
    assert last["session"]["hands_played"] == 6

    hands = SessionLog(log.dir).hands()
    assert [h["hand"] for h in hands] == list(range(6))
    assert len(log.dealt()) == 6
    for h in hands:
        assert h["deck"] == deal(11, h["hand"])
        assert h["button"] == h["hand"] % 2
        assert sum(h["payoffs"]) == 0
        assert h["human_net"] == h["payoffs"][0]
        terminal = replay(engine, config, h)
        assert list(terminal.payoffs()) == h["payoffs"]
        assert h["hole_cards"] == [list(terminal.hole_cards(p)) for p in (0, 1)]
        assert all(a["seconds"] >= 0 for a in h["actions"])
    text = (log.dir / "hands.txt").read_text()
    stars = (log.dir / "pokerstars.txt").read_text()
    assert text.count("# hand ") == 6
    assert stars.count("PokerStars Hand #") == 6
    assert session.stats.hands == 6
    assert session.stats.summary()["net"] == sum(h["human_net"] for h in hands)
    # the sample is complete: no more deals
    assert session.request_next() == (False, "the session's sample is complete")


def test_views_hide_the_bots_cards_unless_shown_down(tmp_path, engine, config):
    session = PlaySession(
        RandomAgent(), config, new_log(tmp_path), seed=3, hands_target=12, engine=engine
    )
    checked = {"live": 0, "showdown": 0, "folded": 0}

    def look(v):
        hand = v["hand"]
        if not hand:
            return
        bot = next(p for p in hand["players"] if not p["human"])
        me = next(p for p in hand["players"] if p["human"])
        assert me["cards"] and len(me["cards"]) == 2
        result = hand["result"]
        if result is None:
            assert bot["cards"] is None
            checked["live"] += 1
        elif result["showdown"]:
            assert bot["cards"] and len(bot["cards"]) == 2
            assert v["last_hand"]["bot_cards"] == bot["cards"]
            checked["showdown"] += 1
        else:
            assert bot["cards"] is None
            assert v["last_hand"]["bot_cards"] is None
            checked["folded"] += 1
        if v["last_hand"] is not None and v["last_hand"]["number"] != hand["number"]:
            assert v["last_hand"]["number"] == hand["number"] - 1

    session.start()
    drive(session, mixed(4), on_view=look)
    assert checked["live"] and checked["showdown"] and checked["folded"]


def test_illegal_and_out_of_turn_requests_are_refused(tmp_path, engine, config):
    session = PlaySession(
        AlwaysCallAgent(), config, new_log(tmp_path), seed=1, hands_target=1, engine=engine
    )
    assert session.submit_action("call")[0] is False  # nobody is to act yet
    session.start()
    v = session.snapshot()
    assert v["status"] == "ready"
    assert session.request_next() == (True, None)
    v = session.snapshot(since=v["version"])
    while v["status"] != "human":
        v = session.snapshot(since=v["version"])
    legal = v["hand"]["legal"]
    ok, err = session.submit_action("raise", 1)
    assert not ok and "raise to between" in err
    ok, err = session.submit_action("dance")
    assert not ok and "unknown action" in err
    assert session.request_next()[0] is False  # mid-hand
    assert session.submit_action("raise", legal["min_raise_to"]) == (True, None)
    assert session.submit_action("call")[0] is False  # one action per prompt
    drive(session, calling_station, since=v["version"])


def test_parse_action_rules(engine, config):
    s = engine.GameState.new_hand(config, 0, list(range(52)))
    legal = s.legal_actions()  # the button (small blind) faces the big blind
    assert int(parse_action(engine, legal, "call").kind) == 1
    with pytest.raises(ValueError, match="bet to call"):
        parse_action(engine, legal, "check")
    a = parse_action(engine, legal, "allin")
    assert int(a.amount) == int(legal.max_raise_to)
    s.apply(engine.Action.check_call())
    legal = s.legal_actions()  # the big blind may check
    with pytest.raises(ValueError, match="checking is free"):
        parse_action(engine, legal, "fold")
    with pytest.raises(ValueError, match="not a number"):
        parse_action(engine, legal, "raise", "lots")


def test_raise_presets_are_legal_and_end_in_all_in(engine, config):
    s = engine.GameState.new_hand(config, 0, list(range(52)))
    legal = s.legal_actions()
    pre = raise_presets(s, 0, legal, 100)
    assert [p["label"] for p in pre] == ["2x", "2.5x", "3x", "4x", "all-in"]
    assert [p["to"] for p in pre] == [200, 250, 300, 400, 2000]
    s.apply(engine.Action.check_call())
    s.apply(engine.Action.check_call())  # flop, pot 200
    legal = s.legal_actions()
    pre = raise_presets(s, int(s.current_player), legal, 100)
    tos = [p["to"] for p in pre]
    assert tos == sorted(tos) and tos[-1] == int(legal.max_raise_to)
    # 33% of 200 is below the minimum bet (100) and 50% is that minimum again
    assert [p["label"] for p in pre] == ["33%", "75%", "pot", "150%", "all-in"]
    assert pre[1] == {"label": "75%", "to": 150}
    assert all(int(legal.min_raise_to) <= t <= int(legal.max_raise_to) for t in tos)


def test_resume_continues_the_deal_and_counts_abandoned_hands(tmp_path, engine, config):
    log = new_log(tmp_path)
    first = PlaySession(AlwaysCallAgent(), config, log, seed=9, engine=engine)
    first.start()
    v = drive(first, calling_station, stop_after=2)  # hands 0 and 1 played, 2 abandoned
    first.join(5)
    assert v["status"] == "finished"
    assert first.abandoned == 1

    log2 = SessionLog(log.dir)
    second = PlaySession(AlwaysCallAgent(), config, log2, seed=9, hands_target=4, engine=engine)
    assert second.next_hand == 3
    assert second.abandoned == 1
    assert second.stats.hands == 2
    second.start()
    drive(second, calling_station)
    hands = log2.hands()
    assert [h["hand"] for h in hands] == [0, 1, 3, 4]
    assert hands[2]["deck"] == deal(9, 3)
    assert [d["hand"] for d in log2.dealt()] == [0, 1, 2, 3, 4]


def test_search_bot_decisions_are_logged(tmp_path, engine):
    from pokerbot.search import SearchAgent, UniformBlueprint

    config = engine.GameConfig(
        num_players=2, stacks=[1000, 1000], small_blind=50, big_blind=100, ante=0
    )
    tiny = {
        "device": "cpu",
        "time_budget": 0.02,
        "min_iterations": 2,
        "fallback_on_error": False,
        "tree": {"max_nodes": 1500, "chance_cards": 3, "max_raises": 2},
        "solver": {"iterations": 4, "max_runouts": 6},
        "leaf": {"rollouts": 1, "max_total_rollouts": 64},
        "gadget": {"rollouts": 8},
    }
    bot = SearchAgent(UniformBlueprint(), tiny, record=True)
    log = new_log(tmp_path)
    session = PlaySession(bot, config, log, seed=2, hands_target=4, engine=engine)
    session.start()
    drive(session, calling_station)
    hands = log.hands()
    sources = {d["source"] for h in hands for d in h["bot_decisions"]}
    assert "search" in sources
    for h in hands:
        if not h["bot_decisions"]:
            continue
        arrays = log.solves(h["hand"])
        bot_actions = [a for a in h["actions"] if a["seat"] == 1]
        assert len(bot_actions) == len(h["bot_decisions"])
        for k, d in enumerate(h["bot_decisions"]):
            strat = arrays[f"d{k}_strategy"]
            assert strat.shape == (1326, len(d["actions"]))
            assert arrays[f"d{k}_reach"].shape == (2, 1326)
            assert np.allclose(strat.sum(1), 1.0, atol=1e-4)
            kind, amount = d["actions"][d["taken"]]
            played = bot_actions[k]
            assert played["kind"] == kind
            if kind == 2:
                assert played["amount"] == amount
            assert (f"d{k}_values" in arrays) == d["has_values"] == (d["source"] == "search")

"""Soak harness (pokerbot.eval.soak) on a tiny search bot."""

from __future__ import annotations

import json

import pytest

from pokerbot.agents import AlwaysCallAgent, BaseAgent, RandomAgent
from pokerbot.engine_select import get_engine
from pokerbot.eval.soak import chunk_path, process_rss_mb, run_soak, summarize
from pokerbot.search import SearchAgent, UniformBlueprint

TINY = {
    "device": "cpu",
    "time_budget": 0.02,
    "min_iterations": 2,
    "fallback_on_error": True,
    "tree": {"max_nodes": 1500, "chance_cards": 3, "max_raises": 2},
    "solver": {"iterations": 4, "max_runouts": 6},
    "leaf": {"rollouts": 1, "max_total_rollouts": 64},
    "gadget": {"rollouts": 8},
}


@pytest.fixture(scope="module")
def config():
    return get_engine().GameConfig(
        num_players=2, stacks=[1000, 1000], small_blind=50, big_blind=100, ante=0
    )


class Faulty(BaseAgent):
    """Calls, but raises on the hands in ``crash`` and plays an illegal raise
    (to 1 chip) on the hands in ``illegal`` (hands counted from its first)."""

    name = "faulty"

    def __init__(self, crash=(), illegal=()):
        super().__init__()
        self.crash, self.illegal, self.hand = set(crash), set(illegal), -1

    def new_hand(self, seat, config):
        super().new_hand(seat, config)
        self.hand += 1

    def act(self, state, seat, rng):
        if self.hand in self.crash:
            raise RuntimeError("boom")
        if self.hand in self.illegal:
            return self.raise_to(1)
        return self.check_call()


def test_soak_runs_resumes_and_passes(tmp_path, config):
    bot = SearchAgent(UniformBlueprint(), TINY)
    opps = [RandomAgent(), AlwaysCallAgent()]
    s = run_soak(bot, opps, config, 12, tmp_path, chunk=5, seed=1)
    assert s["hands"] == 12 and s["chunks"] == 3 and s["gate"]["complete"]
    assert s["gate"]["pass"], s["gate"]
    assert s["crashes"] == 0 and s["illegal"] == 0
    assert s["decisions"]["preflop"] > 0
    assert sum(s["searched"].values()) > 0 and sum(s["fallbacks"].values()) == 0
    assert set(s["opponents"]) == {"random", "always_call"}
    assert sum(o["hands"] for o in s["opponents"].values()) == 12
    assert bot.stats == []  # emptied after every chunk
    assert (tmp_path / "soak.md").read_text(encoding="utf-8").startswith("# Soak test")
    # a lost chunk is replayed exactly; the others are kept
    before = json.loads(chunk_path(tmp_path, 1).read_text())
    kept = chunk_path(tmp_path, 2).stat().st_mtime_ns
    chunk_path(tmp_path, 1).unlink()
    bot2 = SearchAgent(UniformBlueprint(), TINY)
    s2 = run_soak(bot2, [RandomAgent(), AlwaysCallAgent()], config, 12, tmp_path, chunk=5, seed=1)
    after = json.loads(chunk_path(tmp_path, 1).read_text())
    assert after["payoffs"] == before["payoffs"]
    assert chunk_path(tmp_path, 2).stat().st_mtime_ns == kept
    assert s2["hands"] == 12
    with pytest.raises(ValueError, match="other settings"):
        run_soak(bot2, opps, config, 12, tmp_path, chunk=4, seed=1)


def test_crashes_and_illegal_actions_are_counted(tmp_path, config):
    bot = SearchAgent(UniformBlueprint(), TINY)
    s = run_soak(bot, [Faulty(crash={1}, illegal={2})], config, 4, tmp_path, chunk=4)
    assert s["hands"] == 4
    assert s["crashes"] == 1 and s["bot_crashes"] == 0
    assert s["crash_examples"][0]["agent"] == "opponent or engine"
    assert "boom" in s["crash_examples"][0]["error"]
    assert s["illegal"] >= 1 and s["bot_illegal"] == 0
    assert not s["gate"]["checks"]["crashes"] and s["gate"]["checks"]["illegal"]
    rec = json.loads(chunk_path(tmp_path, 0).read_text())
    assert rec["payoffs"][1] is None and rec["payoffs"][0] is not None


def test_bot_faults_and_fallbacks_fail_the_gate(tmp_path, config):
    bot = SearchAgent(UniformBlueprint(), TINY)

    def broken(*a, **k):
        raise RuntimeError("solver failed")

    bot._search_action = broken  # every postflop decision falls back
    s = run_soak(bot, [AlwaysCallAgent()], config, 6, tmp_path / "fb", chunk=3)
    assert s["crashes"] == 0
    assert sum(s["fallbacks"].values()) > 0 and s["fallback_rate"] == pytest.approx(1.0)
    assert not s["gate"]["pass"]

    class IllegalBot(Faulty):
        name = "illegal_bot"

    s = run_soak(IllegalBot(illegal={0}, crash={1}), [AlwaysCallAgent()], config, 2, tmp_path / "b")
    assert s["bot_illegal"] >= 1 and s["bot_crashes"] == 1
    assert not s["gate"]["checks"]["illegal"]
    assert summarize(tmp_path / "b", 2)["bot_crashes"] == 1


def test_process_rss():
    rss = process_rss_mb()
    assert rss is not None and rss > 10

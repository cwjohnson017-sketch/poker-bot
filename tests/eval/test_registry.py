import pytest

from pokerbot.agents import (
    AlwaysCallAgent,
    EquityThresholdAgent,
    RandomAgent,
    UniformPolicyAgent,
    available_agents,
    make_agent,
    parse_spec,
    registry,
)
from pokerbot.agents.base import BaseAgent


def test_parse_spec():
    s = parse_spec("equity:raise_threshold=0.8,samples=100")
    assert s.name == "equity" and s.args == []
    assert s.kwargs == {"raise_threshold": 0.8, "samples": 100}
    s = parse_spec("blueprint:runs/a/iter_0010.pkl,temperature=0.5,greedy=true")
    assert s.args == ["runs/a/iter_0010.pkl"]
    assert s.kwargs == {"temperature": 0.5, "greedy": True}
    assert s.label == "blueprint:iter_0010"
    assert parse_spec("always_call").label == "always_call"
    with pytest.raises(ValueError):
        parse_spec("  ")


def test_baselines_round_trip():
    assert isinstance(make_agent("always_call"), AlwaysCallAgent)
    assert isinstance(make_agent("random"), RandomAgent)
    eq = make_agent("equity:samples=10,call_threshold=0.5")
    assert isinstance(eq, EquityThresholdAgent)
    assert eq.samples == 10 and eq.call_threshold == 0.5
    # config defaults are overridden by the spec's own values
    eq = make_agent("equity:samples=10", samples=99, raise_threshold=0.9)
    assert eq.samples == 10 and eq.raise_threshold == 0.9
    assert isinstance(make_agent("uniform"), UniformPolicyAgent)
    assert registry.make_agent("equity:samples=7").samples == 7


def test_register_decorator_and_checkpoint_factory(tmp_path):
    @registry.register("dummy_ckpt", "dummy2")
    class Dummy(BaseAgent):
        name = "dummy"

        def __init__(self, path=None, scale=1.0):
            super().__init__()
            self.path, self.scale = path, scale

        @classmethod
        def from_checkpoint(cls, path, **kw):
            return cls(path=f"loaded:{path}", **kw)

    try:
        a = make_agent(f"dummy_ckpt:{tmp_path}/x.pt,scale=2")
        assert a.path == f"loaded:{tmp_path}/x.pt" and a.scale == 2
        assert isinstance(make_agent("dummy2"), Dummy)
        assert "dummy_ckpt" in available_agents()
    finally:
        registry._REGISTRY.pop("dummy_ckpt", None)
        registry._REGISTRY.pop("dummy2", None)


def test_lazy_entries_and_clear_errors(monkeypatch):
    names = available_agents()
    for n in ("blueprint", "neural", "search", "always_call", "equity", "uniform"):
        assert n in names
    monkeypatch.setitem(registry.LAZY_AGENTS, "ghost", "pokerbot.no_such_module.agent:Ghost")
    with pytest.raises(registry.AgentSpecError, match="could not be imported"):
        make_agent("ghost:some/ckpt.pt")
    monkeypatch.setitem(registry.LAZY_AGENTS, "ghost2", "pokerbot.agents.baselines:NoSuchAgent")
    with pytest.raises(registry.AgentSpecError, match="no attribute"):
        make_agent("ghost2")
    with pytest.raises(ValueError, match="unknown agent"):
        make_agent("definitely_not_an_agent")

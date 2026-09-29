"""Build any agent from a spec string.

Spec grammar::

    name
    name:arg1,arg2,key=value,key2=value2

``name`` selects a factory. Items after the first ``:`` are split on commas;
``key=value`` items become keyword arguments (values parsed as YAML
scalars, so ``0.8`` is a float and ``true`` a bool), other items are
positional arguments (typically a checkpoint path). Examples::

    always_call
    equity:raise_threshold=0.8,samples=100
    blueprint:checkpoints/mccfr/iter_0100.pkl
    neural:checkpoints/deepcfr/iter_0040.pt,device=cuda

Factories are looked up in this order:

1. entries registered with ``@register("name")`` (any module can register);
2. the baseline table ``pokerbot.agents.AGENTS``;
3. a lazy import table (``LAZY_AGENTS``) of ``module:attribute`` targets for
   heavy agents (blueprint, neural, search), imported only when asked for.
   Importing the module may itself ``@register`` the name; otherwise the
   attribute is used directly.

With a positional argument, a factory that has a ``from_checkpoint``
classmethod is called as ``cls.from_checkpoint(arg, **kwargs)``; otherwise
as ``cls(*args, **kwargs)``.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .base import Agent

_REGISTRY: dict[str, Callable[..., Any]] = {}

LAZY_AGENTS: dict[str, str] = {
    "blueprint": "pokerbot.blueprint.mccfr.agent:BlueprintAgent",
    "neural": "pokerbot.blueprint.deepcfr.agent:NeuralBlueprintAgent",
    "search": "pokerbot.search.agent:SearchAgent",
    "uniform": "pokerbot.agents.policy:UniformPolicyAgent",
    "fixed": "pokerbot.agents.policy:FixedPolicyAgent",
}


class AgentSpecError(ValueError):
    pass


def register(name: str, *aliases: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Class or factory decorator: ``@register("blueprint")``."""

    def deco(factory: Callable[..., Any]) -> Callable[..., Any]:
        for n in (name, *aliases):
            _REGISTRY[n] = factory
        return factory

    return deco


@dataclass
class AgentSpec:
    name: str
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    text: str = ""

    @property
    def label(self) -> str:
        """Short display name: ``name`` or ``name:<file stem>`` for a path arg."""
        if not self.args and not self.kwargs:
            return self.name
        if (
            self.args
            and isinstance(self.args[0], str)
            and ("/" in self.args[0] or "." in self.args[0])
        ):
            return f"{self.name}:{Path(self.args[0]).stem}"
        return self.text


def _scalar(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def parse_spec(spec: str) -> AgentSpec:
    spec = spec.strip()
    if not spec:
        raise AgentSpecError("empty agent spec")
    name, _, rest = spec.partition(":")
    out = AgentSpec(name=name.strip(), text=spec)
    if rest:
        for item in rest.split(","):
            item = item.strip()
            if not item:
                continue
            key, eq, value = item.partition("=")
            if eq and key.strip().isidentifier():
                out.kwargs[key.strip()] = _scalar(value.strip())
            else:
                out.args.append(item)
    return out


def _resolve_target(target: str) -> Callable[..., Any]:
    mod_name, _, attr = target.partition(":")
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as e:
        raise AgentSpecError(
            f"agent module {mod_name!r} could not be imported ({e}); "
            "is that component installed / merged on this branch?"
        ) from e
    try:
        return getattr(mod, attr)
    except AttributeError:
        raise AgentSpecError(f"{mod_name!r} has no attribute {attr!r}") from None


def get_factory(name: str) -> Callable[..., Any]:
    if name in _REGISTRY:
        return _REGISTRY[name]
    from . import AGENTS

    if name in AGENTS:
        return AGENTS[name]
    if name in LAZY_AGENTS:
        factory = _resolve_target(LAZY_AGENTS[name])
        return _REGISTRY.get(name, factory)
    raise AgentSpecError(f"unknown agent {name!r}; choose from {available_agents()}")


def available_agents() -> list[str]:
    from . import AGENTS

    return sorted(set(_REGISTRY) | set(AGENTS) | set(LAZY_AGENTS))


def make_agent(spec: str | AgentSpec, **defaults: Any) -> Agent:
    """Construct an agent from a spec string. ``defaults`` are keyword
    arguments (e.g. from a config file) that the spec's own ``key=value``
    items override."""
    s = parse_spec(spec) if isinstance(spec, str) else spec
    factory = get_factory(s.name)
    kwargs = {**defaults, **s.kwargs}
    if s.args and hasattr(factory, "from_checkpoint"):
        return factory.from_checkpoint(*s.args, **kwargs)
    return factory(*s.args, **kwargs)

"""Run configuration for Deep CFR training (YAML -> dataclasses).

See ``configs/deepcfr_tiny.yaml`` (CPU smoke test) and
``configs/deepcfr_4070ti.yaml`` (the real run) for every key with comments.
Unknown keys are an error so typos do not silently fall back to defaults.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from ...env.actions import DEFAULT_SPEC, ActionSpec, spec_from_lists
from ...env.config import GameConfig
from .features import FeatureConfig
from .networks import NetConfig
from .traversal import TraversalConfig


@dataclass
class TraversalRunConfig:
    traversals_per_iter: int = 1024  # root hands per player per CFR iteration
    roots_per_batch: int = 256  # root hands per frontier batch
    max_frontier_nodes: int = 131072
    max_depth: int = 64
    max_steps: int = 64
    value_scale: float | None = None  # chips per value unit (default: big blind)
    record_strategy: bool = False  # SD-CFR does not need a strategy memory
    infer_chunk: int = 65536  # rows per network call inside a frontier step
    allin_equity: bool = False  # score all-ins called before the river by their equity
    chance_cv: float = 0.0  # beta of the street-change control variate (0 = off)
    preflop_equity_samples: int = 1024  # Monte Carlo runouts for the preflop equity


@dataclass
class MemoryConfig:
    capacity: int = 40_000_000  # advantage samples per player
    strategy_capacity: int = 0  # per player, only with record_strategy
    save_every: int = 1  # iterations between memory snapshots (resume points); 0 = never
    holdout: float = 0.0  # fraction of regret samples kept out of training (validation)
    holdout_capacity: int = 400_000  # per player, reservoir of held-out samples


@dataclass
class TrainingConfig:
    sgd_steps: int = 4000
    batch_size: int = 10000
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    reinit: bool = True  # re-initialize the advantage net every iteration (paper)
    bf16: bool = True  # bf16 autocast on CUDA (always fp32 on CPU)
    prefetch: int = 2  # minibatches prepared ahead by a host thread (0 = inline)
    # > 0: move this many rows to the device at once and slice minibatches there
    # (one host gather per chunk_rows / batch_size steps; replaces prefetch)
    chunk_rows: int = 0
    ema_decay: float = 0.0  # > 0: use an exponential moving average of the weights
    checkpoint_dtype: str = "float32"  # float32 | float16 | bfloat16
    # regret matching when no legal action has a positive advantage: "argmax"
    # (the Deep CFR paper; uniform was ~50% more exploitable in its ablation)
    # or "uniform"
    fallback: str = "argmax"


@dataclass
class EvalConfig:
    every: int = 0  # iterations between evaluations (0 = off)
    deals: int = 200  # duplicate deals per match (2 hands each)
    last_n: int | None = 32  # SD-CFR nets averaged per seat during evaluation
    engine: str = "auto"
    device: str | None = None  # device of the evaluated nets; None = training device
    vs_equity: bool = True
    vs_previous: bool = True
    equity: dict[str, Any] = field(default_factory=dict)  # EquityThresholdAgent kwargs
    seed: int | None = 0  # deal seed, the same for every evaluation; None = iteration number
    sample_net: bool = True  # play the SD-CFR average by sampling one net per hand
    luck_adjust: bool = True  # also report all-in EV + chance-corrected results


@dataclass
class LoggingConfig:
    tensorboard: bool = True
    csv: bool = True
    print: bool = True


@dataclass
class DeepCFRConfig:
    seed: int = 0
    device: str = "auto"  # auto | cpu | cuda | cuda:N
    threads: int | None = None  # torch intra-op CPU threads (None = torch default)
    iterations: int = 100
    out_dir: str = "runs/deepcfr"
    game: dict[str, Any] = field(default_factory=dict)
    actions: dict[str, Any] | None = None  # {streets: [[...] x4], max_raises, dedupe}
    features: FeatureConfig = field(default_factory=FeatureConfig)
    network: dict[str, Any] = field(default_factory=dict)  # NetConfig overrides
    traversal: TraversalRunConfig = field(default_factory=TraversalRunConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # ---------------------------------------------------------------- derived
    def torch_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def game_config(self) -> GameConfig:
        g = {"stacks": [20000, 20000], "small_blind": 50, "big_blind": 100, "ante": 0}
        g.update(self.game or {})
        stacks = g["stacks"]
        if isinstance(stacks, int):
            stacks = [stacks, stacks]
        return GameConfig(
            num_players=2,
            stacks=[int(s) for s in stacks],
            small_blind=int(g["small_blind"]),
            big_blind=int(g["big_blind"]),
            ante=int(g.get("ante", 0)),
        )

    def spec(self) -> ActionSpec:
        return spec_from_dict(self.actions)

    def net_config(self) -> NetConfig:
        spec = self.spec()
        A = spec.num_actions
        base = dict(
            num_actions=A,
            vocab_size=1 + 8 * A,
            history_len=self.features.history_len,
            num_scalars=self.features.num_scalars,
        )
        base.update(self.network or {})
        return NetConfig(**base)

    def traversal_config(self) -> TraversalConfig:
        t = self.traversal
        return TraversalConfig(
            max_frontier_nodes=t.max_frontier_nodes,
            max_depth=t.max_depth,
            max_steps=t.max_steps,
            value_scale=t.value_scale,
            record_strategy=t.record_strategy,
            allin_equity=t.allin_equity,
            chance_cv=t.chance_cv,
            preflop_equity_samples=t.preflop_equity_samples,
            features=self.features,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> DeepCFRConfig:
        d = dict(d or {})
        d.pop("engine", None)  # tolerated top-level key shared with other configs
        sub = {
            "features": FeatureConfig,
            "traversal": TraversalRunConfig,
            "memory": MemoryConfig,
            "training": TrainingConfig,
            "eval": EvalConfig,
            "logging": LoggingConfig,
        }
        kwargs = {}
        names = {f.name for f in dataclasses.fields(DeepCFRConfig)}
        for k, v in d.items():
            if k not in names:
                raise ValueError(f"unknown deepcfr config key {k!r}")
            if k in sub:
                kwargs[k] = _build(sub[k], v, k)
            else:
                kwargs[k] = v
        return DeepCFRConfig(**kwargs)

    @staticmethod
    def load(path: str | Path) -> DeepCFRConfig:
        with open(path) as fh:
            return DeepCFRConfig.from_dict(yaml.safe_load(fh) or {})


def _build(cls: type, v: Any, where: str) -> Any:
    if isinstance(v, cls):
        return v
    v = dict(v or {})
    names = {f.name for f in dataclasses.fields(cls)}
    bad = set(v) - names
    if bad:
        raise ValueError(f"unknown keys in {where}: {sorted(bad)}")
    return cls(**v)


def spec_to_dict(spec: ActionSpec) -> dict[str, Any]:
    return {
        "streets": [[list(a) for a in st] for st in spec.streets],
        "max_raises": spec.max_raises,
        "dedupe": spec.dedupe,
    }


def spec_from_dict(d: dict[str, Any] | None) -> ActionSpec:
    if not d:
        return DEFAULT_SPEC
    streets = [[tuple(a) for a in st] for st in d["streets"]]
    return spec_from_lists(streets, int(d.get("max_raises", 4)), bool(d.get("dedupe", True)))

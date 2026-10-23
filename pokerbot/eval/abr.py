"""Approximate best response (ABR) on the torch vectorized environment.

A best-response learner is trained against a frozen opponent for a fixed
budget and then evaluated against it. What it wins is a lower bound on the
opponent's exploitability that, unlike LBR, can find multi-street lines.

Learner: double DQN with a dueling head over the abstract actions of
:mod:`pokerbot.env.actions`, on :func:`pokerbot.env.obs.encode_obs`
features (card, rank and suit embeddings; a bag of betting-history tokens;
the scalar pot/stack features; optionally Monte Carlo equity vs a random
hand, which makes learning much faster). Episodes are single hands, rewards
are the learner's net chips at the end of the hand (``gamma = 1``).
Transitions go from one learner decision to its next decision in the same
hand (the opponent's actions in between are part of the environment).

Every slot of a :class:`~pokerbot.env.VecNLHE` batch holds one hand. The
learner sits in seat ``slot % 2``; the button alternates per slot on every
re-deal, so it plays both positions. Each env step, slots where the
learner acts take its action and the rest take the opponent's.

Opponents implement :class:`VecPolicy`: ``act(env, mask) -> LongTensor[n]``
(abstract action per slot; only ``mask`` slots matter). Provided:
:class:`UniformRandomVecPolicy`, :class:`CallVecPolicy`, and
:class:`ScalarVecPolicy`, which runs any scalar ``PolicyAgent`` slot by slot
(slow; for tests and small checks). A network policy can implement ``act``
directly on ``env.obs()``; agents expose it through a ``vec_policy(device)``
method, which :func:`make_vec_policy` prefers (the neural blueprint does).
The envs use the opponent's ``spec`` attribute when it has one
(``DEFAULT_SPEC`` otherwise): a policy's abstract actions and the history
tokens it reads are defined by its own action spec.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..agents.policy import abstract_actions, as_policy_agent, policy_vector
from ..engine_select import get_engine
from ..env.cards import NO_CARD, make_generator
from ..env.obs import NUM_SCALARS
from ..env.vec_env import VecNLHE
from .logging import RunLogger
from .stats import WinRate, win_rate

# --------------------------------------------------------------------------- opponents


@runtime_checkable
class VecPolicy(Protocol):
    def act(self, env: VecNLHE, mask: torch.Tensor | None = None) -> torch.Tensor: ...


def _sample_legal(legal: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    p = legal.float()
    p = torch.where(p.sum(1, keepdim=True) > 0, p, torch.ones_like(p))
    return torch.multinomial(p, 1, generator=generator).squeeze(1)


class UniformRandomVecPolicy:
    """Uniform over the legal abstract actions of each slot."""

    name = "uniform"

    def __init__(self, device: torch.device | str = "cpu", seed: int = 0) -> None:
        self.device = torch.device(device)
        self.generator = make_generator(seed, self.device)

    def reseed(self, seed: int) -> None:
        self.generator.manual_seed(int(seed))

    def act(self, env: VecNLHE, mask: torch.Tensor | None = None) -> torch.Tensor:
        return _sample_legal(env.legal_mask(), self.generator)


class CallVecPolicy:
    """Always check/call."""

    name = "call"

    def act(self, env: VecNLHE, mask: torch.Tensor | None = None) -> torch.Tensor:
        return env.tab.call_index[env.street.clamp(0, 3)]


def engine_config(env_config: Any, engine: ModuleType) -> Any:
    return engine.GameConfig(
        num_players=2,
        stacks=[int(s) for s in env_config.stacks],
        small_blind=int(env_config.small_blind),
        big_blind=int(env_config.big_blind),
        ante=int(getattr(env_config, "ante", 0)),
    )


def slot_state(env: VecNLHE, i: int, engine: ModuleType | None = None, config: Any = None) -> Any:
    """Scalar ``GameState`` equal to slot ``i`` of ``env`` (replays the deal and
    the abstract history tokens). Requires an unfilled history buffer."""
    engine = engine or get_engine()
    config = config or engine_config(env.config, engine)
    n_tok = int(env.hist_len[i])
    if n_tok >= env.history_len:
        raise ValueError("history buffer full; cannot reconstruct the slot")
    st = engine.GameState.new_hand(config, int(env.button[i]), env.deck[i].tolist())
    A = env.num_actions
    for tok in env.hist_tok[i, :n_tok].tolist():
        idx = (int(tok) - 1) % A
        choice = next(c for c in abstract_actions(st, env.spec) if c.index == idx)
        st.apply(choice.action(engine))
    return st


class ScalarVecPolicy:
    """Runs a scalar :class:`~pokerbot.agents.policy.PolicyAgent` on every
    masked slot (one ``policy`` call per slot per step: slow path)."""

    def __init__(self, agent: Any, seed: int = 0, engine: ModuleType | None = None) -> None:
        self.agent = agent
        self.spec = getattr(agent, "spec", None)
        self.name = getattr(agent, "name", type(agent).__name__)
        self.rng = np.random.default_rng(seed)
        self.engine = engine or get_engine()

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, env: VecNLHE, mask: torch.Tensor | None = None) -> torch.Tensor:
        from .masking import MaskedState

        legal = env.legal_mask()
        out = env.tab.call_index[env.street.clamp(0, 3)].clone()
        live = ~env.done if mask is None else (mask & ~env.done)
        idx = live.nonzero().squeeze(1).tolist()
        if not idx:
            return out
        config = engine_config(env.config, self.engine)
        legal_np = legal.cpu().numpy()
        for i in idx:
            st = slot_state(env, i, self.engine, config)
            seat = st.current_player
            view = MaskedState(st, seat, config, self.rng)
            p = policy_vector(self.agent.policy(view, seat), legal_np[i])
            out[i] = int(self.rng.choice(len(p), p=p))
        return out


def learner_spec_from(d: dict[str, Any] | None) -> Any:
    """``ActionSpec`` from an ``abr.learner_actions`` mapping (``streets``: four
    lists of actions such as ``[raise, 0.33]``, optional ``max_raises`` and
    ``dedupe``), or None for the opponent's own abstraction."""
    if not d:
        return None
    from ..env.actions import spec_from_lists

    streets = [[tuple(a) for a in st] for st in d["streets"]]
    return spec_from_lists(streets, int(d.get("max_raises", 4)), bool(d.get("dedupe", True)))


class LearnerView:
    """How the ABR learner observes and acts in an env built on the opponent's
    action abstraction.

    Without ``spec`` the learner shares the opponent's abstraction: it sees the
    env's observations and plays the env's abstract actions (the default).

    With a richer ``spec`` the learner chooses among ``spec``'s actions, which
    are applied as concrete chip amounts. The env, and so the opponent, records
    each learner action as the opponent would translate it in real play
    (``offtree="harmonic"``: the randomized pseudo-harmonic mapping of the
    scalar agents; ``"nearest"``: the nearest abstract size). The learner's
    observation keeps its own history of the real actions in ``spec`` tokens
    (opponent actions by their nearest ``spec`` size) and ``spec``'s legal mask.
    """

    def __init__(
        self,
        env_spec: Any,
        spec: Any = None,
        offtree: str = "harmonic",
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        if offtree not in ("harmonic", "nearest"):
            raise ValueError(f"offtree must be 'harmonic' or 'nearest', got {offtree!r}")
        self.rich = spec is not None
        self.spec = spec if spec is not None else env_spec
        self.num_actions = self.spec.num_actions
        self.vocab_size = 1 + 8 * self.num_actions
        self.offtree = offtree
        self.device = torch.device(device)
        self.tab = self.spec.tables(self.device) if self.rich else None
        self.gen = make_generator(seed, self.device)
        self.tok: torch.Tensor | None = None
        self.len: torch.Tensor | None = None

    def _sync(self, env: VecNLHE) -> None:
        if self.tok is None or self.tok.shape != env.hist_tok.shape:
            self.tok = torch.zeros_like(env.hist_tok)
            self.len = torch.zeros_like(env.hist_len)
        fresh = env.hist_len == 0  # a new hand in the slot
        self.tok[fresh] = 0
        self.len = torch.where(fresh, torch.zeros_like(self.len), self.len)

    def _rich_info(self, env: VecNLHE, info: Any) -> tuple[torch.Tensor, torch.Tensor]:
        from ..env import actions as am

        tr = am.raise_targets(
            self.tab,
            info.street,
            info.pot,
            info.max_bet,
            info.to_call,
            info.min_raise_to,
            info.max_raise_to,
        )
        mr = am.legal_mask(
            self.tab,
            info.street,
            info.active,
            info.to_call,
            info.raise_ok,
            env.n_raises,
            tr,
            info.max_raise_to,
        )
        return tr, mr

    @torch.no_grad()
    def obs(self, env: VecNLHE, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """The learner's view of ``obs`` (``env.obs()``)."""
        if not self.rich:
            return obs
        self._sync(env)
        _, mr = self._rich_info(env, env.legal_info())
        return {**obs, "hist": self.tok.clone(), "legal": mr}

    @torch.no_grad()
    def step(
        self, env: VecNLHE, br_turn: torch.Tensor, a_br: torch.Tensor, a_opp: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Step ``env`` with the learner's actions where ``br_turn`` and the
        opponent's elsewhere."""
        if not self.rich:
            return env.step(torch.where(br_turn, a_br, a_opp))
        from ..env import actions as am
        from ..env.actions import RAISE

        self._sync(env)
        info = env.legal_info()
        st = info.street
        te = env._targets(info)
        me = env._mask(info, te)
        tr, mr = self._rich_info(env, info)
        # learner: spec index -> concrete (illegal -> check/call)
        a_l = a_br.to(env.device).long().clamp(0, self.num_actions - 1)
        ok_l = mr.gather(1, a_l[:, None]).squeeze(1)
        a_l = torch.where(ok_l, a_l, self.tab.call_index[st])
        kind_l = self.tab.concrete[st].gather(1, a_l[:, None]).squeeze(1)
        amt_l = torch.where(kind_l == RAISE, tr.gather(1, a_l[:, None]).squeeze(1), 0)
        # opponent: env index -> concrete
        a_o = a_opp.to(env.device).long().clamp(0, env.num_actions - 1)
        kind_o = env.tab.concrete[st].gather(1, a_o[:, None]).squeeze(1)
        amt_o = torch.where(kind_o == RAISE, te.gather(1, a_o[:, None]).squeeze(1), 0)
        kind = torch.where(br_turn, kind_l, kind_o)
        amount = torch.where(br_turn, amt_l, amt_o)
        # what the env (the opponent) records for learner actions
        if self.offtree == "harmonic":
            u = torch.rand(env.n, device=env.device, generator=self.gen)
            env_idx = am.harmonic_abstract(
                env.tab,
                st,
                kind,
                amount,
                te,
                me,
                info.pot,
                info.max_bet,
                info.to_call,
                info.max_raise_to,
                u,
            )
        else:
            env_idx = am.nearest_abstract(
                env.tab, st, kind, amount, te, legal=me, max_raise_to=info.max_raise_to
            )
        record = torch.where(br_turn, env_idx, a_o)
        # the learner's own history: real actions in spec tokens
        opp_idx = am.nearest_abstract(
            self.tab, st, kind_o, amt_o, tr, legal=mr, max_raise_to=info.max_raise_to
        )
        idx = torch.where(br_turn, a_l, opp_idx)
        pos = (env.actor == env.button).long()
        tok = 1 + (st * 2 + pos) * self.num_actions + idx
        T = self.tok.shape[1]
        write = info.active & (self.len < T)
        at = (torch.arange(T, device=env.device)[None, :] == self.len[:, None]) & write[:, None]
        self.tok = torch.where(at, tok[:, None], self.tok)
        self.len = self.len + write.long()
        return env.step_concrete(kind, amount, record=record)


def opponent_spec(opponent: Any, spec: Any = None) -> Any:
    """The action spec for envs that ``opponent`` plays in: ``spec`` when given,
    else the opponent's ``spec`` attribute, else ``DEFAULT_SPEC``."""
    from ..env.actions import DEFAULT_SPEC

    return spec or getattr(opponent, "spec", None) or DEFAULT_SPEC


def make_vec_policy(
    spec: str,
    device: torch.device | str = "cpu",
    seed: int = 0,
    agent_params: dict[str, Any] | None = None,
    samples: int = 16,
) -> Any:
    """``uniform`` and ``call`` map to the vectorized policies; any other
    registry spec is built as an agent and used through ``vec_policy(device)``
    when it has one, else through :class:`ScalarVecPolicy`."""
    from ..agents.registry import make_agent, parse_spec

    name = parse_spec(spec).name
    if spec == "uniform":
        return UniformRandomVecPolicy(device, seed)
    if spec in ("call", "always_call"):
        return CallVecPolicy()
    agent = make_agent(spec, **((agent_params or {}).get(name) or {}))
    if hasattr(agent, "vec_policy"):
        return agent.vec_policy(device=device)
    return ScalarVecPolicy(as_policy_agent(agent, samples=samples), seed)


# --------------------------------------------------------------------------- learner


@dataclass
class ABRConfig:
    n_envs: int = 256
    train_steps: int = 400
    buffer_size: int = 100_000
    batch_size: int = 256
    updates_per_step: int = 1
    learning_starts: int = 1_000
    lr: float = 5e-4
    gamma: float = 1.0
    target_update: int = 100
    grad_clip: float = 10.0
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 300
    hidden: list[int] = field(default_factory=lambda: [128, 128])
    card_dim: int = 32
    hist_dim: int = 32
    equity_samples: int = 32
    reward_scale_bb: float = 20.0
    eval_hands: int = 4_096
    eval_envs: int = 1_024
    eval_initial: bool = True
    eval_every: int = 0
    log_every: int = 50
    seed: int = 0
    # the learner's own action set (None = the opponent's abstraction); see LearnerView
    learner_actions: dict[str, Any] | None = None
    offtree: str = "harmonic"  # how the opponent records learner actions off its tree

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> ABRConfig:
        d = dict(d or {})
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown abr config keys: {sorted(unknown)}")
        return cls(**d)


class DuelingQNet(nn.Module):
    def __init__(
        self,
        num_actions: int,
        vocab: int,
        n_scalars: int = NUM_SCALARS,
        n_extra: int = 0,
        card_dim: int = 32,
        hist_dim: int = 32,
        hidden: tuple[int, ...] | list[int] = (128, 128),
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.card = nn.Embedding(NO_CARD + 1, card_dim, padding_idx=NO_CARD)
        self.rank = nn.Embedding(14, card_dim, padding_idx=13)
        self.suit = nn.Embedding(5, card_dim, padding_idx=4)
        self.hist = nn.Embedding(vocab, hist_dim, padding_idx=0)
        layers: list[nn.Module] = []
        d = 2 * card_dim + hist_dim + n_scalars + n_extra
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.body = nn.Sequential(*layers)
        self.value = nn.Linear(d, 1)
        self.adv = nn.Linear(d, num_actions)

    def forward(
        self,
        cards: torch.Tensor,
        hist: torch.Tensor,
        scalars: torch.Tensor,
        extra: torch.Tensor,
        legal: torch.Tensor,
    ) -> torch.Tensor:
        c = cards.long()
        pad = c == NO_CARD
        e = (
            self.card(c)
            + self.rank(torch.where(pad, 13, torch.div(c, 4, rounding_mode="floor")))
            + self.suit(torch.where(pad, 4, c % 4))
        )
        hole, board = e[:, :2].sum(1), e[:, 2:].sum(1)
        h = hist.long()
        hcount = (h != 0).sum(1, keepdim=True).clamp(min=1)
        hbag = self.hist(h).sum(1) / hcount
        z = self.body(torch.cat([hole, board, hbag, scalars.float(), extra.float()], 1))
        adv = self.adv(z)
        lm = legal.float()
        adv_mean = (adv * lm).sum(1, keepdim=True) / lm.sum(1, keepdim=True).clamp(min=1)
        return self.value(z) + adv - adv_mean


FEATURES = ("cards", "hist", "scalars", "extra", "legal")


def features(obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    extra = []
    if "equity" in obs:
        extra.append(obs["equity"][:, None])
    if "equity_hist" in obs:
        extra.append(obs["equity_hist"])
    n = obs["cards"].shape[0]
    ex = torch.cat(extra, 1) if extra else torch.zeros(n, 0, device=obs["cards"].device)
    return {
        "cards": obs["cards"],
        "hist": obs["hist"],
        "scalars": obs["scalars"],
        "extra": ex,
        "legal": obs["legal"],
    }


def q_values(net: DuelingQNet, f: dict[str, torch.Tensor]) -> torch.Tensor:
    return net(f["cards"], f["hist"], f["scalars"], f["extra"], f["legal"])


def greedy(q: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    return q.masked_fill(~legal, float("-inf")).argmax(1)


class ReplayBuffer:
    """Uniform ring buffer of transitions stored on ``device``."""

    _DTYPES = {
        "cards": torch.uint8,
        "hist": torch.uint8,
        "scalars": torch.float32,
        "extra": torch.float32,
        "legal": torch.bool,
    }

    def __init__(self, capacity: int, like: dict[str, torch.Tensor], device: torch.device) -> None:
        self.capacity = int(capacity)
        self.device = device
        self.size = 0
        self.added = 0
        self.pos = 0
        self.s: dict[str, torch.Tensor] = {}
        self.s2: dict[str, torch.Tensor] = {}
        for k in FEATURES:
            shape = (self.capacity, *like[k].shape[1:])
            self.s[k] = torch.zeros(shape, dtype=self._DTYPES[k], device=device)
            self.s2[k] = torch.zeros(shape, dtype=self._DTYPES[k], device=device)
        self.a = torch.zeros(self.capacity, dtype=torch.long, device=device)
        self.r = torch.zeros(self.capacity, dtype=torch.float32, device=device)
        self.d = torch.zeros(self.capacity, dtype=torch.bool, device=device)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        s: dict[str, torch.Tensor],
        a: torch.Tensor,
        r: torch.Tensor,
        d: torch.Tensor,
        s2: dict[str, torch.Tensor] | None,
    ) -> None:
        m = a.shape[0]
        if m == 0:
            return
        idx = (self.pos + torch.arange(m, device=self.device)) % self.capacity
        for k in FEATURES:
            self.s[k][idx] = s[k].to(self._DTYPES[k])
            if s2 is None:
                self.s2[k][idx] = 0
            else:
                self.s2[k][idx] = s2[k].to(self._DTYPES[k])
        self.a[idx] = a
        self.r[idx] = r.float()
        self.d[idx] = d
        self.pos = (self.pos + m) % self.capacity
        self.size = min(self.capacity, self.size + m)
        self.added += m

    def sample(self, batch: int, generator: torch.Generator | None = None) -> tuple:
        idx = torch.randint(0, self.size, (batch,), device=self.device, generator=generator)
        s = {k: v[idx] for k, v in self.s.items()}
        s2 = {k: v[idx] for k, v in self.s2.items()}
        return s, self.a[idx], self.r[idx], self.d[idx], s2


# --------------------------------------------------------------------------- evaluation


@dataclass
class BREval:
    stats: WinRate
    sb: float
    bb: float
    hands: int


@torch.no_grad()
def evaluate_br(
    policy_fn: Any,
    opponent: Any,
    game: Any,
    hands: int = 4096,
    n_envs: int = 1024,
    device: torch.device | str = "cpu",
    seed: int = 12345,
    equity_samples: int = 0,
    spec: Any = None,
    learner_spec: Any = None,
    offtree: str = "harmonic",
) -> BREval:
    """Play ``policy_fn(obs) -> actions`` (in seat ``slot % 2``) against
    ``opponent`` for ``hands`` hands in rounds of ``n_envs`` hands. With the
    same ``seed`` two evaluations see the same deals (common random numbers).
    ``spec`` defaults to the opponent's ``spec`` (else ``DEFAULT_SPEC``).
    ``learner_spec`` / ``offtree``: the learner's own action set (see
    :class:`LearnerView`); ``policy_fn`` then sees the learner's view."""
    n = max(2, min(int(n_envs), int(hands)))
    env = VecNLHE(n, game, device, seed=seed, validate=False, spec=opponent_spec(opponent, spec))
    gen = make_generator(seed + 1, env.device)
    view = LearnerView(env.spec, learner_spec, offtree, env.device, seed + 3)
    if hasattr(opponent, "reseed"):
        opponent.reseed(seed + 2)
    br_seat = torch.arange(n, device=env.device) % 2
    rounds = math.ceil(hands / n)
    pay, pos = [], []
    for r in range(rounds):
        if r:
            env.reset()
        while not bool(env.done.all()):
            obs = view.obs(env, env.obs(equity_samples=equity_samples, generator=gen))
            live = ~env.done
            br_turn = live & (obs["actor"] == br_seat)
            a_br = policy_fn(obs)
            a_opp = opponent.act(env, live & ~br_turn)
            view.step(env, br_turn, a_br, a_opp)
        pay.append(env.payoffs.gather(1, br_seat[:, None]).squeeze(1).cpu())
        pos.append((env.button == br_seat).cpu())
    x = torch.cat(pay)[:hands].double().numpy()
    is_sb = torch.cat(pos)[:hands].numpy()
    bbv = int(game.big_blind)
    stats = win_rate(x, bbv, rng=seed)

    def mbb(m: np.ndarray) -> float:
        return float(1000.0 * x[m].mean() / bbv) if m.any() else float("nan")

    return BREval(stats, mbb(is_sb), mbb(~is_sb), int(len(x)))


def greedy_policy(net: DuelingQNet) -> Any:
    def fn(obs: dict[str, torch.Tensor]) -> torch.Tensor:
        f = features(obs)
        return greedy(q_values(net, f), f["legal"])

    return fn


# --------------------------------------------------------------------------- training


@dataclass
class ABRResult:
    opponent: str
    initial: BREval | None
    final: BREval
    steps: int
    transitions: int
    seconds: float
    curve: list[dict[str, float]]
    config: dict[str, Any]

    def summary(self) -> str:
        lines = [f"ABR vs {self.opponent}: {self.final.stats}"]
        lines.append(f"  by position: SB {self.final.sb:+.1f}, BB {self.final.bb:+.1f} mbb/h")
        if self.initial is not None:
            lines.append(f"  untrained learner: {self.initial.stats}")
        lines.append(
            f"  {self.steps} env steps, {self.transitions} transitions, {self.seconds:.1f}s"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        def ev(e: BREval | None) -> dict[str, Any] | None:
            if e is None:
                return None
            return {
                "mbb": e.stats.mbb_per_hand,
                "ci_low": e.stats.ci_low,
                "ci_high": e.stats.ci_high,
                "sb_mbb": e.sb,
                "bb_mbb": e.bb,
                "hands": e.hands,
            }

        return {
            "opponent": self.opponent,
            "final": ev(self.final),
            "initial": ev(self.initial),
            "steps": self.steps,
            "transitions": self.transitions,
            "seconds": self.seconds,
            "curve": self.curve,
            "config": self.config,
        }


def resolve_device(device: str | torch.device | None) -> torch.device:
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def train_abr(
    cfg: ABRConfig,
    opponent: Any,
    game: Any = None,
    device: str | torch.device | None = "cpu",
    logger: RunLogger | None = None,
    progress: Any = None,
    checkpoint: str | Path | None = None,
) -> tuple[DuelingQNet, ABRResult]:
    """Train a best response to ``opponent`` (a :class:`VecPolicy`) and evaluate it."""
    from ..env.config import GameConfig

    t0 = time.time()
    dev = resolve_device(device)
    game = game if game is not None else GameConfig()
    torch.manual_seed(cfg.seed)
    spec = opponent_spec(opponent)
    env = VecNLHE(cfg.n_envs, game, dev, seed=cfg.seed, validate=False, spec=spec)
    gen = make_generator(cfg.seed + 1, dev)
    lspec = learner_spec_from(cfg.learner_actions)
    if lspec is not None and isinstance(opponent, ScalarVecPolicy):
        raise ValueError(
            "learner_actions needs a vectorized opponent: ScalarVecPolicy rebuilds each "
            "slot from its abstract history, which off-tree learner raises make inexact"
        )
    view = LearnerView(env.spec, lspec, cfg.offtree, dev, cfg.seed + 3)
    n = env.n
    br_seat = torch.arange(n, device=dev) % 2
    if hasattr(opponent, "reseed"):
        opponent.reseed(cfg.seed + 2)
    n_extra = 1 if cfg.equity_samples > 0 else 0
    net = DuelingQNet(
        view.num_actions,
        view.vocab_size,
        NUM_SCALARS,
        n_extra,
        cfg.card_dim,
        cfg.hist_dim,
        cfg.hidden,
    ).to(dev)
    target = copy.deepcopy(net)
    target.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    opp_name = str(getattr(opponent, "name", type(opponent).__name__))
    eval_kw = dict(
        hands=cfg.eval_hands,
        n_envs=cfg.eval_envs,
        device=dev,
        seed=cfg.seed + 777,
        equity_samples=cfg.equity_samples,
        spec=spec,
        learner_spec=lspec,
        offtree=cfg.offtree,
    )
    initial = None
    if cfg.eval_initial:
        net.eval()
        initial = evaluate_br(greedy_policy(net), opponent, game, **eval_kw)
        if hasattr(opponent, "reseed"):
            opponent.reseed(cfg.seed + 2)
        if logger:
            logger.scalar("abr/eval_mbb", initial.stats.mbb_per_hand, 0)
        if progress:
            progress(f"[0] untrained: {initial.stats}")

    obs0 = features(view.obs(env, env.obs(equity_samples=cfg.equity_samples, generator=gen)))
    buf = ReplayBuffer(cfg.buffer_size, obs0, dev)
    pend = {k: torch.zeros_like(v) for k, v in obs0.items()}
    pend_a = torch.zeros(n, dtype=torch.long, device=dev)
    has_pend = torch.zeros(n, dtype=torch.bool, device=dev)
    scale = float(game.big_blind) * cfg.reward_scale_bb
    recent: list[torch.Tensor] = []
    curve: list[dict[str, float]] = []
    updates = 0
    loss_acc, loss_n = 0.0, 0
    for step in range(1, cfg.train_steps + 1):
        frac = min(1.0, (step - 1) / max(1, cfg.eps_decay_steps))
        eps = cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)
        net.eval()
        with torch.no_grad():
            f = features(view.obs(env, env.obs(equity_samples=cfg.equity_samples, generator=gen)))
            live = ~env.done
            br_turn = live & (env.actor == br_seat)
            comp = br_turn & has_pend
            if bool(comp.any()):
                zero = torch.zeros(int(comp.sum()), device=dev)
                buf.add(
                    {k: v[comp] for k, v in pend.items()},
                    pend_a[comp],
                    zero,
                    torch.zeros_like(zero, dtype=torch.bool),
                    {k: v[comp] for k, v in f.items()},
                )
            q = q_values(net, f)
            a_greedy = greedy(q, f["legal"])
            a_rand = _sample_legal(f["legal"], gen)
            explore = torch.rand(n, device=dev, generator=gen) < eps
            a_br = torch.where(explore, a_rand, a_greedy)
            a_opp = opponent.act(env, live & ~br_turn)
            for k in pend:
                pend[k][br_turn] = f[k][br_turn]
            pend_a = torch.where(br_turn, a_br, pend_a)
            has_pend = has_pend | br_turn
            payoffs, done = view.step(env, br_turn, a_br, a_opp)
            fin = done & live
            chips = payoffs.gather(1, br_seat[:, None]).squeeze(1)
            term = fin & has_pend
            if bool(term.any()):
                buf.add(
                    {k: v[term] for k, v in pend.items()},
                    pend_a[term],
                    chips[term].float() / scale,
                    torch.ones(int(term.sum()), dtype=torch.bool, device=dev),
                    None,
                )
            recent.append(chips[fin].float())
            has_pend = has_pend & ~fin
            env.reset(done)

        if len(buf) >= max(cfg.learning_starts, cfg.batch_size):
            net.train()
            for _ in range(cfg.updates_per_step):
                s, a, r, d, s2 = buf.sample(cfg.batch_size, gen)
                qsa = q_values(net, s).gather(1, a[:, None]).squeeze(1)
                with torch.no_grad():
                    a2 = greedy(q_values(net, s2), s2["legal"])
                    q2 = q_values(target, s2).gather(1, a2[:, None]).squeeze(1)
                    y = r + cfg.gamma * torch.where(d, torch.zeros_like(q2), q2)
                loss = F.smooth_l1_loss(qsa, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
                opt.step()
                updates += 1
                loss_acc += float(loss.detach())
                loss_n += 1
                if updates % cfg.target_update == 0:
                    target.load_state_dict(net.state_dict())

        if cfg.log_every and step % cfg.log_every == 0:
            r = torch.cat(recent) if recent else torch.zeros(0)
            ret = float(1000.0 * r.mean() / game.big_blind) if r.numel() else float("nan")
            point = {
                "step": step,
                "train_mbb": ret,
                "loss": loss_acc / max(1, loss_n),
                "eps": eps,
                "buffer": len(buf),
            }
            curve.append(point)
            if logger:
                logger.scalars(
                    {
                        "abr/train_mbb": ret,
                        "abr/loss": point["loss"],
                        "abr/eps": eps,
                        "abr/buffer": len(buf),
                    },
                    step,
                )
            if progress:
                progress(
                    f"[{step}] train {ret:+.0f} mbb/h (eps-greedy, {r.numel()} hands) "
                    f"loss {point['loss']:.4f} eps {eps:.2f} buffer {len(buf)}"
                )
            recent, loss_acc, loss_n = [], 0.0, 0
        if cfg.eval_every and step % cfg.eval_every == 0 and step < cfg.train_steps:
            net.eval()
            mid = evaluate_br(greedy_policy(net), opponent, game, **eval_kw)
            if logger:
                logger.scalar("abr/eval_mbb", mid.stats.mbb_per_hand, step)
            if progress:
                progress(f"[{step}] eval: {mid.stats}")

    net.eval()
    final = evaluate_br(greedy_policy(net), opponent, game, **eval_kw)
    res = ABRResult(
        opponent=opp_name,
        initial=initial,
        final=final,
        steps=cfg.train_steps,
        transitions=buf.added,
        seconds=time.time() - t0,
        curve=curve,
        config=asdict(cfg),
    )
    if logger:
        logger.scalar("abr/eval_mbb", final.stats.mbb_per_hand, cfg.train_steps)
        logger.write_json("abr.json", res.to_dict())
        logger.flush()
    if checkpoint:
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": net.state_dict(), "result": res.to_dict()}, checkpoint)
    return net, res


# --------------------------------------------------------------------------- command line


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Approximate best response against a frozen policy.")
    ap.add_argument("--config", help="YAML with game:, opponent:, device: and abr: sections")
    ap.add_argument("--opponent", help="uniform | call | agent spec")
    ap.add_argument("--device", help="auto | cpu | cuda")
    ap.add_argument("--steps", type=int, help="override abr.train_steps")
    ap.add_argument("--n-envs", type=int, help="override abr.n_envs")
    ap.add_argument("--eval-hands", type=int, help="override abr.eval_hands")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--threads", type=int, help="torch CPU threads (config: threads)")
    ap.add_argument("--out", help="write the result JSON here")
    ap.add_argument("--checkpoint", help="save the learned Q-network here")
    ap.add_argument("--log-dir")
    return ap


def main(argv: list[str] | None = None) -> int:
    from ..config import GAME_DEFAULTS, git_hash, load_yaml
    from ..env.config import GameConfig
    from .logging import write_json_atomic

    args = build_parser().parse_args(argv)
    cfg = load_yaml(args.config) if args.config else {}
    abr_d = dict(cfg.get("abr") or {})
    for key, val in (
        ("train_steps", args.steps),
        ("n_envs", args.n_envs),
        ("eval_hands", args.eval_hands),
        ("seed", args.seed),
    ):
        if val is not None:
            abr_d[key] = val
    acfg = ABRConfig.from_dict(abr_d)
    g = dict(GAME_DEFAULTS)
    g.update(cfg.get("game") or {})
    game = GameConfig(
        num_players=2,
        stacks=[
            int(s) for s in (g["stacks"] if isinstance(g["stacks"], list) else [g["stacks"]] * 2)
        ],
        small_blind=int(g["small_blind"]),
        big_blind=int(g["big_blind"]),
        ante=int(g["ante"]),
    )
    device = resolve_device(args.device or cfg.get("device", "auto"))
    threads = args.threads or cfg.get("threads")
    if threads:
        torch.set_num_threads(int(threads))
    opp_spec = args.opponent or cfg.get("opponent", "uniform")
    opponent = make_vec_policy(
        opp_spec, device, acfg.seed + 2, cfg.get("agents") or {}, int(cfg.get("samples", 16))
    )
    print(f"# git {git_hash()} | device {device} | opponent {opp_spec} | config {args.config}")
    logger = RunLogger(args.log_dir, config={"argv": argv or sys.argv[1:], **cfg})
    _, res = train_abr(
        acfg, opponent, game, device, logger, progress=print, checkpoint=args.checkpoint
    )
    logger.close()
    res.opponent = opp_spec
    print(res.summary())
    if args.out:
        write_json_atomic(args.out, {**res.to_dict(), "git": git_hash()})
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""River-root states for the leaf value network (``docs/value_net.md``, section 3).

A **river state** is the public state at the start of the river, before any
river action: a 5-card board, the chips ``c`` each player has committed (equal
at the river root), the chips behind ``stack`` (start stack minus ``c``) and
both players' ranges ``[2, 1326]`` in ``(OOP, IP)`` order, i.e. (non-button
seat, button seat). Each range sums to 1 and is zero on combos that share a
card with the board. States are dicts of CPU tensors with a leading sample dim:

* ``boards`` long ``[m, 5]``, ``c`` long ``[m]``, ``stack`` long ``[m]``,
  ``ranges`` float32 ``[m, 2, 1326]``;
* self-play states also carry ``button`` (the button seat), ``hist_len`` (the
  number of actions before the river) and the concrete actions
  ``hist_kind``/``hist_amount`` ``[m, L]`` (padded with -1 / 0) and the
  actual hole cards ``holes [m, 2, 2]`` in ``(OOP, IP)`` order.

Sources:

* :func:`selfplay_river_states`: blueprint self-play. Each actor samples its
  action from the row of its actual hole combo (mixed with ``explore`` uniform
  over the legal actions), and its 1326-combo reach is multiplied by the policy
  column of the chosen action, so the range is the public-belief range of
  :func:`~pokerbot.search.blueprint.range_reach` (zero on board conflicts
  only, never on the opponent's actual cards). A hand is kept when it reaches
  the river root with both players still having chips behind; hands that end
  earlier (folds, all-ins) are discarded.

  The fast path (single-net blueprints, :func:`.vec_rollouts.supports`) runs
  in two phases on ``VecNLHE``. Phase 1 plays ``n_envs`` hands in lockstep with
  one network row per slot (the actual hand only) and records the deal and the
  abstract actions of every hand that reaches the river root. Phase 2 replays
  only those hands in lockstep and computes the policy of every combo at every
  decision with :func:`.vec_rollouts.slot_policies` (bf16 on CUDA), in a few
  large network calls; preflop decisions are deduplicated by public state, and
  hand-strength columns are looked up once per hand and street. Discarded
  hands therefore never pay for the 1326-combo policies. Other blueprints
  (anything with ``policy_matrix``) use a scalar-engine path, one hand at a
  time.
* :func:`perturb_ranges`: log-normal per-combo noise, strength tilts, uniform
  mixing and random support removal applied to existing ranges.
* :func:`random_ranges`: DeepStack-style recursive random splits of the mass
  along the board-relative strength order, on :func:`random_boards` with
  :func:`random_c` (log-uniform).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch

from ..blueprint.deepcfr.features import features_from_obs, index_features
from ..blueprint.deepcfr.range_policy import ALL_COMBOS
from ..blueprint.deepcfr.strength import NUM_STRENGTH, add_strength, load_strength
from ..blueprint.deepcfr.traversal import deal_env
from ..env.cards import NO_CARD
from ..env.vec_env import VecNLHE
from . import vec_rollouts
from .combos import NUM_COMBOS, combo_index, incidence
from .showdown import combo_strengths

OOP, IP = 0, 1
MAX_ACTIONS = 64  # per hand before the river (3 raises per street keep it far below)
C_MIN, C_MAX = 100, 9900  # random_c range (100bb: one big blind .. 99bb committed)


# -- game config helpers ------------------------------------------------------


def _game(game_config: Any) -> dict[str, Any]:
    """``stacks``, ``small_blind``, ``big_blind``, ``ante`` of an engine or env
    ``GameConfig`` or of a plain dict (e.g. ``bp.game``)."""
    if isinstance(game_config, Mapping):
        g = dict(game_config)
    else:
        g = {k: getattr(game_config, k) for k in ("stacks", "small_blind", "big_blind")}
        g["ante"] = getattr(game_config, "ante", 0)
    return {
        "stacks": [int(s) for s in g["stacks"]],
        "small_blind": int(g["small_blind"]),
        "big_blind": int(g["big_blind"]),
        "ante": int(g.get("ante", 0) or 0),
    }


def vec_game_config(game_config: Any) -> Any:
    """The ``VecNLHE`` config of ``game_config``."""
    from ..env.config import GameConfig

    return GameConfig(num_players=2, **_game(game_config))


def engine_game_config(game_config: Any, engine: Any = None) -> Any:
    """The scalar-engine ``GameConfig`` of ``game_config``."""
    from ..engine_select import get_engine

    engine = engine or get_engine()
    return engine.GameConfig(num_players=2, **_game(game_config))


# -- self-play ------------------------------------------------------------------


def _lockstep_ok(bp: Any) -> bool:
    """Single-net blueprint whose features are a function of the public state
    and the hand (no Monte Carlo equity inputs), so every combo's policy can be
    computed from one encoded row."""
    if not vec_rollouts.supports(bp):
        return False
    fc = bp.agent.features
    return fc.equity_samples == 0 and fc.hist_runouts == 0


def _new_stats() -> dict[str, float]:
    return {
        "hands": 0,
        "kept": 0,
        "folded": 0,
        "allin": 0,
        "empty_range": 0,
        "phase1_s": 0.0,
        "phase2_s": 0.0,
        "seconds": 0.0,
    }


@torch.no_grad()
def selfplay_river_states(
    bp: Any,
    game_config: Any,
    n: int,
    device: torch.device | str = "cpu",
    seed: int = 0,
    explore: float = 0.0,
    n_envs: int = 4096,
    bf16: bool | None = None,
    replay_chunk: int = 1024,
    slots_per_call: int = 64,
    stats: dict[str, float] | None = None,
    max_hands: int | None = None,
    log: Callable[[str], Any] | None = None,
) -> dict[str, torch.Tensor]:
    """``n`` river-root states from blueprint self-play (see the module docstring).

    ``stats`` (optional, filled in place) counts the hands played, kept
    (reached the river root without an all-in), folded and all-in before the
    river, ranges the blueprint made empty (replaced by uniform, as
    :func:`range_reach` does) and the seconds per phase. ``max_hands`` bounds
    the hands played (default ``1000 * n + 100000``): a blueprint that never
    reaches the river raises instead of looping forever.
    """
    stats = stats if stats is not None else {}
    stats.update({k: v for k, v in _new_stats().items() if k not in stats})
    t0 = time.time()
    dev = torch.device(device)
    if bf16 is None:
        bf16 = dev.type == "cuda"
    max_hands = int(max_hands) if max_hands is not None else 1000 * int(n) + 100_000
    if n <= 0:
        return _empty_states()
    if _lockstep_ok(bp):
        hands = _play_lockstep(
            bp, game_config, n, dev, seed, explore, n_envs, bf16, stats, max_hands, log
        )
        t1 = time.time()
        stats["phase1_s"] += t1 - t0
        out = _replay_ranges(bp, game_config, hands, dev, bf16, replay_chunk, slots_per_call, stats)
        stats["phase2_s"] += time.time() - t1
    else:
        out = _play_scalar(bp, game_config, n, seed, explore, stats, max_hands)
    stats["seconds"] += time.time() - t0
    return out


def _empty_states() -> dict[str, torch.Tensor]:
    z = torch.zeros(0, dtype=torch.long)
    return {
        "boards": torch.zeros(0, 5, dtype=torch.long),
        "c": z,
        "stack": z.clone(),
        "ranges": torch.zeros(0, 2, NUM_COMBOS),
        "button": z.clone(),
        "hist_len": z.clone(),
        "hist_kind": torch.zeros(0, 0, dtype=torch.long),
        "hist_amount": torch.zeros(0, 0, dtype=torch.long),
        "holes": torch.zeros(0, 2, 2, dtype=torch.long),
    }


def _actor_policy(
    agent: Any, feats: dict[str, torch.Tensor], seat: torch.Tensor, bf16: bool
) -> torch.Tensor:
    """``[k, A]`` policy of each row's own hand for the row's seat, normalised
    over the legal actions (uniform where a row has no legal mass)."""
    k = seat.numel()
    legal = feats["legal"].float()
    P = torch.zeros(k, legal.shape[1], device=legal.device)
    for s in (0, 1):
        rows = (seat == s).nonzero().squeeze(1)
        if rows.numel() == 0:
            continue
        sub = {key: v[rows] for key, v in feats.items()}
        with torch.autocast(legal.device.type, dtype=torch.bfloat16, enabled=bf16):
            P[rows] = agent.policies[s].net_policies(sub)[0].float().to(legal.device)
    P = P.clamp(min=0) * legal
    tot = P.sum(-1, keepdim=True)
    uni = legal / legal.sum(-1, keepdim=True)
    return torch.where(tot > 0, P / tot.clamp(min=1e-30), uni)


def _play_lockstep(
    bp: Any,
    game_config: Any,
    n: int,
    dev: torch.device,
    seed: int,
    explore: float,
    n_envs: int,
    bf16: bool,
    stats: dict[str, float],
    max_hands: int,
    log: Callable[[str], Any] | None,
) -> dict[str, torch.Tensor]:
    """Phase 1: lockstep self-play with one network row per decision (the
    actor's actual hand). Returns the deal and abstract actions of ``n`` hands
    that reach the river root without an all-in.

    Hands are numbered when dealt. Once ``n`` are kept no new hands are dealt,
    the hands in progress are played out, and the first ``n`` kept hands in
    deal order are returned: an i.i.d. sample of the kept hands. (Taking the
    first ``n`` arrivals instead would over-represent short betting lines.)
    """
    agent = bp.agent
    fc = agent.features
    tables = load_strength(fc.strength_tables) if fc.strength_tables else None
    env = VecNLHE(
        int(n_envs),
        vec_game_config(game_config),
        dev,
        seed=int(seed),
        spec=agent.spec,
        validate=False,
    )
    gen = torch.Generator(device=dev).manual_seed(int(seed) + 1)
    acts = torch.full((env.n, MAX_ACTIONS), -1, dtype=torch.long, device=dev)
    nact = torch.zeros(env.n, dtype=torch.long, device=dev)
    hand_id = torch.arange(env.n, device=dev)  # deal order of each slot's hand
    next_id = env.n
    keep: dict[str, list[torch.Tensor]] = {
        "deck": [],
        "button": [],
        "acts": [],
        "len": [],
        "id": [],
    }
    got, hands, last_log, dealing = 0, 0, time.time(), True
    counted = torch.zeros(env.n, dtype=torch.bool, device=dev)  # finished hand already counted
    while True:
        live = ~env.done
        root = live & (env.street == 3)  # just reached the river root: record and stop
        finished = (env.done & ~counted) | root
        if bool(finished.any()):
            ridx = root.nonzero().squeeze(1)
            if ridx.numel():
                keep["deck"].append(env.deck[ridx].long().cpu())
                keep["button"].append(env.button[ridx].cpu())
                keep["acts"].append(acts[ridx].cpu())
                keep["len"].append(nact[ridx].cpu())
                keep["id"].append(hand_id[ridx].cpu())
                got += int(ridx.numel())
            ended = finished & env.done
            stats["hands"] += int(finished.sum())
            stats["kept"] += int(ridx.numel())
            stats["folded"] += int((ended & env.folded.any(1)).sum())
            stats["allin"] += int((ended & ~env.folded.any(1)).sum())
            hands += int(finished.sum())
            env.done[root] = True  # recorded: the slot plays no river action
            counted |= finished
            if dealing and got >= n:
                dealing = False  # play out the hands in progress, deal no more
            if dealing:
                if hands >= max_hands:
                    raise RuntimeError(
                        f"self-play kept {got} of {n} river states after {hands} hands; "
                        "the blueprint rarely reaches the river"
                    )
                fidx = finished.nonzero().squeeze(1)
                env.reset(finished)
                acts[finished] = -1
                nact[finished] = 0
                counted[finished] = False
                hand_id[fidx] = next_id + torch.arange(fidx.numel(), device=dev)
                next_id += int(fidx.numel())
            if log and time.time() - last_log > 30:
                log(f"# self-play: {got}/{n} river states from {hands} hands")
                last_log = time.time()
        live = ~env.done
        idx = live.nonzero().squeeze(1)
        if idx.numel() == 0:
            if dealing:
                continue  # every new hand ended at the deal
            break
        feats = index_features(features_from_obs(env.obs(**fc.obs_kwargs())), idx)
        if tables is not None:
            feats = add_strength(feats, tables)
        P = _actor_policy(agent, feats, env.actor[idx], bf16)
        legal = feats["legal"].float()
        q = (1.0 - explore) * P + explore * legal / legal.sum(-1, keepdim=True)
        a = torch.multinomial(q, 1, generator=gen).squeeze(1)
        actions = env.tab.call_index[env.street.clamp(0, 3)].clone()
        actions[idx] = a
        if bool((nact[idx] >= MAX_ACTIONS).any()):
            raise RuntimeError(f"a hand exceeded {MAX_ACTIONS} actions before the river")
        acts[idx, nact[idx]] = a
        nact[idx] += 1
        env.step(actions)
    out = {k: torch.cat(v) for k, v in keep.items()}
    first = out.pop("id").argsort()[:n]
    return {k: v[first] for k, v in out.items()}


class _StrengthBank:
    """``[1326, 11]`` strength columns of every combo per (street, board
    prefix), looked up in one batch per replay chunk (the
    :class:`.vec_rollouts.StrengthCache` interface without unbounded growth)."""

    def __init__(self, tables: Any, device: torch.device) -> None:
        self.tables = tables
        self.device = device
        self._cache: dict[tuple, torch.Tensor] = {}
        self._holes = ALL_COMBOS.numpy()

    def clear(self) -> None:
        self._cache = {k: v for k, v in self._cache.items() if k[0] == 0}

    def fill(self, keys: list[tuple[int, tuple[int, ...]]]) -> None:
        keys = [k for k in dict.fromkeys(keys) if k not in self._cache]
        if not keys:
            return
        K = len(keys)
        holes = np.tile(self._holes, (K, 1))
        boards = np.full((K * NUM_COMBOS, 5), NO_CARD, dtype=np.int64)
        streets = np.empty(K * NUM_COMBOS, dtype=np.int64)
        for i, (street, board) in enumerate(keys):
            sl = slice(i * NUM_COMBOS, (i + 1) * NUM_COMBOS)
            h = holes[sl]
            clash = np.isin(h, np.asarray(board, dtype=np.int64)).any(1)
            free = [c for c in range(52) if c not in board][:2]
            h[clash] = free  # placeholder rows (zero reach anyway)
            boards[sl, : len(board)] = board
            streets[sl] = street
        cols = self.tables.lookup(holes, boards, streets)
        out = torch.from_numpy(cols).to(self.device, torch.float32).view(K, NUM_COMBOS, -1)
        for i, k in enumerate(keys):
            self._cache[k] = out[i]

    def get(self, board: tuple[int, ...], street: int) -> torch.Tensor:
        key = (int(street), tuple(int(c) for c in board))
        if key not in self._cache:
            self.fill([key])
        return self._cache[key]


def _policy_columns(
    agent: Any,
    env: Any,
    a: torch.Tensor,
    boards: torch.Tensor,
    bank: _StrengthBank | None,
    bf16: bool,
    slots_per_call: int,
) -> torch.Tensor:
    """``[n, 1326]``: every combo's blueprint probability of the action ``a``
    taken in each slot (the actor's policy column). Preflop slots with the same
    public state share one network evaluation."""
    dev = env.device
    feats = features_from_obs(env.obs(**agent.features.obs_kwargs()))
    out = torch.empty(env.n, NUM_COMBOS, device=dev)
    pre = env.street == 0
    # (slots to evaluate, slots that use them, row of each user in the evaluated slots)
    groups: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    pidx = pre.nonzero().squeeze(1)
    if pidx.numel():
        # preflop the policy of every combo is a function of the public state:
        # button, actor and the history (tokens and amounts)
        key = torch.cat(
            [env.button[pidx, None], env.actor[pidx, None], env.hist_tok[pidx], env.hist_amt[pidx]],
            1,
        )
        _, inv = torch.unique(key, dim=0, return_inverse=True)
        first = torch.full((int(inv.max()) + 1,), pidx.numel(), dtype=torch.long, device=dev)
        first.scatter_reduce_(0, inv, torch.arange(pidx.numel(), device=dev), "amin")
        groups.append((pidx[first], pidx, inv))
    post = (~pre).nonzero().squeeze(1)
    if post.numel():
        groups.append((post, post, torch.arange(post.numel(), device=dev)))
    for slots, users, row in groups:
        for lo in range(0, slots.numel(), slots_per_call):
            idx = slots[lo : lo + slots_per_call]
            P = vec_rollouts.slot_policies(agent, env, idx, feats, boards, bank, bf16)
            sel = ((row >= lo) & (row < lo + idx.numel())).nonzero().squeeze(1)
            u = users[sel]
            col = P[row[sel] - lo].gather(2, a[u, None, None].expand(-1, NUM_COMBOS, 1))
            out[u] = col.squeeze(2)
    return out


def _replay_ranges(
    bp: Any,
    game_config: Any,
    hands: dict[str, torch.Tensor],
    dev: torch.device,
    bf16: bool,
    chunk: int,
    slots_per_call: int,
    stats: dict[str, float],
) -> dict[str, torch.Tensor]:
    """Phase 2: replay the kept hands in lockstep and build both players'
    1326-combo reach from the policy columns of the actions taken."""
    agent = bp.agent
    spec = agent.spec
    fc = agent.features
    tables = load_strength(fc.strength_tables) if fc.strength_tables else None
    bank = _StrengthBank(tables, dev) if tables is not None else None
    gc = vec_game_config(game_config)
    N = hands["deck"].shape[0]
    L_all = hands["len"]
    Lmax = int(L_all.max()) if N else 0
    parts: list[dict[str, torch.Tensor]] = []
    for lo in range(0, N, chunk):
        decks = hands["deck"][lo : lo + chunk].to(dev)
        buttons = hands["button"][lo : lo + chunk].to(dev)
        acts = hands["acts"][lo : lo + chunk, :Lmax].to(dev)
        L = L_all[lo : lo + chunk].to(dev)
        K = decks.shape[0]
        env = deal_env(decks, buttons, gc, spec, dev)
        boards = decks[:, 4:9].long()
        reach = torch.ones(K, 2, NUM_COMBOS, dtype=torch.float64, device=dev)
        kinds = torch.full((K, Lmax), -1, dtype=torch.long, device=dev)
        amounts = torch.zeros(K, Lmax, dtype=torch.long, device=dev)
        c = torch.zeros(K, dtype=torch.long, device=dev)
        stack = torch.zeros(K, dtype=torch.long, device=dev)
        ids = torch.arange(K, device=dev)
        if bank is not None:  # flop and turn columns of every hand in one lookup
            keys = []
            for b in boards.tolist():
                keys += [(1, tuple(b[:3])), (2, tuple(b[:4]))]
            bank.fill(keys)
        j = 0
        while ids.numel():
            fin = L[ids] == j
            if bool(fin.any()):
                f = fin.nonzero().squeeze(1)
                at_root = (env.street[f] == 3) & ~env.done[f]
                contrib = env.contrib[f]
                if not bool(at_root.all()) or not bool((contrib[:, 0] == contrib[:, 1]).all()):
                    raise RuntimeError("a replayed hand did not end at the river root")
                c[ids[f]] = contrib[:, 0]
                stack[ids[f]] = env.stacks[f].min(1).values
                keep = (~fin).nonzero().squeeze(1)
                env = env.select(keep)
                ids = ids[keep]
                if ids.numel() == 0:
                    break
            a = acts[ids, j]
            col = _policy_columns(agent, env, a, boards[ids], bank, bf16, slots_per_call)
            seat = env.actor.clone()
            reach[ids, seat] = reach[ids, seat] * col.double()
            env.step(a)
            kinds[ids, j] = env.last_kind
            amounts[ids, j] = torch.where(env.last_kind == 2, env.last_amount, 0)
            j += 1
        if bank is not None:
            bank.clear()
        ranges = _finish_ranges(reach, boards, buttons, stats)
        parts.append(
            {
                "boards": boards.cpu(),
                "c": c.cpu(),
                "stack": stack.cpu(),
                "ranges": ranges.cpu(),
                "button": buttons.cpu(),
                "hist_len": L.cpu(),
                "hist_kind": kinds.cpu(),
                "hist_amount": amounts.cpu(),
                "holes": _holes_oop_ip(decks, buttons).cpu(),
            }
        )
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


def _holes_oop_ip(decks: torch.Tensor, buttons: torch.Tensor) -> torch.Tensor:
    """``[K, 2, 2]`` hole cards in ``(OOP, IP)`` order from deal decks (seat 0's
    cards first, then seat 1's)."""
    seat = torch.stack([decks[:, 0:2], decks[:, 2:4]], 1).long()  # [K, 2 seats, 2]
    oop = 1 - buttons.long()
    ar = torch.arange(decks.shape[0], device=decks.device)
    return torch.stack([seat[ar, oop], seat[ar, 1 - oop]], 1)


def _finish_ranges(
    reach: torch.Tensor, boards: torch.Tensor, buttons: torch.Tensor, stats: dict[str, float]
) -> torch.Tensor:
    """Seat reach ``[K, 2, 1326]`` -> normalised ``(OOP, IP)`` float32 ranges,
    zero on board conflicts; an empty range becomes uniform over the valid
    combos (as :func:`range_reach` does)."""
    K = reach.shape[0]
    valid = board_valid(boards.to(reach.device))  # [K, 1326]
    r = torch.where(valid[:, None], torch.nan_to_num(reach, nan=0.0), 0.0)
    oop = 1 - buttons.long()
    ar = torch.arange(K, device=reach.device)
    r = torch.stack([r[ar, oop], r[ar, 1 - oop]], 1)
    tot = r.sum(-1, keepdim=True)
    empty = tot <= 0
    stats["empty_range"] = stats.get("empty_range", 0) + int(empty.sum())
    uni = valid[:, None].double().expand_as(r)
    r = torch.where(empty, uni, r)
    return (r / r.sum(-1, keepdim=True)).float()


def _play_scalar(
    bp: Any,
    game_config: Any,
    n: int,
    seed: int,
    explore: float,
    stats: dict[str, float],
    max_hands: int,
) -> dict[str, torch.Tensor]:
    """Scalar-engine self-play for any blueprint with ``policy_matrix``, one
    hand at a time (slow; for tests and blueprints without the lockstep path)."""
    from ..engine_select import get_engine
    from .abstract import contributions, legal_options, to_action
    from .blueprint import policy_matrix

    engine = get_engine()
    config = engine_game_config(game_config, engine)
    rng = np.random.default_rng(int(seed))
    spec = bp.spec
    rows: list[dict[str, Any]] = []
    hands = 0
    while len(rows) < n:
        if hands >= max_hands:
            raise RuntimeError(
                f"self-play kept {len(rows)} of {n} river states after {hands} hands"
            )
        hands += 1
        deck = rng.permutation(52).tolist()
        button = int(rng.integers(2))
        state = engine.GameState.new_hand(config, button, deck)
        reach = torch.ones(2, NUM_COMBOS, dtype=torch.float64)
        hist: list[tuple[int, int]] = []
        while True:
            if state.is_terminal:
                if any(state.folded):
                    stats["folded"] += 1
                else:
                    stats["allin"] += 1
                break
            if int(state.street) == 3 and not any(int(s) == 3 for s, _p, _a in state.history):
                if min(int(s) for s in state.stacks) <= 0:
                    raise RuntimeError("river root with a player all-in")
                contrib = contributions(state, config)
                rows.append(
                    {
                        "board": [int(x) for x in state.board],
                        "c": int(contrib[0]),
                        "stack": min(int(s) for s in state.stacks),
                        "reach": reach,
                        "button": button,
                        "deck": deck,
                        "hist": hist,
                    }
                )
                stats["kept"] += 1
                break
            p = int(state.current_player)
            opts = legal_options(state, spec)
            P = policy_matrix(bp, state, p, device="cpu").double()
            hole = list(state.hole_cards(p))
            row = P[combo_index(hole[0], hole[1])] if P.shape[0] > 1 else P[0]
            q = np.array([float(row[o.index]) for o in opts])
            q = (1.0 - explore) * q / max(q.sum(), 1e-300) + explore / len(opts)
            o = opts[int(rng.choice(len(opts), p=q / q.sum()))]
            reach[p] = reach[p] * P[:, o.index]
            hist.append((o.kind, o.amount if o.kind == 2 else 0))
            state.apply(to_action(engine, o.kind, o.amount))
        stats["hands"] += 1
    L = max((len(r["hist"]) for r in rows), default=0)
    kinds = torch.full((n, L), -1, dtype=torch.long)
    amounts = torch.zeros(n, L, dtype=torch.long)
    for i, r in enumerate(rows):
        for j, (k, amt) in enumerate(r["hist"]):
            kinds[i, j] = k
            amounts[i, j] = amt
    boards = torch.tensor([r["board"] for r in rows], dtype=torch.long)
    buttons = torch.tensor([r["button"] for r in rows], dtype=torch.long)
    reach = torch.stack([r["reach"] for r in rows])
    out = {
        "boards": boards,
        "c": torch.tensor([r["c"] for r in rows], dtype=torch.long),
        "stack": torch.tensor([r["stack"] for r in rows], dtype=torch.long),
        "ranges": _finish_ranges(reach, boards, buttons, stats),
        "button": buttons,
        "hist_len": torch.tensor([len(r["hist"]) for r in rows], dtype=torch.long),
        "hist_kind": kinds,
        "hist_amount": amounts,
        "holes": _holes_oop_ip(torch.tensor([r["deck"] for r in rows]), buttons),
    }
    return out


# -- strength percentiles and synthetic ranges ----------------------------------


def board_valid(boards: torch.Tensor) -> torch.Tensor:
    """``[n, k]`` long boards -> ``[n, 1326]`` bool: combos disjoint from the
    board (:func:`~.combos.valid_masks` without a Python loop over boards)."""
    b = boards.long()
    onehot = torch.zeros(b.shape[0], 52, device=b.device).scatter_(1, b, 1.0)
    return (onehot @ incidence(b.device, torch.float32).t()) == 0


def strength_pct(boards: torch.Tensor) -> torch.Tensor:
    """``[n, 5]`` boards -> ``[n, 1326]`` board-relative strength percentile of
    every combo among the valid combos: ``(valid strictly weaker + (tied valid
    - 1) / 2) / (valid - 1)``, so the nuts get 1 and the worst hand 0 (tied
    combos share the midpoint). Zero for combos that conflict with the board."""
    s = combo_strengths(boards.long())  # -1 for board conflicts
    valid = s >= 0
    V = valid.sum(1, keepdim=True)
    ss, _ = s.sort(1)
    lo = torch.searchsorted(ss, s, right=False)
    hi = torch.searchsorted(ss, s, right=True)
    weaker = (lo - (NUM_COMBOS - V)).double()
    ties = (hi - lo).double()
    pct = (weaker + (ties - 1) / 2) / (V - 1).clamp(min=1).double()
    return torch.where(valid, pct, 0.0).float()


def random_boards(
    n: int, generator: torch.Generator | None = None, device: Any = None
) -> torch.Tensor:
    """``[n, 5]`` long: uniformly random river boards."""
    dev = generator.device if generator is not None and device is None else device
    return torch.rand(n, 52, generator=generator, device=dev).argsort(1)[:, :5]


def reachable_c(c: torch.Tensor, big_blind: int = 100) -> torch.Tensor:
    """Snap river-root commitments to amounts a hand can reach: a limped pot
    commits one big blind, and every bet or raise is at least one big blind,
    so ``c`` is ``bb`` or at least ``2 * bb``. Values in between go to the
    nearer of the two in log space; values below ``bb`` go to ``bb``."""
    bb = int(big_blind)
    c = torch.as_tensor(c).long()
    low = c.double() < math.sqrt(2.0) * bb
    gap = (c > bb) & (c < 2 * bb)
    c = torch.where(gap, torch.where(low, bb, 2 * bb), c)
    return c.clamp(min=bb)


def random_c(
    n: int,
    generator: torch.Generator | None = None,
    lo: int = C_MIN,
    hi: int = C_MAX,
    device: Any = None,
    big_blind: int = 100,
) -> torch.Tensor:
    """``[n]`` long: chips committed per player, log-uniform in ``[lo, hi]``,
    then snapped to reachable amounts (:func:`reachable_c`: no ``c`` strictly
    between one and two big blinds, which no river root has)."""
    dev = generator.device if generator is not None and device is None else device
    u = torch.rand(n, generator=generator, device=dev, dtype=torch.float64)
    x = torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
    return reachable_c(x.round().long().clamp(lo, hi), big_blind)


def random_ranges(boards: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """``[n, 1326]`` DeepStack-style random ranges on ``[n, 5]`` boards.

    The valid combos are sorted by strength (random order within ties); the
    mass is then split recursively: a node covering a contiguous block of the
    sorted order gives a ``U(0, 1)`` fraction of its mass to its first half and
    the rest to its second half, down to single combos. One level of the
    recursion is one vectorised step over all boards and combos (a node is
    identified by its first position, which draws its fraction).
    """
    dev = boards.device
    n = boards.shape[0]
    s = combo_strengths(boards.long()).double()
    tie = torch.rand(n, NUM_COMBOS, generator=generator, device=dev, dtype=torch.float64)
    order = (s + 0.5 * tie).argsort(1)  # board conflicts (-1) first, then by strength
    V = (s >= 0).sum(1, keepdim=True)
    pos = torch.arange(NUM_COMBOS, device=dev)[None].expand(n, -1)
    lo = (NUM_COMBOS - V).expand(n, NUM_COMBOS).clone()
    hi = torch.full_like(lo, NUM_COMBOS)
    mass = torch.ones(n, NUM_COMBOS, dtype=torch.float64, device=dev)
    while True:
        size = hi - lo
        split = size >= 2
        if not bool(split.any()):
            break
        mid = lo + size // 2
        u = torch.rand(n, NUM_COMBOS, generator=generator, device=dev, dtype=torch.float64)
        u = u.gather(1, lo.clamp(max=NUM_COMBOS - 1))
        first = pos < mid
        mass = mass * torch.where(split, torch.where(first, u, 1.0 - u), 1.0)
        lo = torch.where(split & ~first, mid, lo)
        hi = torch.where(split & first, mid, hi)
    mass = torch.where(pos >= NUM_COMBOS - V, mass, 0.0)
    r = torch.zeros_like(mass).scatter_(1, order, mass)
    return _normalise(r, s >= 0).float()


def _normalise(r: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Rows normalised to sum 1, zero off ``valid``; an empty row becomes uniform."""
    r = torch.where(valid, r, 0.0)
    tot = r.sum(-1, keepdim=True)
    uni = valid.to(r.dtype)
    r = torch.where(tot > 0, r, uni)
    return r / r.sum(-1, keepdim=True)


def perturb_ranges(
    ranges: torch.Tensor,
    boards: torch.Tensor,
    generator: torch.Generator | None = None,
    p_noise: float = 0.75,
    p_tilt: float = 0.5,
    p_mix: float = 0.5,
    p_zero: float = 0.25,
    max_sigma: float = 1.5,
    tilt_std: float = 2.0,
    max_mix: float = 0.3,
    max_zero: float = 0.5,
) -> torch.Tensor:
    """Random perturbations of ``ranges`` (``[n, 1326]`` or ``[n, k, 1326]``,
    one board per row of ``boards [n, 5]``); every range gets its own draws.

    Each range independently gets (with the given probabilities; at least the
    noise when none is drawn):

    * per-combo log-normal noise ``exp(sigma * N(0, 1))``, ``sigma ~ U(0, 1.5)``;
    * a strength tilt ``exp(a * pct)`` with ``a ~ N(0, 2)`` (``pct`` from
      :func:`strength_pct`);
    * mixing with the uniform range over the valid combos, weight ``U(0, 0.3)``;
    * zeroing a random ``U(0, 0.5)`` fraction of its support (skipped when it
      would empty the range).

    Returns normalised float32 ranges of the input shape, zero on board conflicts.
    """
    shape = ranges.shape
    dev = ranges.device
    k = shape[1] if ranges.dim() == 3 else 1
    r = ranges.reshape(-1, NUM_COMBOS).double()
    N = r.shape[0]
    b = boards.long().to(dev)
    pct = strength_pct(b).double().repeat_interleave(k, 0)
    valid = board_valid(b).repeat_interleave(k, 0)

    def U(*size: int) -> torch.Tensor:
        return torch.rand(*size, generator=generator, device=dev, dtype=torch.float64)

    def G(*size: int) -> torch.Tensor:
        return torch.randn(*size, generator=generator, device=dev, dtype=torch.float64)

    use_noise, use_tilt, use_mix, use_zero = (U(N) < p for p in (p_noise, p_tilt, p_mix, p_zero))
    use_noise |= ~(use_noise | use_tilt | use_mix | use_zero)
    sigma = U(N) * max_sigma * use_noise
    a = G(N) * tilt_std * use_tilt
    log_f = sigma[:, None] * G(N, NUM_COMBOS) + a[:, None] * pct
    log_f = log_f - log_f.max(1, keepdim=True).values  # overflow-safe; rescaling is harmless
    r = _normalise(r * torch.exp(log_f), valid)
    lam = U(N) * max_mix * use_mix
    r = (1.0 - lam[:, None]) * r + lam[:, None] * valid.double() / valid.sum(1, keepdim=True)
    frac = U(N) * max_zero
    drop = (U(N, NUM_COMBOS) < frac[:, None]) & (r > 0) & use_zero[:, None]
    r2 = torch.where(drop, 0.0, r)
    r = torch.where((r2.sum(1, keepdim=True) > 0), r2, r)
    return _normalise(r, valid).float().reshape(shape)


def random_states(
    n: int,
    generator: torch.Generator | None = None,
    start_stack: int = 10000,
    c_lo: int = C_MIN,
    c_hi: int = C_MAX,
    big_blind: int = 100,
) -> dict[str, torch.Tensor]:
    """``n`` river states with random boards, log-uniform ``c`` (:func:`random_c`)
    and independent :func:`random_ranges` for both players (on the generator's
    device)."""
    boards = random_boards(n, generator)
    c = random_c(n, generator, c_lo, c_hi, big_blind=big_blind)
    r0 = random_ranges(boards, generator)
    r1 = random_ranges(boards, generator)
    return {
        "boards": boards,
        "c": c,
        "stack": int(start_stack) - c,
        "ranges": torch.stack([r0, r1], 1),
    }


def range_summary(ranges: torch.Tensor, eps: float = 1e-6) -> dict[str, torch.Tensor]:
    """Per range (``[..., 1326]``): entropy in nats, its exponential (the
    effective number of combos) and the support size (mass above ``eps``)."""
    r = ranges.double()
    ent = -(r * torch.log(r.clamp(min=1e-300))).sum(-1)
    return {"entropy": ent, "effective": ent.exp(), "support": (r > eps).sum(-1)}


__all__ = [
    "C_MAX",
    "C_MIN",
    "IP",
    "NUM_STRENGTH",
    "OOP",
    "board_valid",
    "engine_game_config",
    "perturb_ranges",
    "random_boards",
    "random_c",
    "random_ranges",
    "random_states",
    "reachable_c",
    "range_summary",
    "selfplay_river_states",
    "strength_pct",
    "vec_game_config",
]

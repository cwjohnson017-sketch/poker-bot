"""The scalar feature encoder must reproduce ``VecNLHE.obs()`` exactly."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from pokerbot.blueprint.deepcfr.features import features_from_obs
from pokerbot.blueprint.deepcfr.scalar import (
    ScalarSpec,
    decision_info,
    encode_state,
    engine_config,
    nearest_abstract,
    raise_targets,
)
from pokerbot.env import DEFAULT_SPEC, GameConfig, VecNLHE
from pokerbot.env.actions import legal_mask as torch_legal_mask
from pokerbot.env.actions import raise_targets as torch_raise_targets
from pokerbot.eval.masking import MaskedState

ENGINES = ["pokerbot.reference"]
try:
    importlib.import_module("poker_engine")
    ENGINES.append("poker_engine")
except ImportError:  # pragma: no cover
    pass


def play_env(cfg, n, seed):
    """Random abstract play; per slot the features at every decision and the
    concrete actions the env applied."""
    env = VecNLHE(n, cfg, "cpu", seed=seed)
    g = torch.Generator().manual_seed(seed)
    recs = [[] for _ in range(n)]
    acts = [[] for _ in range(n)]
    decks = env.deck.long().tolist()
    buttons = env.button.tolist()
    while not bool(env.done.all()):
        f = features_from_obs(env.obs())
        live = (~env.done).nonzero().squeeze(1).tolist()
        for i in live:
            recs[i].append(({k: v[i] for k, v in f.items()}, len(acts[i])))
        # bias towards raises so histories get long
        w = f["legal"].float() * torch.tensor([0.3, 1.0, 1.0, 1.0, 1.0, 0.5])
        env.step(torch.multinomial(w, 1, generator=g).squeeze(1))
        for i in live:
            acts[i].append((int(env.last_kind[i]), int(env.last_amount[i])))
    return recs, acts, decks, buttons


@pytest.mark.parametrize("engine_name", ENGINES)
@pytest.mark.parametrize("stacks", [[20000, 20000], [3000, 7000]])
def test_scalar_encoder_matches_env(engine_name, stacks):
    engine = importlib.import_module(engine_name)
    cfg = GameConfig(stacks=stacks)
    ecfg = engine_config(engine, cfg)
    recs, acts, decks, buttons = play_env(cfg, 120, seed=len(stacks) + stacks[1])
    rng = np.random.default_rng(0)
    checked = 0
    max_len = 0
    for i in range(len(recs)):
        for feats, k in recs[i]:
            s = engine.GameState.new_hand(ecfg, buttons[i], decks[i])
            for kind, amt in acts[i][:k]:
                s.apply(engine.Action(kind, amt if kind == 2 else 0))
            seat = s.current_player
            view = MaskedState(s, seat, ecfg, rng) if checked % 2 else s
            got, info = encode_state(view, seat, ecfg, DEFAULT_SPEC)
            for key, want in feats.items():
                assert torch.equal(got[key][0], want), (i, k, key, got[key][0], want)
            max_len = max(max_len, k)
            checked += 1
    assert checked > 300 and max_len >= 6


def test_scalar_action_tables_match_torch():
    tab = DEFAULT_SPEC.tables("cpu")
    sp = ScalarSpec.build(DEFAULT_SPEC)
    rng = np.random.default_rng(1)
    n = 400
    street = torch.tensor(rng.integers(0, 4, n))
    pot = torch.tensor(rng.integers(150, 30000, n))
    max_bet = torch.tensor(rng.integers(0, 8000, n))
    to_call = (max_bet * torch.tensor(rng.random(n))).long()
    mn = max_bet + torch.tensor(rng.integers(100, 3000, n))
    mx = mn + torch.tensor(rng.integers(-500, 12000, n))
    raise_ok = torch.tensor(rng.random(n) < 0.8)
    mn = torch.where(raise_ok, mn, 0)
    mx = torch.where(raise_ok, mx.clamp(min=1), 0)
    n_raises = torch.tensor(rng.integers(0, 5, n))
    tt = torch_raise_targets(tab, street, pot, max_bet, to_call, mn, mx)
    tm = torch_legal_mask(
        tab, street, torch.ones(n, dtype=torch.bool), to_call, raise_ok, n_raises, tt, mx
    )
    from pokerbot.blueprint.deepcfr.scalar import legal_mask

    for i in range(n):
        args = [int(x[i]) for x in (street, pot, max_bet, to_call, mn, mx)]
        t = raise_targets(sp, *args)
        assert t == tt[i].tolist()
        m = legal_mask(sp, args[0], args[3], bool(raise_ok[i]), int(n_raises[i]), t, args[5])
        assert m == tm[i].tolist()


def test_off_tree_bet_maps_to_nearest_legal_size():
    engine = importlib.import_module("pokerbot.reference")
    cfg = engine_config(engine, GameConfig())
    sp = ScalarSpec.build(DEFAULT_SPEC)
    s = engine.GameState.new_hand(cfg, 0, list(range(52)))
    s.apply(engine.Action.check_call())  # SB limps
    s.apply(engine.Action.check_call())  # BB checks -> flop, pot 200
    info = decision_info(s, sp, 0)
    # flop sizes: 0.33 pot = 66, 0.75 = 150, 1.5 = 300 (clamped to min 100)
    assert info.targets[2:5] == [100, 150, 300]
    assert nearest_abstract(sp, 1, 2, 120, info.targets, info.legal) == 2
    assert nearest_abstract(sp, 1, 2, 130, info.targets, info.legal) == 3
    assert nearest_abstract(sp, 1, 2, 900, info.targets, info.legal) == 4
    assert nearest_abstract(sp, 1, 2, 19900, info.targets, info.legal) == 5  # all-in
    # the opponent bets an off-tree 130; the encoder records the 0.75 token
    s.apply(engine.Action.raise_to(130))
    feats, _ = encode_state(s, s.current_player, cfg, DEFAULT_SPEC)
    A = DEFAULT_SPEC.num_actions
    street, is_btn = 1, int(s.history[-1][1] == s.button)
    assert int(feats["hist"][0, 2]) == 1 + (street * 2 + is_btn) * A + 3
    assert float(feats["hist_amt"][0, 2]) == pytest.approx(130 / 20000)


def test_allin_maps_to_the_allin_entry_even_when_raises_are_capped():
    """Mirror of ``VecNLHE.step_concrete``: a raise to exactly the all-in is
    the ``allin`` token, never a sized raise that clamps to the same amount
    (legal or not, e.g. past the raise cap)."""
    sp = ScalarSpec.build(DEFAULT_SPEC)
    targets = [0, 0, 600, 900, 1100, 1100]  # 1.5 pot clamps to the 1100 all-in
    legal = [True, True, True, True, False, True]
    assert nearest_abstract(sp, 1, 2, 1100, targets, legal) == 5
    capped = [True, True, False, False, False, False]
    assert nearest_abstract(sp, 1, 2, 1100, targets, capped) == 5
    assert nearest_abstract(sp, 1, 2, 950, targets, capped) == 3

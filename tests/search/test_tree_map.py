"""Strategy translation between trees of one public root (``tree_map``): the
identity, a coarse flop tree onto a richer one (the searcher's nodes copy
exactly, an opponent's off-tree bet lands in the right subtree, later actions
match by abstract action), lost mass the other way round, and unmatched nodes."""

from __future__ import annotations

import pytest
import torch

from pokerbot.engine_select import get_engine
from pokerbot.env.actions import ActionSpec
from pokerbot.search import spot_eval as se
from pokerbot.search.abstract import CHECK_CALL, RAISE, legal_options, map_concrete
from pokerbot.search.combos import NUM_COMBOS, valid_mask
from pokerbot.search.solver import RangeSolver, SolverConfig
from pokerbot.search.tree import DECISION, VALUE, TreeConfig, build_tree
from pokerbot.search.tree_map import (
    StrategySnapshot,
    _Info,
    action_set_differences,
    match_nodes,
    offtree_edges,
    path_nodes,
    subtree_nodes,
    translate_action,
    translate_sigma,
)
from pokerbot.search.value_leaf import FixedLeafValues

PRE = (("fold",), ("check_call",), ("raise_x", 2.5), ("allin",))
SHORT = (("fold",), ("check_call",), ("allin",))


def _spec(*sizes: float) -> ActionSpec:
    flop = (("fold",), ("check_call",), *(("raise", f) for f in sizes), ("allin",))
    return ActionSpec(streets=(PRE, flop, SHORT, SHORT), max_raises=2)


COARSE = _spec(0.5, 1.5)
RICH = _spec(0.5, 0.75, 1.0, 1.5)
BB, BTN = 1, 0  # "BB first": the big blind (seat 1) searches, the button is the opponent


def _spot():
    engine = get_engine()
    cfg = engine.GameConfig(num_players=2, stacks=[10000, 10000], small_blind=50, big_blind=100)
    (spot,) = se.exploit_spots(engine, cfg, 1, 5, ["bb_first"])
    return engine, cfg, spot.state


def _solver(spec: ActionSpec, seed: int = 0) -> RangeSolver:
    _, cfg, s = _spot()
    tc = TreeConfig(spec=spec, max_nodes=10**6, chance_cards=2, leaf_mode="value_net")
    tree = build_tree(cfg, s.button, s.board, s.history, tc)
    g = torch.Generator().manual_seed(seed)
    ranges = torch.rand(2, NUM_COMBOS, generator=g) * valid_mask(s.board)
    L = int((tree.kind == VALUE).sum())
    leaves = FixedLeafValues(torch.randn(2, L, NUM_COMBOS, generator=g) * 50)
    return RangeSolver(tree, ranges, SolverConfig(allin_mode="runouts"), value_leaves=leaves)


def _random_sigma(solver: RangeSolver, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    s = torch.rand(solver.regret.shape, generator=g) * solver.legal[:, :, None]
    return s / s.sum(1, keepdim=True)


def _node(tree, history: tuple, board_len: int = 3) -> int:
    """The decision node with concrete history ``history`` ((kind, amount) pairs)."""
    for n in (tree.kind == DECISION).nonzero().flatten().tolist():
        h = tuple((k, a if k == RAISE else 0) for _s, _p, k, a in tree.histories[n])
        if h == history and len(tree.boards[int(tree.board_id[n])]) == board_len:
            return n
    raise KeyError(history)


@pytest.fixture(scope="module")
def coarse_rich():
    src, dst = _solver(COARSE), _solver(RICH)
    sig = _random_sigma(src, 1)
    return src, dst, sig


def test_identity_is_exact():
    a = _solver(COARSE)
    sig = _random_sigma(a, 3)
    out, rep = translate_sigma(a, a, BB, sig)
    assert torch.equal(out, sig)
    assert rep["translated_nodes"] == {"searcher": 0, "opponent": 0}
    assert rep["unmatched_nodes"] == {"searcher": 0, "opponent": 0}
    assert rep["lost_mass"]["searcher"]["nodes"] == 0 and set(rep["fill"]) == {"exact"}
    # a separately built copy of the tree, with a solved average strategy
    b = _solver(COARSE)
    a.solve(iterations=3)
    out, _ = translate_sigma(a, b, BB)
    assert torch.equal(out, a.average_strategy())
    snap = StrategySnapshot.of(a)
    out2, _ = translate_sigma(snap, b, BB)
    assert torch.equal(out2, out)


def test_translate_action_follows_map_concrete_when_states_agree():
    _, _, s = _spot()
    tree = _solver(COARSE).tree
    kids_of_root = _Info(tree).kids(0)
    opts = legal_options(s, COARSE)
    la = s.legal_actions()
    checked = 0
    for amount in range(int(la.min_raise_to), int(la.max_raise_to) + 1, 37):
        j, rule = translate_action(RAISE, amount, None, s, s, kids_of_root, True)
        dist = map_concrete(s, COARSE, RAISE, amount, opts)
        amt = {o.index: o.amount for o in opts}
        best = max(dist, key=lambda i: (dist[i], amt[i]))  # the larger size on a tie
        assert kids_of_root[j].amount == amt[best], amount
        assert rule in ("exact", "harmonic", "allin")
        checked += 1
    assert checked > 100
    j, rule = translate_action(CHECK_CALL, 0, None, s, s, kids_of_root, True)
    assert rule == "exact" and kids_of_root[j].kind == CHECK_CALL


def test_coarse_tree_onto_a_rich_one(coarse_rich):
    src, dst, sig = coarse_rich
    assert action_set_differences(src.tree, dst.tree) == []
    out, rep = translate_sigma(src, dst, BB, sig)
    m = match_nodes(src.tree, dst.tree)
    st, dt = src.tree, dst.tree
    s_index = {n: d for d, n in enumerate(src.dec_nodes.tolist())}
    assert rep["unmatched_nodes"] == {"searcher": 0, "opponent": 0}
    assert rep["lost_mass"]["searcher"]["nodes"] == rep["lost_mass"]["opponent"]["nodes"] == 0
    assert rep["translated_nodes"]["searcher"] > 0 and rep["translated_nodes"]["opponent"] > 0
    # every node whose history src has is matched to that node, and the searcher's
    # rows there are copied exactly: src's actions as they are, dst-only actions zero
    copied = 0
    for d, n in enumerate(dst.dec_nodes.tolist()):
        s = m.src_of[n]
        assert s >= 0
        if m.translated[n]:
            continue
        assert st.histories[s] == dt.histories[n]
        assert st.boards[int(st.board_id[s])] == dt.boards[int(dt.board_id[n])]
        if int(dt.actor[n]) != BB:
            continue
        s_acts = st.child_actions(s)
        for j, a in enumerate(dt.child_actions(n)):
            if a in s_acts:
                assert torch.equal(out[d, j], sig[s_index[s], s_acts.index(a)])
            else:
                assert float(out[d, j].abs().max()) == 0
        copied += 1
    assert copied > 10
    # translated rows are distributions
    legal = dst.legal[:, :, None]
    assert torch.allclose((out * legal).sum(1), torch.ones(dst.Dn, NUM_COMBOS), atol=1e-5)


@pytest.mark.parametrize(
    ("bet", "mapped"),
    [(375, 250), (500, 750)],  # 0.75 pot -> 0.5 (p = 0.64); 1.0 pot -> 1.5 (p = 0.375)
)
def test_opponent_offtree_bet_goes_to_the_right_subtree(coarse_rich, bet, mapped):
    src, dst, sig = coarse_rich
    st, dt = src.tree, dst.tree
    out, _ = translate_sigma(src, dst, BB, sig)
    m = match_nodes(st, dt)
    s_index = {n: d for d, n in enumerate(src.dec_nodes.tolist())}
    d_index = {n: d for d, n in enumerate(dst.dec_nodes.tolist())}
    # the button's bet after the BB's check is not in the coarse tree
    btn = _node(dt, ((CHECK_CALL, 0),))
    state = dt.states[btn]
    assert (RAISE, bet) in dt.child_actions(btn)
    assert (RAISE, bet) not in st.child_actions(_node(st, ((CHECK_CALL, 0),)))
    dist = map_concrete(state, COARSE, RAISE, bet)
    amounts = {o.index: o.amount for o in legal_options(state, COARSE)}
    assert amounts[max(dist, key=dist.get)] == mapped  # the more likely side
    # the BB facing it is the BB facing the mapped bet in src
    n = _node(dt, ((CHECK_CALL, 0), (RAISE, bet)))
    s = m.src_of[n]
    assert s == _node(st, ((CHECK_CALL, 0), (RAISE, mapped)))
    assert m.translated[n]
    # the searcher's row: each src action on the dst action of the same spec entry
    d, ds = d_index[n], s_index[s]
    d_acts, s_acts = dt.child_actions(n), st.child_actions(s)
    d_lab = [k.label for k in _Info(dt).kids(n)]
    s_lab = [k.label for k in _Info(st).kids(s)]
    for j, lab in enumerate(d_lab):
        if lab in s_lab:
            assert torch.equal(out[d, j], sig[ds, s_lab.index(lab)]), lab
        else:
            assert lab in (("raise", 0.75), ("raise", 1.0)), lab
            assert float(out[d, j].abs().max()) == 0
    # the BB's 0.5-pot raise over it: the button's node there is src's button facing
    # the BB's 0.5-pot raise over the mapped bet (matched by spec entry, not amount)
    j = d_lab.index(("raise", 0.5))
    i = s_lab.index(("raise", 0.5))
    n2 = int(dt.first_child[n]) + j
    assert d_acts[j] != s_acts[i]  # different amounts
    assert m.src_of[n2] == int(st.first_child[s]) + i
    # and after the call, the turn: same card, the mapped line
    call = int(dt.first_child[n]) + d_acts.index((CHECK_CALL, 0))
    turn = [c for c in dt.children[call].tolist() if c >= 0]
    assert turn and all(int(dt.kind[c]) == DECISION for c in turn)
    s_call = int(st.first_child[s]) + s_acts.index((CHECK_CALL, 0))
    for c in turn:
        sc = m.src_of[c]
        assert int(st.parent[sc]) == s_call
        assert st.boards[int(st.board_id[sc])] == dt.boards[int(dt.board_id[c])]


def test_rich_tree_onto_a_coarse_one_loses_mass():
    src, dst = _solver(RICH), _solver(COARSE)
    sig = _random_sigma(src, 2)
    with pytest.raises(ValueError, match="dst lacks"):
        translate_sigma(src, dst, BB, sig)  # the BB bets 0.75 / 1.0 at the root
    out, rep = translate_sigma(src, dst, BB, sig, strict=False)
    lost = rep["lost_mass"]
    assert lost["searcher"]["nodes"] > 0 and 0 < lost["searcher"]["max"] < 1
    assert lost["opponent"]["nodes"] > 0
    # the root: src's rows on the coarse actions, renormalised
    s_acts, d_acts = src.tree.child_actions(0), dst.tree.child_actions(0)
    d0, s0 = int(dst.dec_index[0]), int(src.dec_index[0])
    rows = torch.stack([sig[s0, s_acts.index(a)] for a in d_acts])
    assert torch.allclose(out[d0, : len(d_acts)], rows / rows.sum(0), atol=1e-6)
    legal = dst.legal[:, :, None]
    assert torch.allclose((out * legal).sum(1), torch.ones(dst.Dn, NUM_COMBOS), atol=1e-5)


def test_unmatched_nodes_play_the_fallback():
    # dst's 10x-pot bet maps to src's all-in; calling it ends the hand in src but
    # not in dst, so dst's turn below that call has no src node
    src, dst = _solver(_spec(0.5)), _solver(_spec(0.5, 10.0))
    sig = _random_sigma(src, 4)
    fallback = _random_sigma(dst, 5)
    out, rep = translate_sigma(src, dst, BB, sig, fallback=fallback)
    m = match_nodes(src.tree, dst.tree)
    unmatched = [d for d, n in enumerate(dst.dec_nodes.tolist()) if m.src_of[n] < 0]
    assert unmatched
    assert sum(rep["unmatched_nodes"].values()) == len(unmatched)
    for d in unmatched:
        assert torch.equal(out[d], fallback[d])
    bet = _node(dst.tree, ((RAISE, 5000),))  # the button facing the BB's 10x-pot bet
    s = m.src_of[bet]
    allin = int(src.tree.states[0].legal_actions().max_raise_to)
    assert src.tree.histories[s][-1][2:] == (RAISE, allin)


def test_roots_must_agree():
    a = _solver(COARSE)
    engine, cfg, s = _spot()
    tc = TreeConfig(spec=COARSE, max_nodes=10**6, chance_cards=2, leaf_mode="value_net")
    # the BB's check observed: the same street root, with the check on the path
    after = s.clone()
    after.apply(engine.Action.check_call())
    b_tree = build_tree(cfg, after.button, after.board, after.history, tc)
    m = match_nodes(a.tree, b_tree)
    dec = (b_tree.kind == DECISION).nonzero().flatten().tolist()
    assert not any(m.translated) and all(m.src_of[n] >= 0 for n in dec)
    # another board
    (_, other) = se.exploit_spots(engine, cfg, 2, 5, ["bb_first"])
    s2 = other.state
    c_tree = build_tree(cfg, s2.button, s2.board, s2.history, tc)
    with pytest.raises(ValueError, match="same public state"):
        match_nodes(a.tree, c_tree)


def test_offtree_edges_and_subtrees(coarse_rich):
    src, dst, sig = coarse_rich
    st, dt = src.tree, dst.tree
    m = match_nodes(st, dt)
    edges = offtree_edges(m, dt, BTN)
    # brute force: the button acts on the flop after a history src has exactly, with
    # an action src lacks there
    want = []
    for n in range(1, dt.num_nodes):
        p = int(dt.parent[n])
        if int(dt.kind[p]) != DECISION or int(dt.actor[p]) != BTN or int(dt.street[p]) != 1:
            continue
        s = m.src_of[p]
        if s < 0 or st.histories[s] != dt.histories[p]:
            continue
        if dt.child_actions(p)[int(dt.slot[n])] not in st.child_actions(s):
            want.append(n)
    assert edges == want and len(edges) > 2
    hist = {dt.histories[n] for n in edges}
    assert ((1, BB, CHECK_CALL, 0), (1, BTN, RAISE, 375)) in hist  # 0.75 pot after a check
    bb = offtree_edges(m, dt, BB)  # the other player's: the big blind's extra sizes
    assert ((1, BB, RAISE, 375),) in {dt.histories[n] for n in bb} and not set(bb) & set(edges)
    assert offtree_edges(match_nodes(dt, dt), dt, BTN) == []
    # no edge lies below another, and each subtree is closed under children
    for e in edges:
        sub = subtree_nodes(dt, e)
        assert sub[0] == e and not set(sub[1:]) & set(edges)
        inside = set(sub)
        assert all(int(dt.parent[x]) in inside for x in sub[1:])
        path = path_nodes(dt, e)
        assert path[0] == 0 and int(dt.parent[e]) == path[-1]
    # translating only a subtree: those rows as in the full translation, the rest fallback
    e = edges[0]
    sub = subtree_nodes(dt, e)
    full, _ = translate_sigma(src, dst, BB, sig)
    fb = _random_sigma(dst, 7)
    part, rep = translate_sigma(src, dst, BB, sig, fallback=fb, nodes=sub)
    inside = set(sub)
    n_dec = 0
    for d, n in enumerate(dst.dec_nodes.tolist()):
        assert torch.equal(part[d], full[d] if n in inside else fb[d]), n
        n_dec += n in inside
    assert sum(rep["decision_nodes"].values()) == n_dec

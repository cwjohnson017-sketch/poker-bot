import threading

import numpy as np
import pytest
from helpers import make_deck

from pokerbot.agents import AlwaysCallAgent, AlwaysRaiseAgent, BaseAgent
from pokerbot.eval import MaskedState, run_match
from pokerbot.protocol import (
    ProtocolError,
    decode_action,
    decode_gamedef,
    decode_matchstate,
    encode_action,
    encode_endhand,
    encode_gamedef,
    encode_matchstate,
    rebuild_state,
    run_client,
    serve_match,
)
from pokerbot.protocol.messages import decode_hello, encode_hello, split_response
from pokerbot.reference import Action, GameConfig, GameState, cards_from_str

pytestmark = pytest.mark.usefixtures("reference_engine")

CFG = GameConfig(num_players=2, stacks=[20000, 20000])


def test_action_roundtrip():
    for a in (Action.fold(), Action.check_call(), Action.raise_to(12345)):
        assert decode_action(encode_action(a)) == a
    assert decode_action("k") == Action.check_call()
    for bad in ("x", "r", "rabc", ""):
        with pytest.raises(ProtocolError):
            decode_action(bad)


def test_gamedef_and_hello_roundtrip():
    cfg = GameConfig(num_players=3, stacks=[100, 200, 300], small_blind=1, big_blind=2, ante=1)
    line = encode_gamedef(cfg)
    assert line == "GAMEDEF:3:1:2:1:100,200,300"
    assert decode_gamedef(line) == cfg
    assert decode_hello(encode_hello("bot:x")) == "bot_x"
    with pytest.raises(ProtocolError):
        decode_hello("HELLO")


def _play_some(state):
    for a in (Action.raise_to(300), Action.check_call(), Action.check_call(), Action.raise_to(200)):
        state.apply(a)
    return state


def test_matchstate_encoding_is_masked():
    s = _play_some(GameState.new_hand(CFG, 0, make_deck(["AsKs", "2c7d"], "QhJhTh9h8h")))
    line = encode_matchstate(MaskedState(s, 1, CFG), 1, 42)
    assert line == "MATCHSTATE:1:42:0:r300c/cr200:|2c7d/QhJhTh"
    line0 = encode_matchstate(MaskedState(s, 0, CFG), 0, 42)
    assert line0 == "MATCHSTATE:0:42:0:r300c/cr200:AsKs|/QhJhTh"


def test_matchstate_roundtrip_rebuilds_state():
    rng = np.random.default_rng(0)
    for h in range(60):
        deck = rng.permutation(52)
        s = GameState.new_hand(CFG, h % 2, deck)
        while not s.is_terminal:
            p = s.current_player
            line = encode_matchstate(MaskedState(s, p, CFG), p, h)
            ms = decode_matchstate(line)
            assert (ms.seat, ms.hand_no, ms.button) == (p, h, h % 2)
            r = rebuild_state(ms, CFG, rng=rng)
            assert r.public_key() == s.public_key()
            assert r.hole_cards(p) == s.hole_cards(p)
            assert r.current_player == p
            assert r.legal_actions() == s.legal_actions()
            la = s.legal_actions()
            choice = rng.integers(3)
            if choice == 2 and la.min_raise_to:
                s.apply(Action.raise_to(max(la.min_raise_to, min(la.max_raise_to, 400))))
            elif choice == 1 and la.can_fold and rng.random() < 0.2:
                s.apply(Action.fold())
            else:
                s.apply(Action.check_call())
        shown = [p for p, f in enumerate(s.folded) if not f]
        shown = shown if len(shown) > 1 else []
        end = encode_endhand(MaskedState(s, 0, CFG, revealed=shown), 0, h)
        ms = decode_matchstate(end)
        assert ms.ended and ms.payoffs == s.payoffs()
        r = rebuild_state(ms, CFG, rng=rng)
        assert r.is_terminal and r.public_key() == s.public_key()
        if shown:
            assert r.payoffs() == s.payoffs()


def test_bad_matchstates():
    for bad in (
        "MATCHSTATE:0:1:0:r300q:AsKs|",
        "MATCHSTATE:0:1:0:r:AsKs|",
        "MATCHSTATE:0:1",
        "HELLO",
        "MATCHSTATE:0:1:0::AsAs|",
    ):
        with pytest.raises(ProtocolError):
            ms = decode_matchstate(bad)
            rebuild_state(ms, CFG)
    ms = decode_matchstate("MATCHSTATE:0:1:0:r50:AsKs|")
    with pytest.raises(ProtocolError):
        rebuild_state(ms, CFG)  # illegal raise
    with pytest.raises(ProtocolError):
        split_response("MATCHSTATE:0:1:0::AsKs|", "MATCHSTATE:0:2:0::AsKs|:c")
    assert split_response("MATCHSTATE:0:1:0::AsKs|", "MATCHSTATE:0:1:0::AsKs|:r300") == "r300"


def _serve_with_clients(seats, clients, hands, duplicate=False, seed=0):
    """Run serve_match in a thread and the clients in threads; return (result, client stats)."""
    port_box = {}
    ready = threading.Event()
    result = {}

    def on_ready(port):
        port_box["port"] = port
        ready.set()

    def server():
        result["res"] = serve_match(
            CFG, seats, hands, seed, port=0, duplicate=duplicate, ready=on_ready, accept_timeout=20
        )

    t = threading.Thread(target=server, daemon=True)
    t.start()
    assert ready.wait(10)
    stats = [None] * len(clients)
    threads = []
    for i, agent in enumerate(clients):

        def run(i=i, agent=agent):
            stats[i] = run_client(agent, port=port_box["port"], seed=i, timeout=20)

        th = threading.Thread(target=run, daemon=True)
        th.start()
        threads.append(th)
        if seats.count(None) > 1:
            th.join(0.2)  # connect in order so seat assignment is deterministic
    for th in threads:
        th.join(60)
    t.join(60)
    assert not t.is_alive()
    return result["res"], stats


def test_remote_client_matches_local_play():
    """A deterministic agent played over the socket gets exactly the same
    results as the same match played locally."""
    local = run_match([AlwaysRaiseAgent(), AlwaysCallAgent()], CFG, 30, seed=4)
    res, stats = _serve_with_clients([AlwaysRaiseAgent(), None], [AlwaysCallAgent()], 30, seed=4)
    assert (res.seat_payoffs == local.seat_payoffs).all()
    assert stats[0].hands == 30
    assert stats[0].total == int(local.seat_payoffs[:, 1].sum())


def test_two_remote_clients_duplicate():
    res, stats = _serve_with_clients(
        [None, None], [AlwaysRaiseAgent(), AlwaysCallAgent()], 10, duplicate=True
    )
    assert res.hands == 10
    assert stats[0].hands == 10 and stats[1].hands == 10
    assert stats[0].total + stats[1].total == 0
    assert res.total_a == stats[0].total


class SpyAgent(BaseAgent):
    name = "spy"

    def __init__(self):
        super().__init__()
        self.hands = 0
        self.other_cards = []
        self.ends = []

    def new_hand(self, seat, config):
        super().new_hand(seat, config)
        self.hands += 1

    def act(self, state, seat, rng):
        self.other_cards.append(state.hole_cards(1 - seat))
        return Action.check_call()

    def observe_end(self, state):
        self.ends.append(state.payoffs()[self.seat])


def test_remote_client_is_masked_and_sees_ends():
    spy = SpyAgent()
    res, stats = _serve_with_clients([AlwaysCallAgent(), None], [spy], 8, seed=2)
    assert spy.hands == 8 and len(spy.ends) == 8
    assert spy.other_cards and all(c == [] for c in spy.other_cards)
    assert sum(spy.ends) == stats[0].total == int(res.seat_payoffs[:, 1].sum())


def test_rebuild_uses_known_cards():
    ms = decode_matchstate("MATCHSTATE:1:0:0:r300c/:|2c7d/QhJhTh")
    r = rebuild_state(ms, CFG, rng=np.random.default_rng(0))
    assert r.hole_cards(1) == cards_from_str("2c7d")
    assert r.board == cards_from_str("QhJhTh")
    assert r.street == 1 and r.current_player == 1

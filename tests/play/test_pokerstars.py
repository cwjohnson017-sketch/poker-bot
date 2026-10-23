"""PokerStars-format export and hand descriptions."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from pokerbot.engine_select import get_engine
from pokerbot.play.pokerstars import describe_hand, format_pokerstars, us_eastern
from pokerbot.reference.cards import card_from_str, cards_from_str

WHEN = datetime(2026, 10, 21, 14, 5, 9, tzinfo=UTC)
NAMES = ["You", "Bot"]


@pytest.fixture
def engine():
    return get_engine()


@pytest.fixture
def config(engine):
    return engine.GameConfig(
        num_players=2, stacks=[10000, 10000], small_blind=50, big_blind=100, ante=0
    )


def deck(*cards: str) -> list[int]:
    """Seat 0's cards, seat 1's, then the board, then the rest of the deck."""
    head = [card_from_str(c) for c in cards]
    return head + [c for c in range(52) if c not in head]


def play(engine, config, button, cards, actions):
    s = engine.GameState.new_hand(config, button, deck(*cards))
    for a in actions:
        if a == "f":
            s.apply(engine.Action.fold())
        elif a == "c":
            s.apply(engine.Action.check_call())
        else:
            s.apply(engine.Action.raise_to(int(a[1:])))
    assert s.is_terminal
    return s


def test_raise_and_fold(engine, config):
    s = play(engine, config, 0, ["Ah", "Kd", "7c", "2s"], ["r250", "f"])
    text = format_pokerstars(s, config, 123, NAMES, 0, WHEN)
    lines = text.splitlines()
    assert lines[0] == ("PokerStars Hand #123:  Hold'em No Limit (50/100) - 2026/10/21 10:05:09 ET")
    assert lines[1] == "Table 'pokerbot' 2-max (Play Money) Seat #1 is the button"
    for expected in [
        "Seat 1: You (10000 in chips)",
        "You: posts small blind 50",
        "Bot: posts big blind 100",
        "*** HOLE CARDS ***",
        "Dealt to You [Ah Kd]",
        "You: raises 150 to 250",
        "Bot: folds",
        "Uncalled bet (150) returned to You",
        "You collected 200 from pot",
        "Total pot 200 | Rake 0",
        "Seat 1: You (button) (small blind) collected (200)",
        "Seat 2: Bot (big blind) folded before Flop",
    ]:
        assert expected in lines
    assert "7c" not in text  # the folded bot's cards stay hidden


def test_checked_down_showdown(engine, config):
    cards = ["Ah", "Kd", "7c", "7s", "2c", "9d", "Jh", "3s", "4h"]
    # button 1: seat 1 posts the small blind and acts first preflop
    s = play(engine, config, 1, cards, ["c", "c"] + ["c", "c"] * 3)
    text = format_pokerstars(s, config, 7, NAMES, 0, WHEN)
    lines = text.splitlines()
    assert "Bot: calls 50" in lines
    assert "You: checks" in lines
    assert "*** FLOP *** [2c 9d Jh]" in lines
    assert "*** TURN *** [2c 9d Jh] [3s]" in lines
    assert "*** RIVER *** [2c 9d Jh 3s] [4h]" in lines
    assert "*** SHOW DOWN ***" in lines
    assert "Bot: shows [7c 7s] (a pair of Sevens)" in lines
    assert "You: shows [Ah Kd] (high card Ace)" in lines
    assert "Bot collected 200 from pot" in lines
    assert (
        "Seat 2: Bot (button) (small blind) showed [7c 7s] and won (200) with a pair of Sevens"
        in lines
    )
    assert "Seat 1: You (big blind) showed [Ah Kd] and lost with high card Ace" in lines


def test_all_in_runs_out_the_board(engine, config):
    cards = ["Ah", "Ad", "Kc", "Ks", "2c", "9d", "Jh", "3s", "4h"]
    s = play(engine, config, 0, cards, ["r10000", "c"])
    lines = format_pokerstars(s, config, 8, NAMES, 1, WHEN).splitlines()
    assert "You: raises 9900 to 10000 and is all-in" in lines
    assert "Bot: calls 9900 and is all-in" in lines
    assert "Dealt to Bot [Kc Ks]" in lines
    i = lines.index("Bot: calls 9900 and is all-in")
    assert lines[i + 1 : i + 4] == [
        "*** FLOP *** [2c 9d Jh]",
        "*** TURN *** [2c 9d Jh] [3s]",
        "*** RIVER *** [2c 9d Jh 3s] [4h]",
    ]
    assert "You collected 20000 from pot" in lines


@pytest.mark.parametrize(
    "cards, text",
    [
        ("Ah 2c 3d 4s 5h Kd Kc", "a straight, Ace to Five"),
        ("Ah Kh Qh Jh Th 2c 3d", "a Royal Flush"),
        ("9s 8s 7s 6s 5s 2c 3d", "a straight flush, Five to Nine"),
        ("Kh Kd Ks 4c 4d 2s 7h", "a full house, Kings full of Fours"),
        ("Ah Ad 4c 4d 9s 9h 2c", "two pair, Aces and Nines"),
        ("6c 6d 6h 2s 9c Jd Kh", "three of a kind, Sixes"),
        ("Ac Jc 8c 4c 2c Kd Qh", "a flush, Ace high"),
        ("2c 2d 2h 2s 9c Jd Kh", "four of a kind, Deuces"),
        ("Ac Jd 8c 4h 2s 7d 9h", "high card Ace"),
    ],
)
def test_describe_hand(cards, text):
    assert describe_hand(cards_from_str(cards)) == text


def test_us_eastern_follows_daylight_saving():
    summer = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
    winter = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
    assert us_eastern(summer) == datetime(2026, 7, 1, 8, 0)
    assert us_eastern(winter) == datetime(2026, 1, 15, 7, 0)
    # 2026: daylight time from Sunday 8 March 7:00 UTC to Sunday 1 November 6:00 UTC
    assert us_eastern(datetime(2026, 3, 8, 6, 59, tzinfo=UTC)).hour == 1
    assert us_eastern(datetime(2026, 3, 8, 7, 0, tzinfo=UTC)).hour == 3
    assert us_eastern(datetime(2026, 11, 1, 5, 59, tzinfo=UTC)).hour == 1
    assert us_eastern(datetime(2026, 11, 1, 6, 0, tzinfo=UTC)).hour == 1

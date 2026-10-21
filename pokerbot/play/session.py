"""A person against a bot, heads-up, one hand at a time: the play app's engine.

:class:`PlaySession` deals hands in a background thread. The person's actions
arrive through :meth:`PlaySession.submit_action` and the next deal through
:meth:`PlaySession.request_next` (the web server calls both). Everything a
browser may show comes from :meth:`PlaySession.snapshot`. In evaluation mode,
the only mode so far, that is what a player at the table sees: the bot's cards
only when they are shown down, and nothing of its strategy.

Deals are reproducible and never repeat. Hand ``h`` of a session with seed
``s`` uses the deck :func:`deal` ``(s, h)``, the bot's RNG
``default_rng((s, h))`` and button ``h % 2``. Every hand is noted in
``dealt.jsonl`` before it is played, so a resumed session continues after the
last hand dealt, and a hand cut short counts as abandoned instead of being
dealt again.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from types import ModuleType
from typing import Any

import numpy as np

from ..engine_select import engine_name, get_engine, to_engine_action
from ..eval.history import format_hand
from ..eval.luck import luck_adjusted
from ..eval.masking import MaskedState
from ..eval.match import showdown_seats
from ..reference.cards import card_to_str
from .log import SessionLog
from .pokerstars import describe_hand, format_pokerstars
from .stats import SessionStats

STREETS = ("preflop", "flop", "turn", "river")


class SessionClosed(Exception):
    """The session was stopped while a hand waited for the person."""


def deal(seed: int, hand: int) -> list[int]:
    """The deck of hand ``hand`` of a session with seed ``seed``."""
    return [int(c) for c in np.random.default_rng((seed, hand, 1)).permutation(52)]


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def legal_view(legal: Any) -> dict:
    return {
        "can_fold": bool(legal.can_fold),
        "can_check": bool(legal.can_check),
        "call": int(legal.call_amount),
        "min_raise_to": int(legal.min_raise_to),
        "max_raise_to": int(legal.max_raise_to),
    }


def parse_action(engine: ModuleType, legal: Any, kind: str, amount: Any = None) -> Any:
    """The person's action as an engine ``Action``; ``ValueError`` (with a message
    for the person) when it is not legal. ``kind``: ``fold``, ``check``,
    ``call``, ``raise`` (``amount`` = the street total to raise to) or ``allin``."""
    kind = str(kind).lower()
    if kind == "fold":
        if not legal.can_fold:
            raise ValueError("checking is free here: check instead of folding")
        return engine.Action.fold()
    if kind in ("check", "call"):
        if kind == "check" and not legal.can_check:
            raise ValueError("there is a bet to call")
        if kind == "call" and legal.can_check:
            raise ValueError("there is nothing to call: check")
        return engine.Action.check_call()
    if kind in ("raise", "bet", "allin"):
        lo, hi = int(legal.min_raise_to), int(legal.max_raise_to)
        if lo <= 0:
            raise ValueError("raising is not possible here")
        try:
            to = hi if kind == "allin" else int(amount)
        except (TypeError, ValueError):
            raise ValueError(f"not a number of chips: {amount!r}") from None
        if to == hi or lo <= to <= hi:
            return engine.Action.raise_to(to)
        if lo > hi:
            raise ValueError(f"only all-in is possible: raise to {hi:,}")
        raise ValueError(f"raise to between {lo:,} and {hi:,}")
    raise ValueError(f"unknown action {kind!r}")


def raise_presets(state: Any, seat: int, legal: Any, big_blind: int) -> list[dict]:
    """Preset raise-to amounts for the person: big-blind multiples preflop
    (multiples of the bet after an open), pot fractions after; then all-in."""
    lo, hi = int(legal.min_raise_to), int(legal.max_raise_to)
    if lo <= 0:
        return []
    bets = [int(b) for b in state.street_bets]
    cur = max(bets)
    to_call = cur - bets[seat]
    if int(state.street) == 0 and cur <= big_blind:
        cands = [(f"{x:g}x", round(x * big_blind)) for x in (2, 2.5, 3, 4)]
    elif int(state.street) == 0:
        cands = [(f"{x:g}x", round(x * cur)) for x in (2.5, 3, 4)]
    else:
        pot = int(state.pot)
        fracs = ((0.33, "33%"), (0.5, "50%"), (0.75, "75%"), (1.0, "pot"), (1.5, "150%"))
        cands = [(label, round(cur + f * (pot + to_call))) for f, label in fracs]
    out: list[dict] = []
    seen: set[int] = set()
    for label, to in cands:
        to = max(lo, min(hi, int(to)))
        if to in seen or to >= hi:
            continue
        seen.add(to)
        out.append({"label": label, "to": to})
    out.append({"label": "all-in", "to": hi})
    return out


def describe_action(state: Any, seat: int, action: Any) -> dict:
    """A public description of ``action`` by ``seat``, from the state before it."""
    bets = [int(b) for b in state.street_bets]
    cur, mine = max(bets), bets[seat]
    behind = int(state.stacks[seat])
    legal = state.legal_actions()
    kind = int(action.kind)
    amount = 0
    if kind == 0:
        token, text, added = "f", "folds", 0
    elif kind == 1 and legal.can_check:
        token, text, added = "k", "checks", 0
    elif kind == 1:
        added = min(cur - mine, behind)
        token, text = "c", f"calls {added:,}"
    else:
        amount = int(action.amount)
        added = amount - mine
        token = f"r{amount}"
        text = f"bets {amount:,}" if cur == 0 else f"raises to {amount:,}"
    all_in = kind != 0 and added > 0 and added == behind
    if all_in:
        text += " (all-in)"
    return {
        "street": int(state.street),
        "seat": int(seat),
        "kind": kind,
        "amount": amount,
        "token": token,
        "text": text,
        "added": int(added),
        "all_in": bool(all_in),
    }


class PlaySession:
    """One person (seat ``human_seat``) against ``bot``, logged to ``log``.

    ``hands_target`` fixes the sample: the session deals no more once that many
    hands are finished. ``warmup``, if given, runs on the session thread before
    the first deal (loading nets, compiling kernels)."""

    def __init__(
        self,
        bot: Any,
        config: Any,
        log: SessionLog,
        *,
        seed: int,
        hands_target: int | None = None,
        human_seat: int = 0,
        names: Sequence[str] = ("You", "Bot"),
        engine: ModuleType | None = None,
        luck_samples: int = 2048,
        warmup: Callable[[], Any] | None = None,
    ) -> None:
        self.engine = engine or get_engine()
        if int(config.num_players) != 2:
            raise ValueError("the play app is heads-up only")
        self.bot = bot
        self.config = config
        self.log = log
        self.seed = int(seed)
        self.hands_target = None if hands_target is None else int(hands_target)
        self.human_seat = int(human_seat)
        self.bot_seat = 1 - self.human_seat
        self.seat_names = [""] * 2
        self.seat_names[self.human_seat] = str(names[0])
        self.seat_names[self.bot_seat] = str(names[1])
        self.luck_samples = int(luck_samples)
        self.warmup = warmup
        self.stats = SessionStats(config.big_blind)
        for rec in log.hands():
            self.stats.add(rec["human_net"], rec["human_adjusted"])
        self.next_hand = len(log.dealt())
        self.abandoned = self.next_hand - self.stats.hands
        self._summary = self.stats.summary()
        self._cmd: queue.Queue = queue.Queue()
        self._cond = threading.Condition()
        self._version = 0
        self._view: dict = {}
        self._legal: Any = None
        self._next_requested = False
        self._thread: threading.Thread | None = None
        self._actions: list[dict] = []
        self._hand_no: int | None = None
        self._last_hand: dict | None = None  # the previous hand's result, for the side panel
        self._publish(status="finished" if self.done else "ready", hand=None, message=None)

    # -- any thread ----------------------------------------------------------

    @property
    def done(self) -> bool:
        return self.hands_target is not None and self.stats.hands >= self.hands_target

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="play-session", daemon=True)
            self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def snapshot(self, since: int | None = None, timeout: float = 20.0) -> dict:
        """The public view. With ``since``, wait (up to ``timeout`` s) for a
        version newer than ``since`` first: a long poll."""
        with self._cond:
            if since is not None:
                self._cond.wait_for(lambda: self._version > since, timeout=timeout)
            return self._view

    def submit_action(self, kind: str, amount: Any = None) -> tuple[bool, str | None]:
        with self._cond:
            legal = self._legal
            if legal is None:
                return False, "it is not your turn"
            try:
                action = parse_action(self.engine, legal, kind, amount)
            except ValueError as e:
                return False, str(e)
            self._legal = None  # one action per prompt
        self._cmd.put(("action", action))
        return True, None

    def request_next(self) -> tuple[bool, str | None]:
        with self._cond:
            status = self._view.get("status")
            if self.done:
                return False, "the session's sample is complete"
            if status not in ("ready", "hand_over"):
                return False, f"cannot deal now ({status})"
            if self._next_requested:
                return True, None
            self._next_requested = True
        self._cmd.put(("next",))
        return True, None

    def stop(self) -> None:
        """End the session. A hand in progress is abandoned."""
        self._cmd.put(("stop",))

    # -- the session thread ---------------------------------------------------

    def _session_info(self) -> dict:
        return {
            "id": self.log.meta.get("session"),
            "mode": self.log.meta.get("mode", "evaluation"),
            "hands_target": self.hands_target,
            "hands_played": self.stats.hands,
            "abandoned": self.abandoned,
            "human_seat": self.human_seat,
            "names": {
                "human": self.seat_names[self.human_seat],
                "bot": self.seat_names[self.bot_seat],
            },
        }

    def _publish(self, **changes: Any) -> None:
        with self._cond:
            view = dict(self._view)
            view.update(changes)
            self._version += 1
            view["version"] = self._version
            view["session"] = self._session_info()
            view["stats"] = self._summary
            view["last_hand"] = self._last_hand
            self._view = view
            self._cond.notify_all()

    def _run(self) -> None:
        try:
            if self.warmup is not None:
                self._publish(status="warmup", message=None)
                self.warmup()
                self._publish(status="finished" if self.done else "ready", message=None)
            while not self.done:
                cmd = self._cmd.get()
                if cmd[0] == "stop":
                    break
                if cmd[0] == "next":
                    self._play_hand(self.next_hand)
            self._publish(status="finished", message=None)
        except SessionClosed:
            self.abandoned += 1
            self._publish(status="finished", message="Session ended mid-hand: hand abandoned")
        except Exception as e:  # shown in the browser and kept in the session directory
            tb = traceback.format_exc()
            with open(self.log.dir / "errors.log", "a", encoding="utf-8") as fh:
                fh.write(f"{now_iso()}\n{tb}\n")
            self._publish(status="error", message=f"{type(e).__name__}: {e}")

    def _await_human(self, state: Any) -> tuple[Any, float]:
        legal = state.legal_actions()
        with self._cond:
            self._legal = legal
        self._publish(status="human", hand=self._hand_view(state, legal=legal), message=None)
        t0 = time.perf_counter()
        while True:
            cmd = self._cmd.get()
            if cmd[0] == "stop":
                raise SessionClosed
            if cmd[0] != "action":
                continue
            try:
                state.child(cmd[1])  # the engine has the last word on legality
            except ValueError as e:
                with self._cond:
                    self._legal = legal
                self._publish(message=f"Illegal action: {e}")
                continue
            return cmd[1], time.perf_counter() - t0

    def _play_hand(self, h: int) -> None:
        cfg, eng = self.config, self.engine
        hs, bs = self.human_seat, self.bot_seat
        deck = deal(self.seed, h)
        button = h % 2
        started = now_iso()
        self.log.note_dealt(h, deck, button, started)
        self.next_hand = h + 1
        with self._cond:
            self._next_requested = False
        state = eng.GameState.new_hand(cfg, button, deck)
        rng = np.random.default_rng((self.seed, h))
        redeal = np.random.default_rng((self.seed, h, 2))
        self.bot.new_hand(bs, cfg)
        self._hand_no = h
        self._actions = self._blind_entries(state)
        contrib = [int(s) - int(b) for s, b in zip(cfg.stacks, state.stacks, strict=True)]
        logged: list[dict] = []
        while not state.is_terminal:
            p = int(state.current_player)
            if p == hs:
                action, seconds = self._await_human(state)
            else:
                self._publish(status="bot", hand=self._hand_view(state), message=None)
                t0 = time.perf_counter()
                raw = self.bot.act(MaskedState(state, p, cfg, redeal), p, rng)
                seconds = time.perf_counter() - t0
                action = to_engine_action(eng, raw)
            entry = describe_action(state, p, action)
            state.apply(action)
            contrib[p] += entry["added"]
            entry["seconds"] = round(seconds, 4)
            self._actions = [*self._actions, {**entry, "street": STREETS[entry["street"]]}]
            logged.append(
                {k: entry[k] for k in ("street", "seat", "kind", "amount", "token", "seconds")}
            )
        shown = showdown_seats(state)
        self.bot.observe_end(MaskedState(state, bs, cfg, redeal, revealed=shown))
        payoffs = [int(x) for x in state.payoffs()]
        adjusted = luck_adjusted([(button, deck, state)], cfg, eng, self.luck_samples, seed=h)
        record = {
            "hand": h,
            "started": started,
            "finished": now_iso(),
            "button": button,
            "human_seat": hs,
            "deck": deck,
            "hole_cards": [[int(c) for c in state.hole_cards(p)] for p in (0, 1)],
            "board": [int(c) for c in state.board],
            "actions": logged,
            "showdown": [int(p) for p in shown],
            "payoffs": payoffs,
            "human_net": payoffs[hs],
            "human_adjusted": float(adjusted[0, hs]),
        }
        decisions = list(getattr(self.bot, "decisions", None) or [])
        text = format_hand(state, cfg, h, self.seat_names, engine_name(eng))
        stars = format_pokerstars(
            state, cfg, self._hand_id(h), self.seat_names, hs, datetime.now(UTC)
        )
        self.log.write_hand(record, decisions, text, stars)
        self.stats.add(record["human_net"], record["human_adjusted"])
        self._summary = self.stats.summary()
        result = self._result_view(state, payoffs, shown, 2 * min(contrib))
        self._last_hand = {
            "number": h,
            "text": result["text"],
            "human_net": result["human_net"],
            "bot_cards": [card_to_str(c) for c in state.hole_cards(bs)] if bs in shown else None,
        }
        self._publish(status="hand_over", hand=self._hand_view(state, result=result))

    # -- views ---------------------------------------------------------------

    def _hand_id(self, h: int) -> int:
        digits = "".join(ch for ch in str(self.log.meta.get("session", "")) if ch.isdigit())
        return int((digits or "0") + f"{h:05d}")

    def _blind_entries(self, state: Any) -> list[dict]:
        start = [int(s) for s in self.config.stacks]
        out = []
        for seat, label in (
            (int(state.button), "small blind"),
            (1 - int(state.button), "big blind"),
        ):
            posted = start[seat] - int(state.stacks[seat])
            out.append(
                {
                    "street": "preflop",
                    "seat": seat,
                    "token": "",
                    "text": f"posts {label} {posted:,}",
                }
            )
        return out

    def _hand_view(self, state: Any, legal: Any = None, result: dict | None = None) -> dict:
        hs = self.human_seat
        terminal = bool(state.is_terminal)
        shown = showdown_seats(state) if terminal else []
        bets = [0, 0] if terminal else [int(b) for b in state.street_bets]
        if terminal:  # the engine's terminal stacks are before the pot is paid out
            stacks = [
                int(s) + int(p) for s, p in zip(self.config.stacks, state.payoffs(), strict=True)
            ]
        else:
            stacks = [int(s) for s in state.stacks]
        players = []
        for seat in (0, 1):
            visible = seat == hs or seat in shown
            players.append(
                {
                    "seat": seat,
                    "name": self.seat_names[seat],
                    "human": seat == hs,
                    "stack": stacks[seat],
                    "bet": bets[seat],
                    "folded": bool(state.folded[seat]),
                    "all_in": bool(state.all_in[seat]),
                    "button": seat == int(state.button),
                    "cards": [card_to_str(c) for c in state.hole_cards(seat)] if visible else None,
                }
            )
        view = {
            "number": self._hand_no,
            "street": STREETS[int(state.street)],
            "board": [card_to_str(c) for c in state.board],
            "pot": int(state.pot) if not terminal else int(result["pot"]) if result else 0,
            "pot_behind": int(state.pot) - sum(bets) if not terminal else 0,
            "players": players,
            "actions": self._actions,
            "to_act": -1 if terminal else int(state.current_player),
            "legal": None,
            "result": result,
        }
        if legal is not None:
            view["legal"] = {
                **legal_view(legal),
                "presets": raise_presets(state, hs, legal, int(self.config.big_blind)),
                "pot": int(state.pot),
            }
        return view

    def _result_view(self, state: Any, payoffs: list[int], shown: list[int], pot: int) -> dict:
        """``pot``: the chips both players matched (an uncalled bet goes back)."""
        hs, bs = self.human_seat, self.bot_seat
        board = [int(c) for c in state.board]
        hands = {}
        for p in shown:
            hands[self.seat_names[p]] = describe_hand([int(c) for c in state.hole_cards(p)] + board)
        net = payoffs[hs]
        if net > 0:
            text = f"{self.seat_names[hs]} won {pot:,}"
        elif net < 0:
            text = f"{self.seat_names[bs]} won {pot:,}"
        else:
            text = f"Split pot of {pot:,}"
        if shown:
            winner = hs if net > 0 else bs if net < 0 else None
            if winner is not None:
                text += f" with {hands[self.seat_names[winner]]}"
        return {"human_net": net, "showdown": bool(shown), "hands": hands, "pot": pot, "text": text}

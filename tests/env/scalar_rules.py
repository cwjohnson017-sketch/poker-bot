"""Independent, straightforward scalar heads-up NLHE rules (test oracle only).

Written separately from the tensor code to catch vectorization bugs; it
follows the contract in docs/INTERFACES.md with the same rule reading as
``pokerbot.env.vec_env`` (see that module's docstring).
"""

from __future__ import annotations

from .naive_eval import naive7

FOLD, CHECK_CALL, RAISE = 0, 1, 2


class ScalarHand:
    def __init__(self, stacks, sb, bb, ante, button, deck):
        self.bb = bb
        self.button = button
        self.deck = list(deck)
        self.start = list(stacks)
        self.stacks = list(stacks)
        self.bets = [0, 0]
        self.contrib = [0, 0]
        self.folded = [False, False]
        self.all_in = [False, False]
        self.acted = [False, False]
        self.street = 0
        self.terminal = False
        self.pay = None
        for p in (0, 1):
            a = min(ante, self.stacks[p])
            self.stacks[p] -= a
            self.contrib[p] += a
        for p, blind in ((button, sb), (1 - button, bb)):
            b = min(blind, self.stacks[p])
            self.stacks[p] -= b
            self.bets[p] += b
            self.contrib[p] += b
            if self.stacks[p] == 0:
                self.all_in[p] = True
        self.last_raise = bb
        self.player = button
        if self.all_in[button]:
            self.player = 1 - button
        if all(self.all_in):
            self._showdown()

    def board(self):
        n = {0: 0, 1: 3, 2: 4, 3: 5}[self.street]
        return self.deck[4 : 4 + n]

    def legal(self):
        p = self.player
        o = 1 - p
        to_call = max(self.bets) - self.bets[p]
        raise_ok = self.stacks[p] > to_call and self.stacks[o] > 0
        return {
            "current_player": p,
            "street": self.street,
            "pot": sum(self.contrib),
            "can_fold": to_call > 0,
            "can_check": to_call == 0,
            "call_amount": min(to_call, self.stacks[p]),
            "min_raise_to": max(self.bets) + max(self.last_raise, self.bb) if raise_ok else 0,
            "max_raise_to": self.bets[p] + self.stacks[p] if raise_ok else 0,
        }

    def _put(self, p, chips):
        self.stacks[p] -= chips
        self.bets[p] += chips
        self.contrib[p] += chips
        if self.stacks[p] == 0:
            self.all_in[p] = True

    def apply(self, kind, amount=0):
        assert not self.terminal
        lg = self.legal()
        p = self.player
        if kind == FOLD:
            if not lg["can_fold"]:
                raise ValueError("fold not legal")
            self.folded[p] = True
            self.terminal = True
            c = self.contrib[p]
            self.pay = [c, c]
            self.pay[p] = -c
            return
        if kind == CHECK_CALL:
            self._put(p, lg["call_amount"])
            self.acted[p] = True
        elif kind == RAISE:
            lo, hi = lg["min_raise_to"], lg["max_raise_to"]
            if lo == 0 or not (lo <= amount <= hi or amount == hi):
                raise ValueError("bad raise")
            prev_max = max(self.bets)
            if amount - prev_max >= max(self.last_raise, self.bb):
                self.last_raise = amount - prev_max
            self._put(p, amount - self.bets[p])
            self.acted[p] = True
            self.acted[1 - p] = False
        else:
            raise ValueError("bad kind")
        top = max(self.bets)
        settled = all(
            self.folded[i] or self.all_in[i] or (self.acted[i] and self.bets[i] == top)
            for i in (0, 1)
        )
        if not settled:
            self.player = 1 - p
            return
        can_bet = sum(1 for i in (0, 1) if not self.folded[i] and not self.all_in[i])
        if can_bet <= 1 or self.street == 3:
            self._showdown()
            return
        self.street += 1
        self.bets = [0, 0]
        self.acted = [False, False]
        self.last_raise = 0
        self.player = 1 - self.button

    def _showdown(self):
        self.street = 3
        self.terminal = True
        board = self.deck[4:9]
        r0 = naive7(self.deck[0:2] + board)
        r1 = naive7(self.deck[2:4] + board)
        stake = min(self.contrib)
        w = 0 if r0 == r1 else (stake if r0 > r1 else -stake)
        self.pay = [w, -w]

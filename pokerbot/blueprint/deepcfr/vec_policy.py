"""The SD-CFR average policy acting directly on a :class:`~pokerbot.env.VecNLHE` batch.

:class:`NeuralVecPolicy` implements the ``VecPolicy`` protocol of
:mod:`pokerbot.eval.abr` (``act(env, mask) -> LongTensor[n]``): one
``env.obs()`` call, one forward pass of every net per seat for the slots
that act, and the reach-weighted SD-CFR average per slot.

Own reach is tracked per slot and seat: after each action the policy
multiplies every net's reach of that slot by the net's probability of the
action it returned (the caller must apply the returned actions, as the ABR
loop does). A slot's reach is reset when it holds a new hand, detected by a
change of its deck (``env.deck``) or of the batch size.

The env must use the blueprint's action spec (the network reads the env's
history tokens and legal mask, which are defined by it).
"""

from __future__ import annotations

import copy
from typing import Any

import torch

from ...env.vec_env import VecNLHE
from .features import features_from_obs, index_features
from .policy import SDCFRPolicy
from .strength import add_strength, load_strength


def _on_device(pol: SDCFRPolicy, device: torch.device) -> SDCFRPolicy:
    if pol.device == device:
        return pol
    nets = [(t, copy.deepcopy(net)) for t, net in zip(pol.iterations, pol.nets, strict=True)]
    preflop = None
    if pol.preflop_tree is not None and pol.preflop_tables is not None:
        tables = dict(zip(pol.iterations, pol.preflop_tables, strict=True))
        preflop = (pol.preflop_tree, tables)
    return SDCFRPolicy(
        nets,
        reach_weighted=pol.reach_weighted,
        fallback=pol.fallback,
        device=device,
        preflop=preflop,
    )


class NeuralVecPolicy:
    def __init__(
        self,
        policies: list[SDCFRPolicy],
        spec: Any,
        features: Any,
        device: torch.device | str = "cpu",
        seed: int = 0,
        greedy: bool = False,
        name: str = "neural",
    ) -> None:
        self.device = torch.device(device)
        self.policies = [_on_device(p, self.device) for p in policies]
        self.spec = spec
        self.features = features
        self.greedy = greedy
        self.name = name
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self._obs_gen = torch.Generator(device=self.device).manual_seed(int(seed) + 1)
        self._log_reach: list[torch.Tensor] | None = None
        self._sig: torch.Tensor | None = None
        path = getattr(features, "strength_tables", None)
        self.strength = load_strength(path) if path else None

    def reseed(self, seed: int) -> None:
        self.generator.manual_seed(int(seed))
        self._obs_gen.manual_seed(int(seed) + 1)

    def _sync(self, env: VecNLHE) -> None:
        sig = env.deck.to(self.device)
        if self._log_reach is None or self._sig is None or self._sig.shape != sig.shape:
            self._log_reach = [
                torch.zeros(len(p), env.n, dtype=torch.float64, device=self.device)
                for p in self.policies
            ]
        else:
            new = (sig != self._sig).any(1)
            for lr in self._log_reach:
                lr[:, new] = 0.0
        self._sig = sig.clone()

    @torch.no_grad()
    def act(self, env: VecNLHE, mask: torch.Tensor | None = None) -> torch.Tensor:
        if env.spec != self.spec:
            raise ValueError(
                "NeuralVecPolicy: the env's action spec differs from the blueprint's; "
                "build the VecNLHE with spec=agent.spec"
            )
        out = env.tab.call_index[env.street.clamp(0, 3)].clone().to(self.device)
        live = ~env.done if mask is None else (mask.to(env.device) & ~env.done)
        self._sync(env)
        if not bool(live.any()):
            return out
        obs = env.obs(**self.features.obs_kwargs(), generator=self._obs_gen)
        feats = features_from_obs(obs)
        for seat, pol in enumerate(self.policies):
            idx = (live & (env.actor == seat)).nonzero().squeeze(1)
            if idx.numel() == 0:
                continue
            f = index_features(feats, idx)
            if self.strength is not None:
                f = add_strength(f, self.strength)
            f = {k: v.to(self.device) for k, v in f.items()}
            lr = self._log_reach[seat][:, idx.to(self.device)] if pol.reach_weighted else None
            avg, P = pol.average(f, lr)
            avg = avg.float() * f["legal"].float()
            if self.greedy:
                a = avg.argmax(1)
            else:
                w = torch.where(avg.sum(1, keepdim=True) > 0, avg, f["legal"].float())
                a = torch.multinomial(w, 1, generator=self.generator).squeeze(1)
            dev_idx = idx.to(self.device)
            out[dev_idx] = a
            if pol.reach_weighted:
                p_a = P.gather(2, a[None, :, None].expand(P.shape[0], -1, 1)).squeeze(2)
                self._log_reach[seat][:, dev_idx] += torch.log(p_a.double().clamp(min=0))
        return out.to(env.device)

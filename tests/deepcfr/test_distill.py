"""Distilling an SD-CFR average into one softmax policy net per seat."""

import torch

from pokerbot.agents.registry import make_agent
from pokerbot.blueprint.deepcfr.distill import DistillConfig, distill


def test_distill_round_trip(tiny_neural_run, tmp_path):
    threads = torch.get_num_threads()
    try:
        cfg = DistillConfig(rows=3000, n_envs=64, steps=60, batch=256, holdout=0.1)
        report = distill(tiny_neural_run, tmp_path, cfg, device="cpu", log=None)
    finally:
        torch.set_num_threads(threads)
    assert set(report) == {"seat0", "seat1"}
    assert all(r["rows"] > 1000 and 0.0 <= r["tv_mean"] <= 1.0 for r in report.values())
    agent = make_agent(f"neural:{tmp_path}")
    for pol in agent.policies:
        assert len(pol) == 1 and pol.policy_head == "softmax" and not pol.reach_weighted

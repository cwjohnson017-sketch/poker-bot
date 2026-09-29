import json

import torch

from pokerbot.agents import UniformPolicyAgent
from pokerbot.env import GameConfig, VecNLHE
from pokerbot.eval.abr import (
    ABRConfig,
    CallVecPolicy,
    ScalarVecPolicy,
    UniformRandomVecPolicy,
    evaluate_br,
    main,
    make_vec_policy,
    train_abr,
)
from pokerbot.eval.logging import read_scalars

SHORT = GameConfig(stacks=[2000, 2000])  # 20bb: lower variance, faster hands


def test_vec_policies_return_legal_actions():
    env = VecNLHE(32, SHORT, "cpu", seed=1)
    for pol in (
        UniformRandomVecPolicy("cpu", 0),
        CallVecPolicy(),
        ScalarVecPolicy(UniformPolicyAgent(), seed=0),
    ):
        env.reset()
        for _ in range(30):
            a = pol.act(env, ~env.done)
            legal = env.legal_mask()
            assert bool(legal.gather(1, a[:, None]).all())
            _, done = env.step(a)  # validates legality
            env.reset(done)
    assert isinstance(make_vec_policy("uniform"), UniformRandomVecPolicy)
    assert isinstance(make_vec_policy("call"), CallVecPolicy)
    assert isinstance(make_vec_policy("fixed:check_call"), ScalarVecPolicy)


def test_evaluate_is_reproducible_with_common_random_numbers():
    def always_call(obs):
        return torch.ones(obs["cards"].shape[0], dtype=torch.long)

    r1 = evaluate_br(always_call, UniformRandomVecPolicy(), SHORT, hands=256, n_envs=64, seed=5)
    r2 = evaluate_br(always_call, UniformRandomVecPolicy(), SHORT, hands=256, n_envs=64, seed=5)
    assert r1.stats.mbb_per_hand == r2.stats.mbb_per_hand
    assert r1.hands == 256


def test_abr_learns_to_beat_uniform_random(tmp_path):
    cfg = ABRConfig(
        n_envs=128,
        train_steps=300,
        learning_starts=500,
        eps_decay_steps=200,
        equity_samples=16,
        eval_hands=4096,
        eval_envs=2048,
        log_every=100,
        seed=1,
    )
    from pokerbot.eval.logging import RunLogger

    with RunLogger(tmp_path / "log", tensorboard=False) as log:
        _, res = train_abr(cfg, UniformRandomVecPolicy("cpu", 0), SHORT, "cpu", log)
    init, final = res.initial.stats, res.final.stats
    # loose margins; the evaluation CI half-width here is about 450 mbb/h
    assert final.mbb_per_hand > init.mbb_per_hand + 1000, (init, final)
    assert final.ci_low > 0, final
    assert len(res.curve) == 3
    assert "abr/train_mbb" in read_scalars(tmp_path / "log")


def test_abr_cli_tiny(tmp_path, capsys):
    cfg = tmp_path / "abr.yaml"
    cfg.write_text(
        "device: cpu\nopponent: call\ngame: {stacks: [2000, 2000]}\n"
        "abr: {n_envs: 16, train_steps: 20, learning_starts: 32, batch_size: 32, "
        "eval_hands: 64, eval_envs: 32, log_every: 10, equity_samples: 0}\n"
    )
    out = tmp_path / "abr.json"
    ckpt = tmp_path / "q.pt"
    assert main(["--config", str(cfg), "--out", str(out), "--checkpoint", str(ckpt)]) == 0
    data = json.loads(out.read_text())
    assert data["opponent"] == "call" and data["final"]["hands"] == 64
    assert "state_dict" in torch.load(ckpt, weights_only=False)
    assert "ABR vs call" in capsys.readouterr().out

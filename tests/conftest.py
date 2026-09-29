import pytest


@pytest.fixture
def reference_engine(monkeypatch):
    """Force the pure-Python engine so tests do not depend on the Rust build."""
    monkeypatch.setenv("POKERBOT_ENGINE", "reference")


def _have_solver() -> bool:
    try:
        import poker_engine as pe
    except ImportError:  # pragma: no cover - depends on the build
        return False
    return hasattr(pe, "Trainer") and hasattr(pe, "BlueprintStrategy")


@pytest.fixture(scope="session")
def small_strategy(tmp_path_factory):
    """``configs/mccfr_small.yaml`` (100bb, 20 postflop buckets) trained for
    20k single-thread iterations (a couple of seconds) and exported. Skips
    when the ``poker_engine`` build has no MCCFR solver."""
    if not _have_solver():
        pytest.skip("poker_engine without the MCCFR solver")
    import poker_engine as pe

    from pokerbot.blueprint.mccfr.train import solver_config
    from pokerbot.config import REPO_ROOT, load_yaml

    sc = solver_config(load_yaml(REPO_ROOT / "configs" / "mccfr_small.yaml"))
    sc.pop("checkpoint_path", None)
    sc.pop("checkpoint_interval", None)
    trainer = pe.Trainer(sc)
    trainer.run(20_000, 1)
    path = tmp_path_factory.mktemp("mccfr_small") / "strategy.bin"
    trainer.export_strategy(str(path))
    return path


@pytest.fixture(scope="session")
def tiny_neural_run(tmp_path_factory):
    """``configs/deepcfr_tiny.yaml`` (20bb, 4 actions per street) trained for
    two iterations (two nets per seat, so the SD-CFR reach weights matter)."""
    import torch

    from pokerbot.blueprint.deepcfr.config import DeepCFRConfig
    from pokerbot.blueprint.deepcfr.trainer import DeepCFRTrainer
    from pokerbot.config import REPO_ROOT

    threads = torch.get_num_threads()
    cfg = DeepCFRConfig.load(REPO_ROOT / "configs" / "deepcfr_tiny.yaml")
    cfg.logging.print = False
    cfg.logging.tensorboard = False
    cfg.eval.every = 0
    run = tmp_path_factory.mktemp("deepcfr_tiny")
    try:
        tr = DeepCFRTrainer(cfg, run)
        tr.run_iteration(1)
        tr.run_iteration(2)
        tr.close()
    finally:
        torch.set_num_threads(threads)
    return run

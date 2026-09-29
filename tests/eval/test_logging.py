import json

from pokerbot.eval.logging import RunLogger, read_scalars


def test_logger_writes_csv_and_json(tmp_path):
    d = tmp_path / "run"
    with RunLogger(d, config={"lr": 0.1}) as log:
        log.scalar("train/loss", 1.5, 1)
        log.scalars({"train/loss": 1.0, "eval/mbb": 250.0}, 2)
        log.text("note", "hello", 2)
        p = log.write_json("result.json", {"mbb": 1.25, "arr": [1, 2]})
    assert (d / "scalars.csv").exists() and (d / "run.json").exists()
    assert json.loads(p.read_text()) == {"mbb": 1.25, "arr": [1, 2]}
    assert json.loads((d / "run.json").read_text())["config"] == {"lr": 0.1}
    s = read_scalars(d)
    assert s["train/loss"] == [(1, 1.5), (2, 1.0)]
    assert s["eval/mbb"] == [(2, 250.0)]
    # appending across restarts keeps a single header
    with RunLogger(d, tensorboard=False) as log:
        assert not log.tensorboard_enabled
        log.scalar("train/loss", 0.5, 3)
    assert read_scalars(d)["train/loss"][-1] == (3, 0.5)
    assert (d / "scalars.csv").read_text().count("wall_time") == 1


def test_null_logger_is_noop():
    log = RunLogger(None)
    log.scalar("x", 1.0, 0)
    assert log.write_json("x.json", {}) is None
    log.close()

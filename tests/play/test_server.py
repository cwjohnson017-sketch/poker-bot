"""PlayServer: the JSON API over HTTP, static files and the host check."""

from __future__ import annotations

import http.client
import json

import pytest

from pokerbot.agents import AlwaysCallAgent
from pokerbot.engine_select import get_engine
from pokerbot.play import PlayServer, PlaySession

from .helpers import new_log


@pytest.fixture
def served(tmp_path):
    engine = get_engine()
    config = engine.GameConfig(
        num_players=2, stacks=[1000, 1000], small_blind=50, big_blind=100, ante=0
    )
    session = PlaySession(
        AlwaysCallAgent(), config, new_log(tmp_path), seed=0, hands_target=2, engine=engine
    )
    server = PlayServer(session, "127.0.0.1", 0)
    server.start()
    session.start()
    yield server, session
    session.stop()
    server.shutdown()
    session.join(5)


def request(server, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=30)
    hdrs = {"Host": f"127.0.0.1:{server.port}"}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        hdrs["Content-Type"] = "application/json"
    hdrs.update(headers or {})
    conn.request(method, path, body=data, headers=hdrs)
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    ctype = resp.getheader("Content-Type") or ""
    return resp.status, (json.loads(raw) if "json" in ctype else raw.decode())


def wait_for(server, version, statuses):
    while True:
        code, v = request(server, "GET", f"/api/state?since={version}")
        assert code == 200
        version = v["version"]
        if v["status"] in statuses:
            return v


def test_a_hand_over_http(served):
    server, _ = served
    code, v = request(server, "GET", "/api/state")
    assert code == 200 and v["status"] == "ready" and v["hand"] is None
    code, r = request(server, "POST", "/api/next", {})
    assert (code, r) == (200, {"ok": True, "error": None})
    v = wait_for(server, v["version"], ("human",))
    assert v["hand"]["legal"]["presets"][-1]["label"] == "all-in"
    bot = next(p for p in v["hand"]["players"] if not p["human"])
    assert bot["cards"] is None
    code, r = request(server, "POST", "/api/action", {"kind": "raise", "amount": 5})
    assert code == 409 and not r["ok"]
    while v["status"] == "human":
        legal = v["hand"]["legal"]
        kind = "check" if legal["can_check"] else "call"
        code, r = request(server, "POST", "/api/action", {"kind": kind})
        assert (code, r["ok"]) == (200, True)
        v = wait_for(server, v["version"], ("human", "hand_over"))
    assert v["status"] == "hand_over"
    result = v["hand"]["result"]
    assert result["showdown"]  # a calling station against a calling station
    bot = next(p for p in v["hand"]["players"] if not p["human"])
    assert len(bot["cards"]) == 2
    assert v["stats"]["hands"] == 1


def test_static_files_and_404s(served):
    server, _ = served
    code, page = request(server, "GET", "/")
    assert code == 200 and "pokerbot" in page and "/static/app.js" in page
    assert request(server, "GET", "/static/app.js")[0] == 200
    assert request(server, "GET", "/static/style.css")[0] == 200
    assert request(server, "GET", "/static/../session.py")[0] == 404
    assert request(server, "GET", "/nope")[0] == 404


def test_foreign_hosts_and_non_json_posts_are_refused(served):
    server, _ = served
    assert request(server, "GET", "/api/state", headers={"Host": "evil.example:80"})[0] == 403
    assert request(server, "POST", "/api/next", {}, headers={"Host": "evil.example"})[0] == 403
    code, r = request(server, "POST", "/api/next", headers={"Content-Type": "text/plain"})
    assert code == 415
    code, r = request(server, "POST", "/api/next", [1, 2])
    assert code == 400
    assert request(server, "GET", "/api/state", headers={"Host": "localhost"})[0] == 200

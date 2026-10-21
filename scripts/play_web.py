#!/usr/bin/env python
"""Play the bot in your browser: a local web table, every hand logged.

    python scripts/play_web.py                      # the production bot, a new session
    python scripts/play_web.py --hands 300          # a sample fixed at 300 hands
    python scripts/play_web.py --resume runs/play/20261021-140509
    python scripts/play_web.py --bot equity         # a cheap opponent, to try the app

The bot is ``match.a`` of ``--config`` (configs/match_search_vn.yaml: the
production search bot on the GPU), built with that file's ``agents:`` settings.
Evaluation mode shows only what a player at the table sees. Everything goes to
``runs/play/<session>/`` (pokerbot/play/log.py). Ctrl+C stops the server; a
hand in progress is then abandoned.
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
import time
import webbrowser
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokerbot.agents import AlwaysCallAgent, make_agent  # noqa: E402
from pokerbot.config import REPO_ROOT, game_config, git_hash, load_yaml  # noqa: E402
from pokerbot.engine_select import engine_name, get_engine  # noqa: E402
from pokerbot.eval.match import play_hand  # noqa: E402
from pokerbot.play import PlayServer, PlaySession, SessionLog  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default="configs/match_search_vn.yaml", help="match config")
    p.add_argument("--bot", default=None, help="agent spec (default: the config's match.a)")
    p.add_argument("--hands", type=int, default=None, help="fix the sample at this many hands")
    p.add_argument("--seed", type=int, default=None, help="deal seed (default: random, logged)")
    p.add_argument("--out", default="runs/play", help="directory for session directories")
    p.add_argument("--resume", default=None, help="a session directory to continue")
    p.add_argument("--name", default="You", help="your name at the table")
    p.add_argument("--bot-name", default="Bot", help="the bot's name at the table")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765, help="0: any free port")
    p.add_argument("--no-browser", action="store_true", help="do not open a browser")
    p.add_argument("--no-warmup", action="store_true", help="skip the warm-up hand")
    return p


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_meta(args: argparse.Namespace, engine: Any) -> dict:
    cfg = load_yaml(args.config)
    spec = args.bot or (cfg.get("match") or {}).get("a")
    if not spec:
        raise SystemExit(f"{args.config} has no match.a; pass --bot")
    params = dict((cfg.get("agents") or {}).get(spec.partition(":")[0]) or {})
    search_file = None
    if spec.startswith("search:") and isinstance(params.get("config"), str):
        search_file = params["config"]
        params["config"] = load_yaml(search_file)  # a resumed session keeps these settings
    game = game_config(cfg.get("game"), engine)
    session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    return {
        "session": session_id,
        "created": now_iso(),
        "mode": "evaluation",
        "seed": int(args.seed) if args.seed is not None else secrets.randbits(62),
        "hands_target": args.hands,
        "human_seat": 0,
        "names": [args.name, args.bot_name],
        "bot": spec,
        "bot_params": params,
        "config_file": str(args.config),
        "search_config_file": search_file,
        "game": {
            "num_players": int(game.num_players),
            "stacks": [int(s) for s in game.stacks],
            "small_blind": int(game.small_blind),
            "big_blind": int(game.big_blind),
            "ante": int(game.ante),
        },
        "git": git_hash(),
        "engine": engine_name(engine),
        "resumes": [],
    }


def make_warmup(bot: Any, config: Any, engine: Any):
    """One unlogged hand against a calling station with the bot holding aces, so
    its first real decisions do not pay for loading nets and compiling kernels."""

    def warm() -> None:
        record = getattr(bot, "record", None)
        if record is not None:
            bot.record = False
        aces = [51, 50]
        rest = [c for c in np.random.default_rng(7).permutation(52).tolist() if c not in aces]
        play_hand(
            config, [bot, AlwaysCallAgent()], 0, aces + rest, np.random.default_rng(7), engine
        )
        if record is not None:
            bot.record = record

    return warm


def summary(session: PlaySession, log: SessionLog) -> str:
    s = session.stats.summary()
    lines = [f"Session {log.meta['session']}: {s['hands']} hands, {session.abandoned} abandoned"]
    if s["hands"]:
        raw, adj = s["raw"], s["adjusted"]
        lines.append(f"  {log.meta['names'][0]}: {s['net']:+,.0f} chips ({s['net_bb']:+,.1f} bb)")
        if s["hands"] < 30:  # a bootstrap CI over a handful of hands means little
            lines.append(
                f"  bot win rate: {-raw['mbb']:+.0f} mbb/hand, luck-adjusted "
                f"{-adj['mbb']:+.0f} (CIs after 30 hands)"
            )
        else:
            lines.append(
                f"  bot win rate: {-raw['mbb']:+.0f} mbb/hand, 95% CI [{-raw['hi']:+.0f}, "
                f"{-raw['lo']:+.0f}]; luck-adjusted {-adj['mbb']:+.0f} [{-adj['hi']:+.0f}, "
                f"{-adj['lo']:+.0f}]"
            )
    lines.append(f"  logs: {log.dir}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # paths given here are relative to where the command runs; paths inside the
    # configs are relative to the repository
    args.config = str(Path(args.config).resolve())
    out = Path(args.out).resolve()
    resume = Path(args.resume).resolve() if args.resume else None
    os.chdir(REPO_ROOT)
    engine = get_engine()

    if resume:
        log = SessionLog(resume)
        meta = log.meta
        if args.bot and args.bot != meta["bot"]:
            raise SystemExit(f"this session plays {meta['bot']!r}; --bot cannot change it")
        if args.hands is not None and args.hands != meta.get("hands_target"):
            print(f"note: the sample stays at {meta.get('hands_target')} hands (--hands ignored)")
        meta.setdefault("resumes", []).append({"at": now_iso(), "git": git_hash()})
        log.save_meta()
    else:
        meta = new_meta(args, engine)
        log = SessionLog(out / meta["session"], meta)

    config = game_config(meta["game"], engine)
    print(f"building {meta['bot']} ...", flush=True)
    bot = make_agent(meta["bot"], **meta["bot_params"])
    if hasattr(bot, "record"):
        bot.record = True
    session = PlaySession(
        bot,
        config,
        log,
        seed=int(meta["seed"]),
        hands_target=meta.get("hands_target"),
        human_seat=int(meta.get("human_seat", 0)),
        names=meta["names"],
        engine=engine,
        warmup=None if args.no_warmup else make_warmup(bot, config, engine),
    )
    try:
        server = PlayServer(session, args.host, args.port)
    except OSError as e:
        print(f"port {args.port} is busy ({e}); using a free port")
        server = PlayServer(session, args.host, 0)
    server.start()
    session.start()
    print(f"session {meta['session']} -> {log.dir}")
    print(f"play at {server.url}  (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        webbrowser.open(server.url)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        session.stop()
        session.join(timeout=5)
        server.shutdown()
        print(summary(session, log))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
r"""Run a job's blocks on the local GPU and on spot GPU instances from EC2 Fleet.

A job is split into **blocks**: independent pieces that each write one directory,
``runs/fleet/<job>/blocks/<block>/``, the same relative path here and on the
instances. A block is a list of commands (``python <script> ...`` from the repository
root, ``PYTHONPATH=.``) that ``cloud/aws/run_block.py`` runs in order, writing
``.done`` in the block's directory when all succeed. The scripts resume from what
their directory already holds, so a block cut short continues on its next worker.

Jobs (``--job``, one or more: several jobs share one queue and one set of workers, under
``runs/fleet/<job>-<job>.../blocks/<job>/``):

* ``calibrate``: ``scripts/calibrate_search.py`` with production's time budgets and
  with fixed iterations, one block per instance type in ``--types`` (and one for the
  local GPU with ``--local``): what each GPU costs per hand.
* ``archetypes``: ``scripts/run_archetypes.py`` against every archetype of the suite
  (roadmap Gate 1): ``--archetype-deals`` duplicate deals each, in blocks of
  ``--block-deals``.
* ``lbr``: ``scripts/run_lbr.py`` for each of ``--lbr-configs`` (raises/preflop),
  ``--lbr-hands`` hands each in blocks of ``--block-hands``. LBR does not resume, so
  its blocks are short.
* ``soak``: ``scripts/soak.py``, ``--soak-hands`` hands in blocks of ``--block-hands``.

Every job but ``calibrate`` plays the fixed-iteration search
(``configs/search_value_net_fixed.yaml``), so the results do not depend on the GPU.

**Workers.** The local GPU (with ``--local``) and up to ``--slots`` spot instances per
region. A launch is one EC2 Fleet ``instant`` request for one spot instance over every
``--types`` instance type in every zone of the region, with the
``price-capacity-optimized`` allocation: EC2 picks the capacity pool least likely to
be interrupted among the cheap ones, under ``--price-cap``. (AWS calls the older Spot
Fleet API legacy; EC2 Fleet replaces it.) A region whose launch failed is retried every
``--retry-min`` minutes: spot capacity comes and goes.

**Guards**, as in ``h2h_coordinator.py``:

* every instance shuts itself down (and so terminates) ``--max-hours`` after boot;
* spend is tracked at each instance's launch-time spot price: no launches once it would
  pass 80% of ``--budget``, and everything is terminated at ``--budget``;
* idle instances are terminated as soon as the queue is empty;
* only instances this runner launched (in its state file) are touched.

**Restart.** State is saved every poll (``<dir>/state.json``); rerunning the same command
adopts the recorded instances and blocks, and waits for a local block an earlier runner
left running. The runner keeps Windows awake while it runs. ``--dry-run`` prints the
blocks and exits; ``--merge-only`` merges what has finished into ``<dir>/summary.md``
and ``summary.json`` (done automatically at the end); ``--status`` prints the blocks per
job, the workers and the spend; ``--terminate`` stops the runner and its local block and
terminates every instance tagged with the run.

The instances fetch ``s3://<bucket>/<bundle-key>`` (``cloud/aws/fleet_bundle.sh``)
through a pre-signed URL and set up with ``cloud/aws/remote_setup.sh``.

python cloud/aws/fleet.py --job calibrate --types g5.xlarge g6.xlarge --budget 3
python cloud/aws/fleet.py --job archetypes lbr soak --soak-hands 20000 --types g5.xlarge \
    --local --budget 200
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import itertools
import json
import math
import os
import random
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
GIT_USR_BIN = Path(r"C:\Program Files\Git\usr\bin")
SSH = str(GIT_USR_BIN / "ssh.exe") if (GIT_USR_BIN / "ssh.exe").exists() else "ssh"
SCP = str(GIT_USR_BIN / "scp.exe") if (GIT_USR_BIN / "scp.exe").exists() else "scp"
AWS = shutil.which("aws") or "aws"
PY = sys.executable
KEYS = {"us-east-2": "pokerbot-pilot", "us-east-1": "pokerbot-us-east-1"}
SG_NAME = "pokerbot-ssh"
SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30", "-o", "LogLevel=ERROR",
]  # fmt: skip
DEAD_STATES = {"shutting-down", "terminated", "stopping", "stopped"}
# fleet errors that hold for the whole region for a while (quota, credentials)
REGION_STOP_ERRORS = ("Limit", "MaxSpotInstanceCountExceeded", "Unauthorized", "Expired", "Auth")

MATCH = "configs/match_search_vn.yaml"
BUDGETED = "configs/search_value_net.yaml"
FIXED = "configs/search_value_net_fixed.yaml"
PRODUCTION = "search:preflop:runs/preflop/preflop_r15.pt"
BLUEPRINT = "neural:runs/dcfr4_distilled_v2"
LBR_CONFIG = "configs/lbr_search_vn.yaml"


def msys_path(p: Path) -> str:
    """``C:/x/y`` -> ``/c/x/y`` for Git Bash's scp (it would read ``C:`` as a host)."""
    s = Path(p).resolve().as_posix()
    if os.name == "nt" and len(s) > 1 and s[1] == ":":
        return f"/{s[0].lower()}{s[2:]}"
    return s


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def pid_alive(pid: int) -> bool:
    """Whether ``pid`` is a running Python process (Windows reuses PIDs)."""
    if os.name == "nt":  # os.kill(pid, 0) would terminate it there
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
        ).stdout
        return f'"{pid}"' in out and "python" in out.lower()
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def keep_awake() -> None:
    """Keep Windows from sleeping while this process lives: the runner and the local
    GPU work for days."""
    if os.name == "nt":
        import ctypes

        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


# -- jobs ---------------------------------------------------------------------------------


@dataclass
class Block:
    name: str
    commands: list[list[str]]  # argument lists of `python ...` from the repository root
    out: str  # the block's directory, relative to the repository root (posix)
    type: str | None = None  # an instance type it must run on; "local": the local GPU only
    group: str = ""  # what it adds up with when merged (an archetype, an LBR config)
    index: int = 0  # its place in the group
    job: str = ""  # the job it belongs to (a run can hold several)
    hours: float = 0.0  # a block that cannot resume: about how long it takes on a spot GPU


def calibrate_blocks(args: argparse.Namespace, base: str) -> list[Block]:
    types = list(args.types) + (["local"] if args.local else [])
    blocks = []
    for t in types:
        out = f"{base}/{t.replace('.', '_')}"
        cmds = [
            [
                "scripts/calibrate_search.py",
                "--config",
                MATCH,
                "--search-config",
                search_cfg,
                "--hands",
                str(args.calibrate_hands),
                "--label",
                label,
                "--out",
                f"{out}/{label}.json",
            ]  # fmt: skip
            for label, search_cfg in (("budget", BUDGETED), ("fixed", FIXED))
        ]
        blocks.append(Block(t.replace(".", "_"), cmds, out, type=t, group=t))
    return blocks


def archetype_blocks(args: argparse.Namespace, base: str) -> list[Block]:
    from pokerbot.agents import SUITE

    names = args.archetypes.split(",") if args.archetypes else list(SUITE)
    per = math.ceil(args.archetype_deals / args.block_deals)
    blocks = []
    for i, name in enumerate(names):
        for j in range(per):
            deals = min(args.block_deals, args.archetype_deals - j * args.block_deals)
            # run_archetypes seeds chunk k of its first archetype with seed + k: blocks
            # 1,000 apart never share a deal (a block has under 1,000 chunks)
            seed = args.seed + 10_000 * i + 1_000 * j
            out = f"{base}/{name}_{j:02d}"
            cmd = [
                "scripts/run_archetypes.py", "--config", MATCH, "--search-config", FIXED,
                "--agent", args.agent, "--reference", args.reference, "--archetypes", name,
                "--deals", str(deals), "--chunk", str(args.chunk), "--seed", str(seed),
                "--adjust-device", "cuda", "--out", f"{out}/result.json", "--resume",
            ]  # fmt: skip
            blocks.append(Block(f"{name}_{j:02d}", [cmd], out, group=name, index=j))
    return blocks


def lbr_blocks(args: argparse.Namespace, base: str) -> list[Block]:
    per = math.ceil(args.lbr_hands / args.block_hands)
    hours = args.block_hands * args.lbr_hand_seconds / 3600
    blocks = []
    for config in args.lbr_configs:
        raises, preflop = config.split("/")
        for j in range(per):
            hands = min(args.block_hands, args.lbr_hands - j * args.block_hands)
            seed = args.seed + 100_000 * j
            name = f"{raises}_{preflop}_{j:02d}"
            out = f"{base}/{name}"
            cmd = [
                "scripts/run_lbr.py", "--config", LBR_CONFIG, "--search-config", FIXED,
                "--raises", raises, "--preflop", preflop, "--hands", str(hands),
                "--seed", str(seed), "--out", f"{out}/lbr.json", "--log-dir", f"{out}/log",
            ]  # fmt: skip
            blocks.append(Block(name, [cmd], out, group=config, index=j, hours=hours))
    return blocks


def soak_blocks(args: argparse.Namespace, base: str) -> list[Block]:
    size = args.soak_block_hands
    per = math.ceil(args.soak_hands / size)
    blocks = []
    for j in range(per):
        hands = min(size, args.soak_hands - j * size)
        out = f"{base}/soak_{j:03d}"
        cmd = [
            "scripts/soak.py", "--config", MATCH, "--search-config", FIXED,
            "--hands", str(hands), "--seed", str(args.seed + j), "--out-dir", out,
            "--opponents", *args.soak_opponents,
        ]  # fmt: skip
        blocks.append(Block(f"soak_{j:03d}", [cmd], out, group="soak", index=j))
    return blocks


JOBS = {
    "calibrate": calibrate_blocks,
    "archetypes": archetype_blocks,
    "lbr": lbr_blocks,
    "soak": soak_blocks,
}


# -- merging --------------------------------------------------------------------------------


def _load_script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod  # dataclasses look their module up while it executes
    spec.loader.exec_module(mod)
    return mod


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def merge_calibrate(root: Path, blocks: list[Block], meta: dict) -> dict:
    rows = []
    for b in blocks:
        budget = _read(root / b.out / "budget.json")
        fixed = _read(root / b.out / "fixed.json")
        if not budget or not fixed:
            continue
        price = (meta.get(b.name) or {}).get("price")
        sph = fixed["seconds_per_hand"]
        rows.append(
            {
                "type": b.type,
                "gpu": fixed["machine"]["gpu"],
                "price": price,
                "budget_iterations": {
                    k: v["iterations_mean"] for k, v in budget["streets"].items()
                },
                "fixed_seconds": {k: v["seconds_mean"] for k, v in fixed["streets"].items()},
                "fixed_seconds_per_hand": sph,
                "budget_seconds_per_hand": budget["seconds_per_hand"],
                "dollars_per_1000_hands": None if price is None else price * sph * 1000 / 3600,
            }
        )
    lines = [
        "| type | GPU | iterations in 4/2/1 s (flop/turn/river) | fixed: s per decision "
        "(flop/turn/river) | fixed: s/hand | $/h | $ per 1,000 hands |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for r in rows:
        it = " / ".join(
            f"{r['budget_iterations'].get(s, 0):.0f}" for s in ("flop", "turn", "river")
        )
        sec = " / ".join(f"{r['fixed_seconds'].get(s, 0):.2f}" for s in ("flop", "turn", "river"))
        price = "local" if r["price"] is None else f"{r['price']:.3f}"
        cost = "-" if r["dollars_per_1000_hands"] is None else f"{r['dollars_per_1000_hands']:.2f}"
        lines.append(
            f"| {r['type']} | {r['gpu']} | {it} | {sec} | {r['fixed_seconds_per_hand']:.2f} "
            f"| {price} | {cost} |"
        )
    return {"lines": lines, "rows": rows}


def merge_archetypes(root: Path, blocks: list[Block], meta: dict) -> dict:
    mod = _load_script("run_archetypes")
    merged: dict[str, Any] | None = None
    deals: dict[str, int] = {}
    for b in sorted(blocks, key=lambda b: (b.group, b.index)):
        data = _read(root / b.out / "result.json")
        if not data:
            continue
        if merged is None:
            merged = {k: v for k, v in data.items() if k != "archetypes"}
            merged["archetypes"] = {}
        for name, entry in data["archetypes"].items():
            out = merged["archetypes"].setdefault(name, {})
            for role, part in entry.items():
                chunks = out.setdefault(role, {"chunks": []})["chunks"]
                for c in part.get("chunks", []):
                    chunks.append({**c, "k": 1_000 * b.index + int(c["k"])})
        deals[b.group] = deals.get(b.group, 0) + int(data.get("deals", 0))
    if merged is None:
        return {"lines": ["no archetype results yet"], "merged": None}
    merged["deals"] = max(deals.values())  # per archetype, as in run_archetypes
    bb = int((merged.get("game") or {}).get("big_blind", 100))
    summary = mod.summarise(merged, bb, 2000)
    return {"lines": mod.table(summary, merged), "merged": merged, "summary": summary}


def merge_lbr(root: Path, blocks: list[Block], meta: dict) -> dict:
    import numpy as np

    from pokerbot.eval.lbr import STREETS, LBRResult

    lines: list[str] = []
    out: dict[str, Any] = {}
    for group in sorted({b.group for b in blocks}):
        pay, adj, end, seconds, resets = [], [], [], 0.0, 0
        actions: dict[str, dict[str, int]] = {s: {} for s in STREETS}
        cost: dict[str, list[float]] = {}
        opponent = ""
        for b in sorted((b for b in blocks if b.group == group), key=lambda b: b.index):
            d = _read(root / b.out / "lbr.json")
            if not d or "samples" not in d:
                continue
            opponent = d.get("opponent", opponent)
            pay += d["samples"]["payoffs"]
            adj += d["samples"]["adjusted"]
            end += d["samples"]["end_street"]
            seconds += float(d.get("seconds", 0.0))
            resets += int(d.get("range_resets", 0))
            for row in d.get("by_street", []):
                for k, v in row.get("lbr_actions", {}).items():
                    actions[row["street"]][k] = actions[row["street"]].get(k, 0) + int(v)
            for k, (cnt, sec) in (d.get("opponent_cost") or {}).items():
                c = cost.setdefault(k, [0, 0.0])
                c[0] += cnt
                c[1] += sec
        if not pay:
            continue
        res = LBRResult(
            opponent=opponent,
            big_blind=100,
            payoffs=np.array(pay),
            adjusted=np.array(adj),
            end_street=np.array(end),
            actions=actions,
            seconds=seconds,
            range_resets=resets,
            opponent_cost={k: [int(v[0]), float(v[1])] for k, v in cost.items()},
        )
        lines += [f"### {group}", "", "```", res.summary(), "```", ""]
        d = res.to_dict()
        d.pop("samples", None)
        out[group] = d
    return {"lines": lines or ["no LBR results yet"], "configs": out}


def merge_soak(root: Path, blocks: list[Block], meta: dict, dest: Path | None = None) -> dict:
    from pokerbot.eval.soak import summarize

    dest = dest or root / "runs" / "fleet" / "soak" / "merged"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    n = 0
    hands = 0
    for b in sorted(blocks, key=lambda b: b.index):
        for p in sorted((root / b.out).glob("chunk_*.json")):
            rec = _read(p)
            if rec is None:
                continue
            hands += int(rec.get("hands", 0) or 0)
            shutil.copyfile(p, dest / f"chunk_{n:05d}.json")
            n += 1
    if n == 0:
        return {"lines": ["no soak chunks yet"], "summary": None}
    s = summarize(dest, None, 100)
    text = (dest / "soak.md").read_text(encoding="utf-8") if (dest / "soak.md").exists() else ""
    return {"lines": text.splitlines(), "summary": s, "chunks": n, "hands": hands}


MERGES = {
    "calibrate": merge_calibrate,
    "archetypes": merge_archetypes,
    "lbr": merge_lbr,
    "soak": merge_soak,
}


# -- launch requests ------------------------------------------------------------------------


def launch_template_data(
    ami: str, key: str, sg: str, root_device: str, disk_gb: int, max_hours: float, job: str
) -> dict:
    """EC2 launch template data: the AMI, SSH key and security group, a disk that
    outlives nothing, and a shutdown timer that terminates the instance."""
    user_data = f"#!/bin/bash\nshutdown -h +{int(max_hours * 60)}\n"
    tags = [
        {"Key": "Name", "Value": f"pokerbot-fleet-{job}"},
        {"Key": "project", "Value": "pokerbot"},
        {"Key": "purpose", "Value": f"fleet-{job}"},
    ]
    return {
        "ImageId": ami,
        "KeyName": key,
        "SecurityGroupIds": [sg],
        "InstanceInitiatedShutdownBehavior": "terminate",
        "UserData": base64.b64encode(user_data.encode()).decode(),
        "BlockDeviceMappings": [
            {
                "DeviceName": root_device,
                "Ebs": {"VolumeSize": disk_gb, "VolumeType": "gp3", "DeleteOnTermination": True},
            }
        ],
        "TagSpecifications": [
            {"ResourceType": "instance", "Tags": tags},
            {"ResourceType": "volume", "Tags": tags},
        ],
        "MetadataOptions": {"HttpTokens": "required"},
    }


def fleet_configs(
    template: str, version: str, types: list[str], subnets: list[tuple[str, str]], cap: float
) -> list[dict]:
    """``--launch-template-configs`` of a one-instance spot request over every instance
    type and zone (subnet) of a region."""
    overrides = [
        {"InstanceType": t, "SubnetId": s, "MaxPrice": f"{cap:.4f}"}
        for t in types
        for s, _ in subnets
    ]
    return [
        {
            "LaunchTemplateSpecification": {"LaunchTemplateName": template, "Version": version},
            "Overrides": overrides,
        }
    ]


# -- the runner -----------------------------------------------------------------------------


@dataclass
class Worker:
    name: str
    kind: str  # local | aws
    region: str = ""
    instance_id: str = ""
    type: str = "local"
    zone: str = ""
    host: str = ""
    price: float = 0.0  # $/h at launch
    launched: float = 0.0
    ended: float = 0.0
    state: str = "ready"  # aws: booting, setup, ready, running, dead; local: ready, running, off
    block: str | None = None
    since: float = 0.0
    unreachable: float = 0.0
    fails: int = 0


@dataclass
class State:
    blocks: dict[str, str] = field(default_factory=dict)  # todo | running | done
    meta: dict[str, dict] = field(default_factory=dict)  # block -> who ran it and at what price
    n_launched: int = 0


class Fleet:
    def __init__(self, args: argparse.Namespace, blocks: list[Block]) -> None:
        self.args = args
        self.dir = ROOT / args.dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.by_name = {b.name: b for b in blocks}
        self.state_path = self.dir / "state.json"
        self.st = State(blocks={b.name: "todo" for b in blocks})
        self.workers: dict[str, Worker] = {}
        self.last_try: dict[str, float] = {}
        self.templates: dict[str, tuple[str, str]] = {}
        self.local_proc: subprocess.Popen | None = None
        self.stopped_launching = False
        self.fails: dict[str, int] = {}  # per block, this runner's
        self._load()
        for b in blocks:  # a finished block stays finished across restarts
            if self.done(b):
                self.st.blocks[b.name] = "done"
        if args.local:
            w = self.workers.setdefault("local", Worker("local", "local"))
            if w.state == "off":
                w.state, w.fails = "ready", 0

    # -- state ----------------------------------------------------------------------------

    def _load(self) -> None:
        st = _read(self.state_path)
        if not st:
            return
        for name, status in st["blocks"].items():
            if name in self.st.blocks:
                self.st.blocks[name] = "todo" if status in ("running", "failed") else status
        self.st.meta = st.get("meta", {})
        self.st.n_launched = st.get("n_launched", 0)
        self.workers = {k: Worker(**w) for k, w in st["workers"].items()}
        for w in self.workers.values():
            if w.block is not None and w.state != "dead":
                self.st.blocks[w.block] = "running"
        log(f"# resumed: {self.counts()}, workers {[w.name for w in self.alive()]}")

    def save(self) -> None:
        data = {
            "blocks": self.st.blocks,
            "meta": self.st.meta,
            "n_launched": self.st.n_launched,
            "workers": {k: asdict(w) for k, w in self.workers.items()},
            "spent": self.spent(),
        }
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(self.state_path)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.st.blocks.values():
            out[s] = out.get(s, 0) + 1
        return out

    def alive(self) -> list[Worker]:
        return [w for w in self.workers.values() if w.kind == "aws" and w.state != "dead"]

    def spent(self) -> float:
        now = time.time()
        return sum(
            ((w.ended or now) - w.launched) / 3600 * w.price
            for w in self.workers.values()
            if w.kind == "aws" and w.launched
        )

    # -- blocks ---------------------------------------------------------------------------

    def done(self, b: Block) -> bool:
        return (ROOT / b.out / ".done").exists()

    def progress(self, b: Block) -> int:
        """Bytes already in the block's directory: partial blocks go first."""
        d = ROOT / b.out
        return sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) if d.exists() else 0

    def needed_hours(self, b: Block) -> float:
        """How long a block that cannot resume needs on a spot GPU: its estimate, or
        the longest a finished block of its job took there, if longer."""
        if not b.hours:
            return 0.0
        took = [
            m.get("hours") or 0.0
            for n, m in self.st.meta.items()
            if n in self.by_name and self.by_name[n].job == b.job and m.get("type") != "local"
        ]
        return max([b.hours, *took])

    def next_block(self, worker_type: str, hours_left: float = math.inf) -> Block | None:
        """The next block for a worker: partial blocks first, and none that cannot
        resume and would outlast the worker."""
        todo = [
            self.by_name[n]
            for n, s in self.st.blocks.items()
            if s == "todo"
            and self.by_name[n].type in (None, worker_type)
            and self.needed_hours(self.by_name[n]) <= hours_left
        ]
        todo.sort(key=lambda b: -self.progress(b))
        return todo[0] if todo else None

    def write_spec(self, b: Block) -> Path:
        d = ROOT / b.out
        d.mkdir(parents=True, exist_ok=True)
        (d / ".done").unlink(missing_ok=True)
        path = d / "block.json"
        path.write_text(json.dumps({"name": b.name, "out": b.out, "commands": b.commands}))
        return path

    def block_failed(self, b: Block, why: str) -> None:
        """Back to the queue, or out of it at its third failure: a broken block must not
        take one worker after another down with it. A rerun of the runner retries it."""
        n = self.fails[b.name] = self.fails.get(b.name, 0) + 1
        self.st.blocks[b.name] = "failed" if n >= 3 else "todo"
        log(f"{b.name} exited unfinished ({why}){'; given up' if n >= 3 else ''}")

    def assign(self, w: Worker, b: Block) -> None:
        self.st.blocks[b.name] = "running"
        self.st.meta[b.name] = {"worker": w.name, "type": w.type, "price": w.price or None}
        w.state, w.block, w.since = "running", b.name, time.time()

    # -- commands -------------------------------------------------------------------------

    def run(self, cmd: list[str], timeout: float = 120) -> tuple[int, str]:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return 124, "timeout"
        return p.returncode, (p.stdout + p.stderr).strip()

    def aws(self, region: str, *args: str, timeout: float = 120) -> tuple[int, str]:
        return self.run([AWS, "--region", region, *args], timeout=timeout)

    def key_file(self, w: Worker) -> str:
        return msys_path(Path.home() / ".ssh" / f"{KEYS[w.region]}.pem")

    def ssh(self, w: Worker, command: str, timeout: float = 60) -> tuple[int, str]:
        return self.run(
            [SSH, "-i", self.key_file(w), *SSH_OPTS, "-n", f"ubuntu@{w.host}", command],
            timeout=timeout,
        )

    def scp(self, w: Worker, src: str, dst: str, timeout: float = 300) -> bool:
        return self.run([SCP, "-q", "-i", self.key_file(w), *SSH_OPTS, src, dst], timeout)[0] == 0

    # -- the local worker -------------------------------------------------------------------

    def step_local(self) -> None:
        w = self.workers.get("local")
        if w is None or w.state == "off":
            return
        if w.state == "running":
            if self.local_proc is not None:
                rc = self.local_proc.poll()
            else:  # started by an earlier runner: it runs while its process does
                pid = read_pid(self.dir / "local.pid")
                rc = None if pid is not None and pid_alive(pid) else -1
            if rc is None:
                return
            b = self.by_name[w.block]
            w.state, w.block, self.local_proc = "ready", None, None
            (self.dir / "local.pid").unlink(missing_ok=True)
            if self.done(b):
                self.st.blocks[b.name] = "done"
                log(f"local: {b.name} done")
            else:
                w.fails += 1
                self.block_failed(b, f"local, exit {rc}, its failure {w.fails}")
                if w.fails >= 3:
                    w.state = "off"
                    log("local: 3 failures, local worker off (a rerun turns it back on)")
                    return
        if w.state == "ready":
            b = self.next_block("local")
            if b is None:
                return
            spec = self.write_spec(b)
            logf = open(ROOT / b.out / "local.log", "a")  # noqa: SIM115
            self.local_proc = subprocess.Popen(
                [PY, "cloud/aws/run_block.py", str(spec)],
                cwd=ROOT,
                stdout=logf,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONPATH": "."},
            )
            (self.dir / "local.pid").write_text(str(self.local_proc.pid))
            self.assign(w, b)
            log(f"local: {b.name} started")

    # -- spot workers ---------------------------------------------------------------------------

    def subnets(self, region: str) -> list[tuple[str, str]]:
        rc, out = self.aws(
            region, "ec2", "describe-subnets", "--filters", "Name=default-for-az,Values=true",
            "--query", "Subnets[].[SubnetId,AvailabilityZone]", "--output", "text",
        )  # fmt: skip
        if rc:
            return []
        subs = [tuple(line.split()) for line in out.splitlines() if line.strip()]
        random.shuffle(subs)
        return subs  # type: ignore[return-value]

    def security_group(self, region: str) -> str:
        rc, ip = self.run(["curl", "-s", "https://checkip.amazonaws.com"], timeout=30)
        rc2, sg = self.aws(
            region, "ec2", "describe-security-groups",
            "--filters", f"Name=group-name,Values={SG_NAME}",
            "--query", "SecurityGroups[0].GroupId", "--output", "text",
        )  # fmt: skip
        if rc == 0 and rc2 == 0 and ip.strip():
            self.aws(
                region, "ec2", "authorize-security-group-ingress", "--group-id", sg.strip(),
                "--protocol", "tcp", "--port", "22", "--cidr", f"{ip.strip()}/32",
            )  # fmt: skip
        return sg.strip()

    def ami(self, region: str) -> tuple[str, str, int]:
        """The Deep Learning Base AMI (Ubuntu 22.04), its root device and snapshot GB."""
        rc, out = self.aws(
            region, "ssm", "get-parameter", "--name",
            "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id",
            "--query", "Parameter.Value", "--output", "text",
        )  # fmt: skip
        if rc:
            return "", "", 0
        ami = out.strip()
        rc, out = self.aws(
            region, "ec2", "describe-images", "--image-ids", ami, "--query",
            "Images[0].[RootDeviceName,BlockDeviceMappings[0].Ebs.VolumeSize]", "--output", "text",
        )  # fmt: skip
        parts = out.split()
        if rc or len(parts) < 2:
            return "", "", 0
        return ami, parts[0], int(parts[1])

    def launch_template(self, region: str) -> tuple[str, str] | None:
        """This run's launch template in ``region``: a new version of
        ``pokerbot-fleet-<job>`` made once per run."""
        if region in self.templates:
            return self.templates[region]
        (ami, root_dev, snap_gb), sg = self.ami(region), self.security_group(region)
        if not ami or not sg or sg == "None":
            log(f"{region}: no AMI or security group ({ami!r}, {sg!r}); region skipped")
            return None
        data = launch_template_data(
            ami, KEYS[region], sg, root_dev, max(snap_gb, 120), self.args.max_hours, self.args.job
        )
        path = self.dir / f"template_{region}.json"
        path.write_text(json.dumps(data))
        name = f"pokerbot-fleet-{self.args.job}"
        src = f"file://{path.resolve().as_posix()}"
        rc, out = self.aws(
            region, "ec2", "describe-launch-templates", "--launch-template-names", name,
            "--query", "LaunchTemplates[0].LaunchTemplateId", "--output", "text",
        )  # fmt: skip
        if rc == 0 and out.strip().startswith("lt-"):
            rc, out = self.aws(
                region, "ec2", "create-launch-template-version", "--launch-template-name", name,
                "--launch-template-data", src,
                "--query", "LaunchTemplateVersion.VersionNumber", "--output", "text",
            )  # fmt: skip
        else:
            rc, out = self.aws(
                region, "ec2", "create-launch-template", "--launch-template-name", name,
                "--launch-template-data", src,
                "--query", "LaunchTemplate.LatestVersionNumber", "--output", "text",
            )  # fmt: skip
        if rc:
            log(f"{region}: launch template failed: {out[-300:]}")
            return None
        self.templates[region] = (name, out.strip())
        return self.templates[region]

    def spot_price(self, region: str, itype: str, zone: str) -> float:
        rc, out = self.aws(
            region, "ec2", "describe-spot-price-history", "--instance-types", itype,
            "--product-descriptions", "Linux/UNIX", "--availability-zone", zone,
            "--start-time", time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "--query", "SpotPriceHistory[0].SpotPrice", "--output", "text",
        )  # fmt: skip
        try:
            return float(out.strip())
        except ValueError:
            return self.args.price_cap

    def try_launch(self, region: str, types: list[str]) -> None:
        self.last_try[region] = time.time()
        tpl = self.launch_template(region)
        subnets = self.subnets(region)
        if tpl is None or not subnets:
            return
        path = self.dir / f"fleet_{region}.json"
        path.write_text(json.dumps(fleet_configs(*tpl, types, subnets, self.args.price_cap)))
        rc, out = self.aws(
            region, "ec2", "create-fleet", "--type", "instant",
            "--target-capacity-specification",
            "TotalTargetCapacity=1,DefaultTargetCapacityType=spot",
            "--spot-options",
            "AllocationStrategy=price-capacity-optimized,InstanceInterruptionBehavior=terminate",
            "--launch-template-configs", f"file://{path.resolve().as_posix()}",
            "--output", "json", timeout=180,
        )  # fmt: skip
        if rc:
            log(f"{region}: create-fleet failed: {out[-300:]}")
            return
        res = json.loads(out)
        zones = dict(subnets)
        for inst in res.get("Instances") or []:
            ids = inst.get("InstanceIds") or []
            if not ids:
                continue
            ov = (inst.get("LaunchTemplateAndOverrides") or {}).get("Overrides") or {}
            itype = inst.get("InstanceType") or ov.get("InstanceType", "?")
            zone = zones.get(ov.get("SubnetId", ""), "")
            self.st.n_launched += 1
            name = f"aws-{self.st.n_launched}"
            w = Worker(name, "aws", region, ids[0], type=itype, zone=zone, state="booting")
            w.launched = w.since = time.time()
            w.price = self.spot_price(region, itype, zone)
            self.workers[name] = w
            where = f"{itype} in {zone}"
            log(f"{region}: launched {w.instance_id} ({name}, {where}) at ${w.price:.3f}/h")
            return
        errors = sorted({e.get("ErrorCode", "?") for e in res.get("Errors") or []})
        log(f"{region}: no spot instance for {types}: {errors}")
        if any(any(x in e for x in REGION_STOP_ERRORS) for e in errors):
            self.last_try[region] = time.time() + 3600  # quota or credentials: back off an hour

    def refresh_instances(self) -> None:
        by_region: dict[str, list[Worker]] = {}
        for w in self.alive():
            by_region.setdefault(w.region, []).append(w)
        for region, ws in by_region.items():
            rc, out = self.aws(
                region, "ec2", "describe-instances", "--instance-ids", *[w.instance_id for w in ws],
                "--query",
                "Reservations[].Instances[].[InstanceId,State.Name,PublicDnsName,StateReason.Message]",
                "--output", "text",
            )  # fmt: skip
            if rc:
                log(f"{region}: describe-instances failed: {out[-200:]}")
                continue
            info = {line.split("\t")[0]: line.split("\t") for line in out.splitlines() if line}
            for w in ws:
                p = info.get(w.instance_id)
                if p is None:
                    continue
                if p[1] in DEAD_STATES:
                    self.mark_dead(w, f"{p[1]} ({p[3] if len(p) > 3 else ''})")
                elif p[1] == "running" and p[2] and p[2] != "None":
                    w.host = p[2]

    def mark_dead(self, w: Worker, why: str) -> None:
        if w.block is not None:
            b = self.by_name[w.block]
            self.pull(w)
            if self.st.blocks.get(b.name) == "running":
                self.st.blocks[b.name] = "done" if self.done(b) else "todo"
            log(f"{w.name}: {b.name} back to the queue with {self.progress(b)} bytes")
        w.state, w.block, w.ended = "dead", None, w.ended or time.time()
        log(f"{w.name}: {w.instance_id} ended: {why}")

    def terminate(self, w: Worker, why: str) -> None:
        """Pull its block, shut the instance down over SSH (it terminates on shutdown),
        then ask EC2 to terminate it too."""
        if w.host:
            if w.block is not None:
                self.pull(w)
            self.ssh(w, "sudo shutdown -h now", timeout=30)
        rc, out = self.aws(w.region, "ec2", "terminate-instances", "--instance-ids", w.instance_id)
        if rc:
            log(f"{w.name}: terminate-instances failed ({out[-200:]}); relying on its shutdown")
        self.mark_dead(w, f"terminated by the runner: {why}")

    def presigned_url(self) -> str:
        rc, out = self.aws(
            self.args.bucket_region, "s3", "presign",
            f"s3://{self.args.bucket}/{self.args.bundle_key}", "--expires-in", "3600",
        )  # fmt: skip
        if rc:
            raise RuntimeError(f"presign failed: {out[-300:]}")
        return out.strip()

    def push(self, w: Worker, b: Block) -> bool:
        """The block's directory (its spec and any partial output) to the instance."""
        tgz = self.dir / "transfer" / f"push_{b.name}.tgz"
        tgz.parent.mkdir(exist_ok=True)
        with tarfile.open(tgz, "w:gz") as tar:
            tar.add(ROOT / b.out, arcname=b.out)
        if not self.scp(w, msys_path(tgz), f"ubuntu@{w.host}:/tmp/push_{b.name}.tgz"):
            return False
        rc, _ = self.ssh(w, f"cd ~/poker-bot && tar -xzf /tmp/push_{b.name}.tgz")
        return rc == 0

    def pull(self, w: Worker) -> bool:
        """The block's directory from the instance, over the local copy; True when done."""
        if w.block is None or not w.host:
            return False
        b = self.by_name[w.block]
        rc, _ = self.ssh(
            w, f"cd ~/poker-bot && tar -czf /tmp/pull_{b.name}.tgz {shlex.quote(b.out)}"
        )
        tgz = self.dir / "transfer" / f"pull_{b.name}.tgz"
        tgz.parent.mkdir(exist_ok=True)
        if rc == 0 and self.scp(w, f"ubuntu@{w.host}:/tmp/pull_{b.name}.tgz", msys_path(tgz)):
            with tarfile.open(tgz) as tar:
                members = [m for m in tar.getmembers() if m.name.startswith(b.out)]
                tar.extractall(ROOT, members=members, filter="data")
        return self.done(b)

    def step_aws(self, w: Worker) -> None:
        now = time.time()
        if w.state == "booting":
            if not w.host:
                return
            rc, _ = self.ssh(w, "true", timeout=40)
            if rc != 0:
                if now - w.since > 900:
                    self.terminate(w, "no SSH 15 min after launch")
                return
            url = self.presigned_url()
            cmd = (
                "nohup bash -c 'set -eo pipefail; mkdir -p ~/poker-bot && cd ~/poker-bot && "
                f'curl -sSfL "{url}" | tar -x && bash cloud/aws/remote_setup.sh && '
                "touch ~/setup.done' > ~/setup.log 2>&1 < /dev/null & echo started"
            )
            if self.ssh(w, cmd)[0] == 0:
                w.state, w.since = "setup", now
                log(f"{w.name}: setting up")
            return
        if w.state == "setup":
            if self.ssh(w, "test -f ~/setup.done")[0] == 0:
                w.state, w.since = "ready", now
                log(f"{w.name}: set up in {(now - w.launched) / 60:.0f} min since launch")
            elif now - w.since > 2400:
                _, tail = self.ssh(w, "tail -5 ~/setup.log")
                self.terminate(w, f"setup not done after 40 min: {tail[-300:]}")
            return
        if w.state == "running":
            b = self.by_name[w.block]
            pid = shlex.quote(f"{b.out}/.pid")
            # is it running, then what it wrote (.done comes before its process exits)
            rc, out = self.ssh(
                w, f"cd ~/poker-bot && kill -0 $(cat {pid}) 2>/dev/null && echo up || echo down"
            )
            if rc != 0:  # unreachable: an interruption the API has not shown yet?
                w.unreachable = w.unreachable or now
                if now - w.unreachable > 600:
                    self.terminate(w, "no SSH for 10 min")
                return
            w.unreachable = 0.0
            if self.pull(w):
                self.st.blocks[b.name] = "done"
                self.st.meta[b.name]["hours"] = (now - w.since) / 3600
                log(f"{w.name}: {b.name} done")
                w.state, w.block, w.since = "ready", None, now
            elif out.strip().endswith("down"):
                _, tail = self.ssh(w, f"tail -3 ~/block_{b.name}.log")
                w.fails += 1
                w.state, w.block = "ready", None
                self.block_failed(b, f"{w.name}, its failure {w.fails}: {tail[-300:]}")
                if w.fails >= 3:
                    self.terminate(w, "3 block failures")
                return
        if w.state == "ready":
            # it shuts itself down --max-hours after boot; keep a quarter hour spare
            left = self.args.max_hours - (now - w.launched) / 3600 - 0.25
            b = self.next_block(w.type, left)
            if b is None:
                self.terminate(w, f"nothing left for it in its last {left:.1f} h")
                return
            self.write_spec(b)
            if not self.push(w, b):
                return
            spec = f"{b.out}/block.json"
            start = (
                "cd ~/poker-bot && source .venv/bin/activate && "
                f"(nohup python cloud/aws/run_block.py {shlex.quote(spec)} "
                f"> ~/block_{b.name}.log 2>&1 < /dev/null & echo $! > {shlex.quote(b.out)}/.pid) "
                "&& echo started"
            )
            if self.ssh(w, start)[0] == 0:
                self.assign(w, b)
                log(f"{w.name}: {b.name} started")

    def launch_types(self) -> list[str] | None:
        """Instance types for the next launch, or None when every waiting block has an
        instance coming (booting, setting up or idle). A block that needs one instance
        type gets a launch of that type alone; any other gets every ``--types``."""
        idle: dict[str, int] = {}
        for w in self.alive():
            if w.state in ("booting", "setup", "ready"):
                idle[w.type] = idle.get(w.type, 0) + 1
        spare = sum(idle.values())
        for name, s in self.st.blocks.items():
            b = self.by_name[name]
            if s != "todo" or b.type == "local":
                continue
            if b.type:
                if idle.get(b.type, 0) > 0:
                    idle[b.type] -= 1
                    spare -= 1
                    continue
                return [b.type]
            if spare > 0:
                spare -= 1
                continue
            return list(self.args.types)
        return None

    def maybe_launch(self) -> None:
        if self.stopped_launching:
            return
        types = self.launch_types()
        if types is None:
            return
        alive = self.alive()
        projected = self.spent() + sum(w.price for w in alive) + self.args.price_cap
        if projected > 0.8 * self.args.budget:
            log(f"# budget: spent ${self.spent():.2f}, no more launches")
            self.stopped_launching = True
            return
        for region in self.args.regions:
            if region not in KEYS:
                continue
            if sum(1 for w in alive if w.region == region) >= self.args.slots:
                continue
            if time.time() - self.last_try.get(region, 0) < self.args.retry_min * 60:
                continue
            self.try_launch(region, types)
            return  # one launch attempt per poll

    def merge(self) -> None:
        done = [b for b in self.by_name.values() if self.done(b)]
        n = len(self.by_name)
        head = [
            f"# fleet run {self.args.job}: {len(done)} of {n} blocks done",
            "",
            f"AWS spend about ${self.spent():.2f} over {self.st.n_launched} launches.",
            "",
        ]
        lines: list[str] = []
        data: dict[str, Any] = {"run": self.args.job, "done": len(done), "blocks": n}
        for job in self.args.jobs:
            blocks = [b for b in done if b.job == job]
            total = sum(1 for b in self.by_name.values() if b.job == job)
            if job == "soak":
                res = merge_soak(ROOT, blocks, self.st.meta, dest=self.dir / "soak_merged")
            else:
                res = MERGES[job](ROOT, blocks, self.st.meta)
            if len(self.args.jobs) > 1:
                lines += [f"## {job}: {len(blocks)} of {total} blocks", ""]
            lines += [*res["lines"], ""]
            data[job] = {k: v for k, v in res.items() if k != "lines"}
        data["spent"] = self.spent()
        (self.dir / "summary.md").write_text("\n".join(head + lines) + "\n", encoding="utf-8")
        (self.dir / "summary.json").write_text(json.dumps(data, indent=1, default=str))
        for line in head + lines:
            log(f"# {line}")

    def status(self) -> str:
        """Blocks per job and state, the workers alive, and the spend so far."""
        out = [f"run {self.args.job}: spent ${self.spent():.2f} over {self.st.n_launched} launches"]
        for job in self.args.jobs:
            counts: dict[str, int] = {}
            for b in self.by_name.values():
                if b.job == job:
                    s = self.st.blocks[b.name]
                    counts[s] = counts.get(s, 0) + 1
            out.append(f"  {job}: {counts}")
        now = time.time()
        for w in self.workers.values():
            if w.kind == "local" or w.state != "dead":
                age = f"{(now - w.launched) / 3600:.1f} h, ${w.price:.3f}/h" if w.launched else ""
                out.append(f"  {w.name} {w.type} {w.region} {w.state} {w.block or ''} {age}")
        return "\n".join(out)

    def stop_everything(self) -> None:
        """Stop the runner process and the local block it started (their PIDs are in
        the state directory), then terminate every instance tagged with this run, recorded
        or not (a runner killed mid-launch can leave one unrecorded)."""
        for f in ("runner.pid", "local.pid"):
            path = self.dir / f
            if not path.exists():
                continue
            pid = path.read_text().strip()
            if os.name == "nt":
                self.run(["taskkill", "/T", "/F", "/PID", pid], timeout=60)
            else:
                self.run(["kill", pid], timeout=30)
            path.unlink(missing_ok=True)
            log(f"stopped {f} {pid}")
        self._load()  # what the runner saved last
        for region in self.args.regions:
            rc, out = self.aws(
                region, "ec2", "describe-instances", "--filters",
                f"Name=tag:purpose,Values=fleet-{self.args.job}",
                "Name=instance-state-name,Values=pending,running,stopping",
                "--query", "Reservations[].Instances[].InstanceId", "--output", "text",
            )  # fmt: skip
            ids = out.split() if rc == 0 else []
            if ids:
                self.aws(region, "ec2", "terminate-instances", "--instance-ids", *ids)
                log(f"{region}: terminated {' '.join(ids)}")
        for w in self.alive():
            w.state, w.ended = "dead", w.ended or time.time()
            if w.block is not None and self.st.blocks.get(w.block) == "running":
                self.st.blocks[w.block] = "todo"
            w.block = None
        local = self.workers.get("local")
        if local is not None and local.state == "running":
            if local.block is not None:
                self.st.blocks[local.block] = "todo"
            local.state, local.block = "ready", None
        self.save()

    def loop(self) -> None:
        pid_file = self.dir / "runner.pid"
        pid = read_pid(pid_file)
        if pid is not None and pid != os.getpid() and pid_alive(pid):
            raise SystemExit(f"a runner (pid {pid}, {pid_file}) is already running this job")
        pid_file.write_text(str(os.getpid()))
        keep_awake()
        log(f"# {self.args.job}: {len(self.by_name)} blocks, {self.counts()}")
        while True:
            try:
                self.step_local()
                self.refresh_instances()
                for w in list(self.alive()):
                    self.step_aws(w)
                if self.spent() >= self.args.budget:
                    for w in self.alive():
                        self.terminate(w, f"budget ${self.args.budget} reached")
                    self.stopped_launching = True
                self.maybe_launch()
            except Exception as e:  # a failed AWS or SSH call is retried next poll
                log(f"! {e!r}")
            self.save()
            if all(s in ("done", "failed") for s in self.st.blocks.values()):
                failed = [n for n, s in self.st.blocks.items() if s == "failed"]
                if failed:
                    log(f"# given up after 3 failures each (rerun to retry): {failed}")
                for w in self.alive():
                    self.terminate(w, "job complete")
                self.save()
                self.merge()
                (self.dir / "runner.pid").unlink(missing_ok=True)
                return
            local = self.workers.get("local")
            no_local = local is None or local.state == "off"
            if not self.alive() and no_local and self.stopped_launching:
                log("# no workers left and no more launches; stopping (rerun to resume)")
                self.merge()
                (self.dir / "runner.pid").unlink(missing_ok=True)
                return
            time.sleep(self.args.poll)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--job", required=True, nargs="+", choices=sorted(JOBS), help="one or more jobs, one queue"
    )
    ap.add_argument("--dir", help="state and block results (default runs/fleet/<job>[-<job>...])")
    ap.add_argument("--dry-run", action="store_true", help="print the blocks and exit")
    ap.add_argument("--merge-only", action="store_true", help="merge finished blocks and exit")
    ap.add_argument("--status", action="store_true", help="print blocks, workers, spend; exit")
    ap.add_argument(
        "--terminate",
        action="store_true",
        help="stop the runner and its local block, terminate the run's instances, exit",
    )
    jobs = ap.add_argument_group("jobs")
    jobs.add_argument("--seed", type=int, default=50_000)
    jobs.add_argument("--calibrate-hands", type=int, default=24)
    jobs.add_argument("--agent", default=PRODUCTION, help="archetypes: the agent under test")
    jobs.add_argument("--reference", default=BLUEPRINT, help="archetypes: the reference agent")
    jobs.add_argument("--archetypes", help="archetypes: comma-separated (default: the suite)")
    jobs.add_argument("--archetype-deals", type=int, default=10_000)
    jobs.add_argument("--block-deals", type=int, default=2_500)
    jobs.add_argument("--chunk", type=int, default=250, help="archetypes: deals per saved chunk")
    jobs.add_argument(
        "--lbr-configs",
        nargs="+",
        default=["fcpa/call", "fcpa/fc", "fcpa/full", "spec/call", "spec/fc", "spec/full"],
    )
    jobs.add_argument("--lbr-hands", type=int, default=2_000)
    jobs.add_argument(
        "--lbr-hand-seconds",
        type=float,
        default=45.0,
        help="lbr: seconds per hand on a spot GPU (LBR cannot resume: no block starts on an "
        "instance with less time left than its blocks take)",
    )
    jobs.add_argument("--soak-hands", type=int, default=100_000)
    jobs.add_argument("--block-hands", type=int, default=250, help="lbr: hands per block")
    jobs.add_argument(
        "--soak-block-hands", type=int, default=2_000, help="soak: hands per block (resumable)"
    )
    jobs.add_argument("--soak-opponents", nargs="+", default=["random", "equity", "always_call"])
    aws = ap.add_argument_group("workers")
    aws.add_argument("--local", action="store_true", help="also run blocks on the local GPU")
    aws.add_argument("--regions", nargs="+", default=["us-east-1", "us-east-2"])
    aws.add_argument("--types", nargs="+", default=["g5.xlarge", "g6.xlarge"])
    aws.add_argument("--slots", type=int, default=2, help="spot instances per region")
    aws.add_argument("--price-cap", type=float, default=0.70, help="spot max price, $/h")
    aws.add_argument("--budget", type=float, required=False, default=0.0, help="total spend, $")
    aws.add_argument("--max-hours", type=float, default=12.0, help="instance self-termination")
    aws.add_argument("--retry-min", type=float, default=10.0)
    aws.add_argument("--poll", type=float, default=60.0, help="seconds between polls")
    aws.add_argument("--bucket", default="pokerbot-402616727959")
    aws.add_argument("--bucket-region", default="us-east-2")
    aws.add_argument("--bundle-key", default="fleet/bundle.tar")
    return ap


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    args.jobs = list(dict.fromkeys(args.job))
    args.job = "-".join(args.jobs)  # the run's label: its directory, launch template, tags
    args.dir = args.dir or f"runs/fleet/{args.job}"
    return args


def run_blocks(args: argparse.Namespace) -> list[Block]:
    """Every job's blocks in one queue: the first block of every group (archetype, LBR
    configuration) before the second of any, and the jobs taking turns, so every
    result starts early. With several jobs, a block's name and directory start with its
    job."""
    many = len(args.jobs) > 1
    queues = []
    for job in args.jobs:
        blocks = JOBS[job](args, f"{args.dir}/blocks/{job}" if many else f"{args.dir}/blocks")
        for b in blocks:
            b.job = job
            if many:
                b.name = f"{job}-{b.name}"
        queues.append(sorted(blocks, key=lambda b: b.index))
    return [b for turn in itertools.zip_longest(*queues) for b in turn if b is not None]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sys.path.insert(0, str(ROOT))
    blocks = run_blocks(args)
    if args.dry_run:
        for b in blocks:
            where = b.type or "any worker"
            print(
                f"{b.name} ({where}): " + " && ".join("python " + " ".join(c) for c in b.commands)
            )
        print(f"{len(blocks)} blocks")
        return 0
    fleet = Fleet(args, blocks)
    if args.status:
        print(fleet.status())
        return 0
    if args.terminate:
        fleet.stop_everything()
        print(fleet.status())
        return 0
    if args.merge_only:
        fleet.merge()
        return 0
    if args.budget <= 0 and any(b.type != "local" for b in blocks):
        raise SystemExit("--budget (total spot spend, $) is required for AWS workers")
    fleet.loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

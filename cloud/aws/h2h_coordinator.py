#!/usr/bin/env python
r"""Head-to-head match of the Round 4 search against Round 3's on the local GPU plus spot
GPU instances, as one queue of blocks.

The match is ``--blocks`` blocks of ``--block-deals`` duplicate deals; block ``b`` plays
chunk seeds ``--seed + b * 100 + k``, so blocks are disjoint and each is a resumable
``scripts/match_search.py`` run (``runs/r4/h2h/block_<b>.json``).

* **Local worker.** Plays blocks on this machine's GPU, one at a time.
* **Spot workers.** Up to ``--slots`` spot instances (``--type``, default g5.xlarge) per
  region in ``--regions``. While blocks are left, a region with a free slot is retried
  every ``--retry-min`` minutes, zone by zone (spot capacity comes and goes). A new
  instance fetches the bundle (``cloud/aws/h2h_bundle.sh``) through a pre-signed URL,
  sets up (``remote_setup.sh``), then plays blocks. Its block's JSON is pulled every
  poll, so an interruption loses at most the chunk in progress: the block goes back to
  the queue and resumes from the last pulled chunk on the next free worker.
* **Guards.**
  * Every instance shuts itself down (and so terminates) ``--max-hours`` after boot.
  * Spend is tracked at each instance's launch-time spot price; no launches once it
    would pass 80% of ``--budget``, and everything is terminated at ``--budget``.
  * Idle instances are terminated as soon as the queue is empty.
  * Only instances this coordinator launched (recorded in the state file) are touched.
* **Restart.** State is saved every poll (``<--dir>/state.json``, default
  ``runs/r4/h2h``); rerunning adopts the recorded instances and blocks.

At the end the blocks are merged into ``runs/r4/h2h/summary.txt``: A's win rate (raw and
luck-adjusted, mbb/hand, 95% CI) and both searches' per-street statistics.

python runs/value_net/keepawake.py cloud/aws/h2h_coordinator.py --blocks 40
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GIT_USR_BIN = Path(r"C:\Program Files\Git\usr\bin")
SSH = str(GIT_USR_BIN / "ssh.exe")
SCP = str(GIT_USR_BIN / "scp.exe")
AWS = shutil.which("aws") or "aws"
PY = sys.executable
MAIN = Path(r"C:\Users\Connor\Documents\Poker-Bot\poker-bot")
BP_LOCAL = (MAIN / "runs" / "dcfr4_distilled_v2").as_posix()
KEYS = {"us-east-2": "pokerbot-pilot", "us-east-1": "pokerbot-us-east-1"}
SG_NAME = "pokerbot-ssh"
SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30", "-o", "LogLevel=ERROR",
]  # fmt: skip
DEAD_STATES = {"shutting-down", "terminated", "stopping", "stopped"}


def msys_path(p: Path) -> str:
    """``C:/x/y`` -> ``/c/x/y`` for Git Bash's scp (it would read ``C:`` as a host)."""
    s = Path(p).resolve().as_posix()
    return f"/{s[0].lower()}{s[2:]}" if len(s) > 1 and s[1] == ":" else s


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


@dataclass
class Worker:
    name: str
    kind: str  # local | aws
    region: str = ""
    instance_id: str = ""
    host: str = ""
    price: float = 0.0  # $/h at launch
    launched: float = 0.0
    ended: float = 0.0
    state: str = "ready"  # aws: booting, setup, ready, running, dead; local: ready, running, off
    block: int | None = None
    since: float = 0.0  # when the current state began
    unreachable: float = 0.0  # since when SSH has failed (0: reachable)
    fails: int = 0


class Coordinator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.dir = Path(args.dir) if Path(args.dir).is_absolute() else ROOT / args.dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.blocks: dict[int, str] = {b: "todo" for b in range(args.blocks)}
        self.workers: dict[str, Worker] = {}
        self.last_try: dict[str, float] = {}
        self.local_proc: subprocess.Popen | None = None
        self.stopped_launching = False
        self.n_launched = 0
        self._load()
        if "local" not in self.workers and not args.no_local:
            self.workers["local"] = Worker("local", "local")

    # -- state ------------------------------------------------------------------------

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        st = json.loads(self.state_path.read_text())
        self.blocks = {int(k): v for k, v in st["blocks"].items()}
        self.workers = {k: Worker(**w) for k, w in st["workers"].items()}
        self.n_launched = st.get("n_launched", 0)
        for w in self.workers.values():  # a local block in flight died with the old process
            if w.kind == "local" and w.state == "running":
                self.blocks[w.block] = "todo"
                w.state, w.block = "ready", None
        log(f"# resumed: {self.counts()}, workers {[w.name for w in self.alive()]}")

    def save(self) -> None:
        st = {
            "blocks": self.blocks,
            "workers": {k: asdict(w) for k, w in self.workers.items()},
            "n_launched": self.n_launched,
            "spent": self.spent(),
        }
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=1))
        tmp.replace(self.state_path)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.blocks.values():
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

    # -- blocks -----------------------------------------------------------------------

    def block_path(self, b: int) -> Path:
        return self.dir / f"block_{b:03d}.json"

    def seed(self, b: int) -> int:
        return self.args.seed + 100 * b

    def chunks_done(self, path: Path) -> int:
        try:
            return len(json.loads(path.read_text())["chunks"])
        except (OSError, ValueError, KeyError):
            return 0

    def n_chunks(self) -> int:
        return math.ceil(self.args.block_deals / self.args.chunk)

    def next_block(self) -> int | None:
        todo = [b for b, s in self.blocks.items() if s == "todo"]
        # blocks with partial results first: they are closest to done
        todo.sort(key=lambda b: (-self.chunks_done(self.block_path(b)), b))
        return todo[0] if todo else None

    # -- commands -----------------------------------------------------------------------

    def run(self, cmd: list[str], timeout: float = 120, check: bool = False) -> tuple[int, str]:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return 124, "timeout"
        out = (p.stdout + p.stderr).strip()
        if check and p.returncode:
            raise RuntimeError(f"{' '.join(cmd[:4])}...: {out[-500:]}")
        return p.returncode, out

    def aws(self, region: str, *args: str, timeout: float = 120) -> tuple[int, str]:
        return self.run([AWS, "--region", region, *args], timeout=timeout)

    def key_file(self, w: Worker) -> str:
        return f"/c/Users/Connor/.ssh/{KEYS[w.region]}.pem"

    def ssh(self, w: Worker, command: str, timeout: float = 60) -> tuple[int, str]:
        return self.run(
            [SSH, "-i", self.key_file(w), *SSH_OPTS, "-n", f"ubuntu@{w.host}", command],
            timeout=timeout,
        )

    def scp_get(self, w: Worker, remote: str, local: Path, timeout: float = 120) -> bool:
        cmd = [SCP, "-q", "-i", self.key_file(w), *SSH_OPTS, f"ubuntu@{w.host}:{remote}"]
        return self.run([*cmd, msys_path(local)], timeout=timeout)[0] == 0

    def scp_put(self, w: Worker, local: Path, remote: str, timeout: float = 120) -> bool:
        cmd = [SCP, "-q", "-i", self.key_file(w), *SSH_OPTS, msys_path(local)]
        return self.run([*cmd, f"ubuntu@{w.host}:{remote}"], timeout=timeout)[0] == 0

    # -- local worker -------------------------------------------------------------------

    def step_local(self) -> None:
        w = self.workers.get("local")
        if w is None or w.state == "off":
            return
        if w.state == "running" and self.local_proc is not None:
            rc = self.local_proc.poll()
            if rc is None:
                return
            b = w.block
            done = self.chunks_done(self.block_path(b)) >= self.n_chunks()
            if done:
                self.blocks[b] = "done"
                log(f"local: block {b} done")
            else:
                self.blocks[b] = "todo"
                w.fails += 1
                log(f"local: block {b} exited with {rc} unfinished (failure {w.fails})")
                if w.fails >= 3:
                    w.state = "off"
                    log("local: 3 failures, local worker off")
                    return
            w.state, w.block, self.local_proc = "ready", None, None
        if w.state == "ready":
            b = self.next_block()
            if b is None:
                return
            out = self.block_path(b)
            cmd = [
                PY, "runs/value_net/keepawake.py", "scripts/match_search.py",
                "--config", "configs/match_search_vn.yaml",
                "--a", f"search:neural:{BP_LOCAL}", "--b", f"search:neural:{BP_LOCAL}",
                "--b-search-config", "configs/search_value_net_r3.yaml",
                "--deals", str(self.args.block_deals), "--chunk", str(self.args.chunk),
                "--seed", str(self.seed(b)), "--resume", "--out", out.as_posix(),
            ]  # fmt: skip
            logf = open(self.dir / f"local_block_{b:03d}.log", "a")  # noqa: SIM115
            env = {**os.environ, "PYTHONPATH": "."}
            self.local_proc = subprocess.Popen(
                cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT, env=env
            )
            self.blocks[b] = "running"
            w.state, w.block, w.since = "running", b, time.time()
            log(f"local: block {b} started ({self.chunks_done(out)} chunks already)")

    # -- spot workers -------------------------------------------------------------------

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

    def spot_price(self, region: str, az: str) -> float:
        rc, out = self.aws(
            region, "ec2", "describe-spot-price-history", "--instance-types", self.args.type,
            "--product-descriptions", "Linux/UNIX", "--availability-zone", az,
            "--start-time", time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "--query", "SpotPriceHistory[0].SpotPrice", "--output", "text",
        )  # fmt: skip
        try:
            return float(out.strip())
        except ValueError:
            return self.args.price_cap

    def try_launch(self, region: str) -> None:
        self.last_try[region] = time.time()
        (ami, root_dev, snap_gb), sg = self.ami(region), self.security_group(region)
        if not ami or not sg or sg == "None":
            log(f"{region}: no AMI / security group ({ami!r}, {sg!r}); skipped")
            return
        user_data = f"#!/bin/bash\nshutdown -h +{int(self.args.max_hours * 60)}\n"
        for subnet, az in self.subnets(region):
            name = f"pokerbot-h2h-{self.n_launched + 1}"
            rc, out = self.aws(
                region, "ec2", "run-instances",
                "--image-id", ami, "--instance-type", self.args.type,
                "--key-name", KEYS[region], "--security-group-ids", sg, "--subnet-id", subnet,
                "--instance-market-options",
                "MarketType=spot,SpotOptions={SpotInstanceType=one-time,"
                f"InstanceInterruptionBehavior=terminate,MaxPrice={self.args.price_cap}}}",
                "--block-device-mappings",
                f"DeviceName={root_dev},Ebs={{VolumeSize={max(snap_gb, 120)},VolumeType=gp3,"
                "DeleteOnTermination=true}",
                "--instance-initiated-shutdown-behavior", "terminate", "--user-data", user_data,
                "--tag-specifications",
                f"ResourceType=instance,Tags=[{{Key=Name,Value={name}}},{{Key=project,Value=pokerbot}},"
                "{Key=purpose,Value=h2h}]",
                "--query", "Instances[0].InstanceId", "--output", "text",
            )  # fmt: skip
            if rc == 0 and out.startswith("i-"):
                self.n_launched += 1
                w = Worker(name, "aws", region, out.strip(), state="booting")
                w.launched = w.since = time.time()
                w.price = self.spot_price(region, az)
                self.workers[name] = w
                log(f"{region}: launched {w.instance_id} ({name}) in {az} at ${w.price:.3f}/h")
                return
            reason = "no capacity" if "InsufficientInstanceCapacity" in out else out[-300:]
            log(f"{region}/{az}: spot launch failed: {reason}")
            if "InsufficientInstanceCapacity" not in out:
                return  # quota, price or a config error: no point trying other zones now

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
            info = {}
            for line in out.splitlines():
                parts = line.split("\t")
                info[parts[0]] = parts
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
            self.pull(w)
            if self.blocks.get(w.block) == "running":
                self.blocks[w.block] = "todo"
            n = self.chunks_done(self.block_path(w.block))
            log(f"{w.name}: block {w.block} back to the queue at {n} chunks")
        w.state, w.block, w.ended = "dead", None, w.ended or time.time()
        log(f"{w.name}: {w.instance_id} ended: {why}")

    def terminate(self, w: Worker, why: str) -> None:
        """Shut the instance down over SSH (it terminates on shutdown, and this works
        without AWS credentials), then ask EC2 to terminate it too."""
        if w.host:
            self.ssh(w, "sudo shutdown -h now", timeout=30)
        rc, out = self.aws(w.region, "ec2", "terminate-instances", "--instance-ids", w.instance_id)
        if rc:
            log(f"{w.name}: terminate-instances failed ({out[-200:]}); relying on its shutdown")
        self.mark_dead(w, f"terminated by the coordinator: {why}")

    def presigned_url(self) -> str:
        rc, out = self.aws(
            self.args.bucket_region, "s3", "presign", f"s3://{self.args.bucket}/{self.args.bundle_key}",
            "--expires-in", "3600",
        )  # fmt: skip
        if rc:
            raise RuntimeError(f"presign failed: {out[-300:]}")
        return out.strip()

    def pull(self, w: Worker) -> bool:
        """Copy the worker's block JSON here if it has more chunks; True when done."""
        b = w.block
        if b is None or not w.host:
            return False
        tmp = self.dir / f"pull_{w.name}.json"
        got = self.scp_get(w, f"poker-bot/h2h/block_{b:03d}.json", tmp)
        if got and self.chunks_done(tmp) > self.chunks_done(self.block_path(b)):
            tmp.replace(self.block_path(b))
        return self.chunks_done(self.block_path(b)) >= self.n_chunks()

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
            rc, out = self.ssh(w, cmd)
            if rc == 0:
                w.state, w.since = "setup", now
                log(f"{w.name}: setting up")
            return
        if w.state == "setup":
            rc, _ = self.ssh(w, "test -f ~/setup.done")
            if rc == 0:
                w.state, w.since = "ready", now
                log(f"{w.name}: set up in {(now - w.launched) / 60:.0f} min since launch")
            elif now - w.since > 2400:
                _, tail = self.ssh(w, "tail -5 ~/setup.log")
                self.terminate(w, f"setup not done after 40 min: {tail[-300:]}")
            return
        if w.state == "running":
            done = self.pull(w)
            b = w.block
            if done:
                self.blocks[b] = "done"
                log(f"{w.name}: block {b} done")
                w.state, w.block, w.since = "ready", None, now
            else:
                # [s]: the pattern must not match the shell that runs pgrep
                alive_cmd = (
                    "pgrep -f '[s]cripts/match_search.py' >/dev/null && echo up || echo down"
                )
                rc, out = self.ssh(w, alive_cmd)
                if rc != 0:  # unreachable: an interruption the API has not shown yet?
                    w.unreachable = w.unreachable or now
                    if now - w.unreachable > 600:
                        self.terminate(w, "no SSH for 10 min")
                    return
                w.unreachable = 0.0
                if out.strip().endswith("down"):
                    self.pull(w)
                    _, tail = self.ssh(w, f"tail -3 ~/block_{b:03d}.log")
                    w.fails += 1
                    self.blocks[b] = "todo"
                    w.state, w.block = "ready", None
                    log(f"{w.name}: block {b} runner exited unfinished, failure {w.fails}")
                    log(f"{w.name}: {tail[-300:]}")
                    if w.fails >= 3:
                        self.terminate(w, "3 runner failures")
                    return
        if w.state == "ready":
            b = self.next_block()
            if b is None:
                self.terminate(w, "queue empty")
                return
            local = self.block_path(b)
            self.ssh(w, "mkdir -p ~/poker-bot/h2h")
            if local.exists() and not self.scp_put(w, local, f"poker-bot/h2h/block_{b:03d}.json"):
                return
            args = f"{b:03d} {self.seed(b)} {self.args.block_deals} {self.args.chunk}"
            rc, out = self.ssh(
                w,
                f"nohup bash ~/poker-bot/cloud/aws/remote_h2h_block.sh {args} "
                f"> ~/block_{b:03d}.log 2>&1 < /dev/null & echo started",
            )
            if rc == 0:
                self.blocks[b] = "running"
                w.state, w.block, w.since = "running", b, now
                log(f"{w.name}: block {b} started ({self.chunks_done(local)} chunks already)")

    def maybe_launch(self) -> None:
        if self.stopped_launching:
            return
        todo = sum(1 for s in self.blocks.values() if s == "todo")
        if todo == 0:
            return
        alive = self.alive()
        projected = self.spent() + sum(w.price for w in alive) * 1.0 + self.args.price_cap * 1.0
        if projected > 0.8 * self.args.budget:
            log(f"# budget: spent ${self.spent():.2f}, no more launches")
            self.stopped_launching = True
            return
        idle_soon = sum(1 for w in alive if w.state in ("booting", "setup", "ready"))
        if todo <= idle_soon:
            return
        for region in self.args.regions:
            n = sum(1 for w in alive if w.region == region)
            if n >= self.args.slots:
                continue
            if time.time() - self.last_try.get(region, 0) < self.args.retry_min * 60:
                continue
            self.try_launch(region)
            return  # one launch attempt per poll

    def finish(self) -> None:
        for w in self.alive():
            self.terminate(w, "match complete")
        chunks = []
        for b in sorted(self.blocks):
            chunks += json.loads(self.block_path(b).read_text())["chunks"]
        lines = [f"Round 4 (A) vs Round 3 (B), {len(chunks)} chunks, seeds from {self.args.seed}"]
        for key, label in (("a", "raw"), ("adjusted", "luck-adjusted")):
            xs = [x for c in chunks for x in c[key]]
            n = len(xs)
            m = sum(xs) / n
            sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
            h = 1.96 * sd / math.sqrt(n)
            lines.append(
                f"{label}: {5 * m:+.1f} mbb/hand, 95% CI [{5 * (m - h):+.1f}, {5 * (m + h):+.1f}],"
                f" {n} deals ({2 * n} hands)"
            )
        for key, who in (("search_stats", "A (Round 4)"), ("b_search_stats", "B (Round 3)")):
            stats = [s for c in chunks for s in c.get(key, [])]
            for st, name in ((1, "flop"), (2, "turn"), (3, "river")):
                xs = [s for s in stats if s.get("street") == st]
                if xs:
                    fb = sum(1 for s in xs if s.get("fallback"))
                    it = [s["iterations"] for s in xs if "iterations" in s]
                    lines.append(
                        f"{who} {name}: {len(xs)} decisions, {fb} fallbacks, "
                        f"iterations mean {sum(it) / max(1, len(it)):.0f}"
                    )
        lines.append(f"AWS spend about ${self.spent():.2f} over {self.n_launched} launches")
        (self.dir / "summary.txt").write_text("\n".join(lines) + "\n")
        for line in lines:
            log(f"# {line}")

    def loop(self) -> None:
        log(f"# h2h: {self.args.blocks} blocks x {self.args.block_deals} deals, {self.counts()}")
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
            except Exception as e:  # keep going: a failed AWS or SSH call is retried next poll
                log(f"! {e!r}")
            self.save()
            if all(s == "done" for s in self.blocks.values()):
                self.finish()
                self.save()
                return
            local = self.workers.get("local")
            if (
                not self.alive()
                and (local is None or local.state == "off")
                and self.stopped_launching
            ):
                log("# no workers left and no more launches; stopping (rerun to resume)")
                return
            time.sleep(self.args.poll)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--blocks", type=int, default=40)
    ap.add_argument("--block-deals", type=int, default=250)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--regions", nargs="+", default=["us-east-1", "us-east-2"])
    ap.add_argument("--slots", type=int, default=2, help="spot instances per region")
    ap.add_argument("--type", default="g5.xlarge")
    ap.add_argument("--price-cap", type=float, default=0.70, help="spot max price, $/h")
    ap.add_argument("--budget", type=float, default=20.0, help="total spot spend, $")
    ap.add_argument("--max-hours", type=float, default=8.0, help="instance self-termination")
    ap.add_argument("--retry-min", type=float, default=10.0)
    ap.add_argument("--poll", type=float, default=90.0, help="seconds between polls")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--bucket-region", default="us-east-2")
    ap.add_argument("--bundle-key", default="h2h/bundle.tar")
    ap.add_argument("--no-local", action="store_true", help="spot instances only")
    ap.add_argument("--dir", default="runs/r4/h2h", help="block results and state")
    args = ap.parse_args()
    Coordinator(args).loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

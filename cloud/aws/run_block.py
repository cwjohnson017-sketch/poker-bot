#!/usr/bin/env python
"""Run one fleet block (``cloud/aws/fleet.py``): the commands of ``<block>/block.json``.

Each command is the argument list of a ``python`` run from the repository root with
``PYTHONPATH=.``, run in order with this interpreter. When every command exits 0,
``<block>/.done`` is written; the first failure stops the block with its exit code.
The commands are expected to resume from what is already in the block directory, so
a block cut short continues where it stopped.

    python cloud/aws/run_block.py runs/fleet/<job>/blocks/<name>/block.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    spec_path = Path(args[0])
    spec = json.loads((spec_path if spec_path.is_absolute() else ROOT / spec_path).read_text())
    out = ROOT / spec["out"]
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": "."}
    for cmd in spec["commands"]:
        t0 = time.time()
        print(f"$ python {' '.join(cmd)}", flush=True)
        rc = subprocess.call([sys.executable, *cmd], cwd=ROOT, env=env)
        print(f"exit {rc} after {time.time() - t0:.0f} s", flush=True)
        if rc:
            return rc
    (out / ".done").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

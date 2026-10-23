#!/usr/bin/env python
r"""Preflop best response against the blueprint, with flop-root solves as leaves
(``pokerbot.search.preflop_eval``).

1. Walk the blueprint's preflop tree (button = seat 0): decisions, folds, preflop
   all-ins and flop starts, with both seats' blueprint reach.
2. Draw ``--flops`` flops (``--seed``; the same flops for every flop start). At each
   flop start and flop, solve the flop-root subgame as production's flop search does
   (``--config``: its tree and leaf nets) for ``--iterations`` DCFR iterations, from the
   blueprint's ranges, and keep both seats' root counterfactual values. Values go to
   ``--out-dir/flop_NN.pt`` after every solve; a rerun skips what is there.
3. Best responses to the blueprint's preflop for each seat, in mbb/hand:
   in-sample (biased up) and cross-fitted over two halves of the flops (biased down).
   Results in ``--out-dir/preflop_eval.json`` and ``.md``.

``--only-eval`` skips step 2 and evaluates whatever values are on disk (flop starts
with no values are reported and the run stops). ``--max-flop-starts`` / ``--flop-starts``
limit step 2 (e.g. a timing pilot).

python scripts/eval_preflop.py --blueprint runs/dcfr4_distilled_v2 --flops 100 \
    --iterations 100 --out-dir runs/preflop/pf100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from pokerbot.engine_select import get_engine  # noqa: E402
from pokerbot.search.agent import SearchAgent  # noqa: E402
from pokerbot.search.blueprint import BLUEPRINTS, make_blueprint  # noqa: E402
from pokerbot.search.config import search_config  # noqa: E402
from pokerbot.search.preflop_eval import (  # noqa: E402
    FlopValues,
    evaluate,
    preflop_allin_matrix,
    preflop_tree,
    sample_flops,
    solve_flop_root,
    suit_perms,
)


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def card_str(c: int) -> str:
    return "23456789TJQKA"[c // 4] + "cdhs"[c % 4]


def freqs(labels: list[str], xs: list[float]) -> str:
    return ", ".join(f"{a} {x:.2f}" for a, x in zip(labels, xs, strict=True) if x >= 0.005)


def render(res: dict, nodes: list, args: argparse.Namespace) -> str:
    seat_name = ("button (seat 0)", "big blind (seat 1)")
    out = [
        "# Preflop best response against the blueprint",
        "",
        f"Blueprint `{args.blueprint}`, leaves: flop-root solves with `{args.config}`, "
        f"{args.iterations} iterations, {res['flops']} flops x 24 suit permutations.",
        "",
        "| mbb/hand | button | big blind | mean (exploitability) |",
        "|---|---:|---:|---:|",
    ]
    s = res["seats"]
    out.append(
        f"| blueprint value | {s[0]['blueprint']:+.1f} | {s[1]['blueprint']:+.1f} | "
        f"zero-sum check {res['zero_sum_check']:+.2f} |"
    )
    for key, label in (
        ("br_in_sample", "best response, in-sample (high)"),
        ("br_cross_fit", "best response, cross-fitted (low)"),
    ):
        ex = res[
            "exploitability_in_sample" if key == "br_in_sample" else "exploitability_cross_fit"
        ]
        out.append(f"| {label} | {s[0][key]:+.1f} | {s[1][key]:+.1f} | **{ex:.1f}** |")
    out.append(
        f"| blueprint on the halves | {s[0]['blueprint_halves']:+.1f} | "
        f"{s[1]['blueprint_halves']:+.1f} | |"
    )
    for p in (0, 1):
        out += [
            "",
            f"## Largest one-step gains, {seat_name[p]}",
            "",
            "| line | gain | blueprint frequencies | best-action frequencies |",
            "|---|---:|---|---|",
        ]
        for n in s[p]["nodes"][:10]:
            bf = freqs(n["labels"], n["blueprint_freq"])
            rf = freqs(n["labels"], n["best_freq"])
            out.append(f"| `{n['line']}` | {n['gain']:.1f} | {bf} | {rf} |")
    out += [
        "",
        "## Flop starts",
        "",
        "| line | blueprint pair mass | solves | mean s | flop exploitability (pot fraction) |",
        "|---|---:|---:|---:|---:|",
    ]
    for f in res["flop_starts"]:
        ex = f.get("exploitability_pot")
        out.append(
            f"| `{f['line']}` | {f['mass']:.4f} | {f['solves']} | {f['seconds']:.1f} | "
            f"{'' if ex is None else f'{ex:.4f}'} |"
        )
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--blueprint", required=True)
    ap.add_argument("--config", default="configs/search_value_net.yaml")
    ap.add_argument("--flops", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument(
        "--exploitability-every",
        type=int,
        default=10,
        help="score the flop solve's own exploitability every N-th flop (0: never)",
    )
    ap.add_argument("--allin-runouts", type=int, default=2000)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only-eval", action="store_true")
    ap.add_argument(
        "--max-flop-starts",
        type=int,
        default=None,
        help="solve only the N flop starts with the most blueprint mass",
    )
    ap.add_argument(
        "--flop-starts", type=int, nargs="*", default=None, help="solve only these flop-start ids"
    )
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = (
        args.blueprint
        if args.blueprint.partition(":")[0] in BLUEPRINTS
        else f"neural:{args.blueprint}"
    )
    dev = torch.device(args.device)
    engine = get_engine()
    gcfg = engine.GameConfig(num_players=2, stacks=[10000, 10000], small_blind=50, big_blind=100)
    bp = make_blueprint(spec, device=str(dev))
    nodes = preflop_tree(bp, gcfg, engine, dev)
    flops_at = [i for i, n in enumerate(nodes) if n.kind == "flop"]
    mass = {
        nodes[i].flop_id: float((nodes[i].reach[0][:, None] * nodes[i].reach[1][None, :]).sum())
        / 1326**2
        for i in flops_at
    }
    log(
        f"preflop tree: {len(nodes)} nodes, {sum(n.kind == 'dec' for n in nodes)} decisions, "
        f"{len(flops_at)} flop starts"
    )
    flops = sample_flops(args.flops, args.seed)
    meta = {k: getattr(args, k) for k in ("blueprint", "config", "flops", "seed", "iterations")}
    (out / "args.json").write_text(json.dumps(meta, indent=1))

    if not args.only_eval:
        agent = SearchAgent(
            bp, search_config(args.config, device=str(dev), fallback_on_error=False)
        )
        agent._check_leaf_nets()
        order = sorted(flops_at, key=lambda i: -mass[nodes[i].flop_id])
        if args.flop_starts is not None:
            order = [i for i in order if nodes[i].flop_id in set(args.flop_starts)]
        if args.max_flop_starts is not None:
            order = order[: args.max_flop_starts]
        for i in order:
            n = nodes[i]
            path = out / f"flop_{n.flop_id:02d}.pt"
            data = torch.load(path) if path.exists() else {"line": n.line(), "flops": {}}
            todo = [f for f in flops if f not in data["flops"]]
            if not todo:
                continue
            log(
                f"flop start {n.flop_id} `{n.line()}` (mass {mass[n.flop_id]:.4f}): "
                f"{len(todo)} flops"
            )
            for f in todo:
                k = flops.index(f)
                ex = args.exploitability_every and k % args.exploitability_every == 0
                r = solve_flop_root(
                    agent, gcfg, engine, n.history, f, n.reach, args.iterations, bool(ex)
                )
                r["values"] = r["values"].to(torch.float32)
                data["flops"][f] = r
                torch.save(data, path)
                msg = (
                    f"  {''.join(card_str(c) for c in f)}: {r['nodes']} nodes, {r['seconds']:.1f} s"
                )
                if ex:
                    msg += f", exploitability {r['exploitability']:.1f} chips (pot {r['pot']})"
                log(msg)
            torch.cuda.empty_cache()

    # -- evaluation ------------------------------------------------------------------
    perms = suit_perms()
    fv: dict[int, FlopValues] = {}
    starts = []
    missing = []
    for i in flops_at:
        n = nodes[i]
        path = out / f"flop_{n.flop_id:02d}.pt"
        data = torch.load(path) if path.exists() else {"flops": {}}
        got = [f for f in flops if f in data["flops"]]
        if len(got) < len(flops):
            missing.append((n.flop_id, n.line(), len(got)))
        v = FlopValues(perms)
        for f in got:
            v.add(flops.index(f) % 2, f, data["flops"][f]["values"])
        fv[n.flop_id] = v
        rs = [data["flops"][f] for f in got]
        exs = [r["exploitability"] / r["pot"] for r in rs if "exploitability" in r]
        starts.append(
            {
                "flop_id": n.flop_id,
                "line": n.line(),
                "mass": mass[n.flop_id],
                "solves": len(rs),
                "seconds": sum(r["seconds"] for r in rs) / max(len(rs), 1),
                "exploitability_pot": sum(exs) / len(exs) if exs else None,
            }
        )
    if missing:
        for fid, line, k in missing:
            log(f"flop start {fid} `{line}`: {k} of {len(flops)} flops solved")
        log("not every flop start has all its flops; no evaluation")
        return 1
    E = preflop_allin_matrix(args.allin_runouts)
    res = evaluate(nodes, fv, E, int(gcfg.big_blind))
    res["flops"] = len(flops)
    res["flop_starts"] = sorted(starts, key=lambda s: -s["mass"])
    res["args"] = meta
    (out / "preflop_eval.json").write_text(json.dumps(res, indent=1))
    md = render(res, nodes, args)
    (out / "preflop_eval.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Read this machine's AI traces: what each job asked the model, what came back, what it cost and how long it took.

    .venv\\Scripts\\python scripts\\traces.py                     # the last 20 traces
    .venv\\Scripts\\python scripts\\traces.py --summary           # totals for the last 7 days
    .venv\\Scripts\\python scripts\\traces.py --summary --days 30
    .venv\\Scripts\\python scripts\\traces.py --id 3f2a9c1b7d4e5f60  # one trace and every model call in it
    .venv\\Scripts\\python scripts\\traces.py --json              # the raw records, for another tool

Read-only: the traces are this profile's `ai_traces.jsonl`. Without --profile, the profile last used in the app.
Nothing leaves the machine."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import observability, profiles  # noqa: E402


def _tok(n) -> str:
    n = int(n or 0)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _when(stamp: str) -> str:
    return (stamp or "").replace("T", " ")[:19]


def _cost(usd) -> str:
    return "-" if usd is None else f"${usd:.4f}"


def _ms(ms) -> str:
    return "-" if ms is None else f"{ms:g} ms"


def show(traces: list[dict]) -> None:
    if not traces:
        print("No traces yet: they are written as the app runs a model.")
        return
    print(f"{'when':19}  {'name':18} {'status':8} {'runs':>4} {'tokens':>13} {'cost':>9} {'took':>7}  id")
    for t in traces:
        usage = t.get("usage") or {}
        runs = t.get("generation_count") or len(t.get("generations") or [])
        ms = t.get("duration_ms")
        print(f"{_when(t.get('started_at')):19}  {str(t.get('name'))[:18]:18} {str(t.get('status')):8} {runs:>4} "
              f"{_tok(usage.get('input')) + '->' + _tok(usage.get('output')):>13} "
              f"{_cost(t.get('cost_usd')):>9} {(f'{ms / 1000:.1f}s' if ms else '-'):>7}  {t.get('id')}")


def detail(t: dict) -> None:
    usage = t.get("usage") or {}
    print(f"trace {t.get('id')}  {t.get('name')}  [{t.get('status')}]  {_when(t.get('started_at'))}")
    print(f"  session {t.get('session_id') or '-'}  profile {t.get('user_id') or '-'}  "
          f"{_tok(usage.get('input'))} in / {_tok(usage.get('output'))} out  {_cost(t.get('cost_usd'))}")
    if t.get("status_message"):
        print(f"  error: {t['status_message']}")
    for g in t.get("generations") or []:
        u = g.get("usage") or {}
        print(f"  - {_when(g.get('started_at'))[11:]}  {g.get('name')}  {g.get('model') or g.get('engine')}  "
              f"{g.get('latency_ms', 0) / 1000:.1f}s  {_tok(u.get('input'))}->{_tok(u.get('output'))}  "
              f"{_cost(g.get('cost_usd'))}  {g.get('level')}{(' ' + g['status_message']) if g.get('status_message') else ''}")
    for s in t.get("scores") or []:
        print(f"  score {s.get('name')} = {s.get('value')}{('  ' + s['comment']) if s.get('comment') else ''}")


def show_summary(days: int) -> None:
    s = observability.summary(days)
    print(f"Last {s['days']} days  ({s['path']})")
    print(f"  traces {s['traces']}  model runs {s['generations']}  failed traces {s['trace_errors']}  "
          f"failed runs {s['generation_errors']}  error rate {s['error_rate']:.0%}")
    tokens = s["tokens"]
    print(f"  tokens {_tok(tokens['input'])} in / {_tok(tokens['output'])} out / {_tok(tokens['cache_read'])} cached  "
          f"cost {_cost(s['cost_usd'])}" + (f"  ({s['unpriced_runs']} unpriced runs)" if s["unpriced_runs"] else ""))
    lat = s["latency_ms"]
    print(f"  latency p50 {_ms(lat['p50'])}  p95 {_ms(lat['p95'])}  max {_ms(lat['max'])}")
    if s["by_model"]:
        print("\n  by model")
        for model, row in s["by_model"].items():
            print(f"    {model:24} {row['runs']:>4} runs  {row['errors']:>3} failed  {_tok(row['tokens']):>8} tokens  "
                  f"{_cost(row['cost_usd']):>9}  {_ms(row['avg_ms'])} avg")
    if s["by_name"]:
        print("\n  by operation")
        for name, row in s["by_name"].items():
            print(f"    {name:24} {row['runs']:>4} runs  {row['errors']:>3} failed  {_tok(row['tokens']):>8} tokens  "
                  f"{_cost(row['cost_usd']):>9}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Read this machine's AI traces (never leaves the machine).")
    ap.add_argument("--profile", help="profile id (default: the one last used in the app)")
    ap.add_argument("--limit", type=int, default=20, help="traces to show (default 20)")
    ap.add_argument("--days", type=int, default=7, help="window for --summary (default 7)")
    ap.add_argument("--id", help="one trace by id, with every model call in it")
    ap.add_argument("--summary", action="store_true", help="totals instead of the trace list")
    ap.add_argument("--json", action="store_true", help="print the raw records as JSON")
    args = ap.parse_args()

    profiles.activate(args.profile or profiles.last_used())
    if args.id:
        found = observability.get(args.id)
        if not found:
            sys.exit(f"No trace {args.id} in {observability.trace_path()}")
        if args.json:
            print(json.dumps(found, indent=2, ensure_ascii=False))
        else:
            detail(found)
    elif args.summary:
        if args.json:
            print(json.dumps(observability.summary(args.days), indent=2))
        else:
            show_summary(args.days)
    else:
        traces = observability.recent(args.limit, days=None)
        if args.json:
            print(json.dumps(traces, indent=2, ensure_ascii=False))
        else:
            show(traces)


if __name__ == "__main__":
    main()

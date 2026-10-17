"""Run the experiment suites and write the comparison tables and charts.

    python scripts/run_experiments.py --suite prompt --dry-run       # what would run, and the cost estimate
    python scripts/run_experiments.py --suite prompt                 # personas | prompt | config | all
    python scripts/run_experiments.py --suite config --factor temperature --factor model
    python scripts/run_experiments.py --suite personas --tasks T05,T07 --no-judge
    python scripts/run_experiments.py --suite prompt --report-only   # rebuild tables/charts from saved runs
    python scripts/run_experiments.py --suite all --rejudge          # re-score saved answers with the judge

Results: experiments/results/<suite>_runs.jsonl, _traces.jsonl, _summary.csv, _summary.md and
experiments/charts/<suite>_<factor>.png.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from douroub.core import has_api_key  # noqa: E402
from douroub.experiments import (SUITES, estimate_cost, load_runs, plan, rejudge, run_suite, summarize,  # noqa: E402
                                 write_report)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=[*SUITES, "all"])
    ap.add_argument("--factor", action="append", help="only arms that vary this factor (repeatable)")
    ap.add_argument("--tasks", help="comma-separated task IDs, e.g. T01,T05")
    ap.add_argument("--repeats", type=int, help="override the number of repeats of every arm")
    ap.add_argument("--workers", type=int, default=4, help="runs in parallel (lower it on rate limits)")
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM judge scores")
    ap.add_argument("--llm-router", action="store_true", help="route with the LLM router instead of the rules")
    ap.add_argument("--dry-run", action="store_true", help="list the runs and the cost estimate, call nothing")
    ap.add_argument("--report-only", action="store_true", help="rebuild tables and charts from saved results")
    ap.add_argument("--rejudge", action="store_true", help="score the saved answers again with the current judge")
    args = ap.parse_args()
    suites = list(SUITES) if args.suite == "all" else [args.suite]
    tasks = args.tasks.split(",") if args.tasks else None

    for suite in suites:
        if args.rejudge:
            if not has_api_key():
                sys.exit("OPENAI_API_KEY is not set: add it to .env first.")
            rejudge(suite, workers=args.workers)
            continue
        if args.report_only:
            df = load_runs(suite)
            if df.empty:
                print(f"{suite}: no saved results yet")
                continue
            write_report(summarize(df, suite), suite)
            print(f"{suite}: tables and charts rebuilt from {len(df)} runs")
            continue
        runs = plan(suite, args.factor, tasks, args.repeats)
        if args.dry_run:
            per_arm = Counter(arm.label for arm, _, _ in runs)
            print(f"\n{suite}: {len(runs)} runs, at most about ${estimate_cost(runs, not args.no_judge):.2f}")
            for label, n in per_arm.items():
                print(f"  {n:>3}  {label}")
            continue
        if not has_api_key():
            sys.exit("OPENAI_API_KEY is not set: add it to .env first.")
        run_suite(suite, args.factor, tasks, args.repeats, use_judge=not args.no_judge, workers=args.workers,
                  use_llm_router=args.llm_router)


if __name__ == "__main__":
    main()

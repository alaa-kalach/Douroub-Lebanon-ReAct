"""Experiment suites: persona, prompt, and configuration variants run on the test set, logged and compared.

An Arm is one setup (persona choice + prompt variant + config changes) run on a list of tasks, possibly several
times. A suite is a list of arms grouped by the factor they vary. Results go to experiments/results/:

    <suite>_runs.jsonl     one row per run: checks, metrics, cost, judge scores, the final answer
    <suite>_traces.jsonl   the full ReAct trace of every run (raw model replies included)
    <suite>_summary.csv    one row per arm, the numbers behind the tables and charts
    <suite>_summary.md     one comparison table per factor
and charts to experiments/charts/<suite>_<factor>.png.

Experiment runs never touch the real logs: inquiries and feedback go to experiments/sandbox_logs/, and no
notifications are printed.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import pandas as pd
from pathlib import Path

from . import tools
from .agent import build_graph, run_turn
from .core import EXPERIMENTS_DIR, PRICES, QuotaExhausted
from .evaluation import (CORE_TASKS, TASKS, TASKS_BY_ID, check_run, diversity, judge, numbers_in,
                         successful_tools, text_metrics)
from .personas import PERSONAS
from .react import PromptVariant

RESULTS_DIR = EXPERIMENTS_DIR / "results"
CHARTS_DIR = EXPERIMENTS_DIR / "charts"
SANDBOX_DIR = EXPERIMENTS_DIR / "sandbox_logs"
ALL_TASKS = tuple(t.id for t in TASKS)


@dataclass(frozen=True)
class Arm:
    factor: str                         # what this arm varies ("temperature", "examples"...); "baseline" is shared
    label: str
    persona: str = "auto"
    variant: PromptVariant = PromptVariant()
    override: dict = field(default_factory=dict)    # LLMConfig changes applied to whichever persona answers
    tasks: tuple = CORE_TASKS
    repeats: int = 1


# --------------------------------------------------------------------------- the suites
# The personas, prompt, and config suites were run with the first version of the rules, so they stay pinned to it
# and can be re-run as they were; the iteration suite compares that version with the current one.
V1 = PromptVariant(rules=1)


def _personas_suite() -> list:
    fidelity_tasks = ("T01", "T05", "T07", "T15")
    return ([Arm("routing", "Auto (4 personas)", variant=V1, tasks=ALL_TASKS),
             Arm("routing", "Host only (1 persona)", persona="host", variant=V1, tasks=ALL_TASKS)]
            + [Arm("fidelity", f"{p.name} ({key})", persona=key, variant=V1, tasks=fidelity_tasks) for key, p in PERSONAS.items()])


def _prompt_suite() -> list:
    return [Arm("baseline", "baseline: CoT, one-shot, system", variant=V1),
            Arm("chain of thought", "no CoT (actions only)", variant=PromptVariant(thought=False, rules=1)),
            Arm("examples", "zero-shot", variant=PromptVariant(shots="zero", rules=1)),
            Arm("examples", "few-shot (4 examples)", variant=PromptVariant(shots="few", rules=1)),
            Arm("placement", "instructions in the message", variant=PromptVariant(placement="message", rules=1)),
            Arm("instructions", "no instructions block", variant=PromptVariant(instructions=False, rules=1))]


def _config_suite() -> list:
    story = ("T05", "T06")
    arms = [Arm("temperature", f"temperature {t}", variant=V1, override={"temperature": t}, tasks=("T01", "T05", "T06"), repeats=3)
            for t in (0.0, 0.6, 1.1, 1.6)]
    arms += [Arm("top_p", f"top_p {p}", variant=V1, persona="storyteller", override={"top_p": p}, tasks=story, repeats=2)
             for p in (0.3, 0.7, 1.0)]
    arms += [Arm("penalties", label, variant=V1, persona="storyteller", override=o, tasks=story, repeats=2) for label, o in (
        ("no penalties", {"presence_penalty": 0.0, "frequency_penalty": 0.0}),
        ("Layla's penalties (0.6 / 0.3)", {}),
        ("high penalties (1.5 / 1.0)", {"presence_penalty": 1.5, "frequency_penalty": 1.0}))]
    arms += [Arm("max tokens", f"max_completion_tokens {n}", variant=V1, override={"max_completion_tokens": n},
                 tasks=("T01", "T03", "T05")) for n in (150, 400, 900)]
    models = (("gpt-5.4-nano", {}), ("gpt-5.4-mini", {}), ("gpt-6-luna", {}),
              ("gpt-4.1-mini", {}), ("gpt-4o-mini", {}), ("gpt-6.1-sol", {"reasoning_effort": "low"}))
    arms += [Arm("model", m, variant=V1, override={"model": m} | extra, tasks=("T01", "T02", "T05", "T07", "T10", "T13"))
             for m, extra in models]
    arms += [Arm("reasoning effort", label, variant=V1, override=o, tasks=("T01", "T03", "T04", "T20")) for label, o in (
        ("effort none (sampling sent)", {"model": "gpt-5.4-mini", "reasoning_effort": "none"}),
        ("effort low (sampling dropped)", {"model": "gpt-5.4-mini", "reasoning_effort": "low"}))]
    arms += [Arm("stop sequence", label, variant=V1, override=o, tasks=("T01", "T03", "T04", "T07")) for label, o in (
        ("gpt-4.1-mini, parser only", {"model": "gpt-4.1-mini"}),
        ("gpt-4.1-mini + stop 'Observation:'", {"model": "gpt-4.1-mini", "stop": ("Observation:",)}))]
    return arms


def _iteration_suite() -> list:
    """Instructions v1 vs v2: the two added rules, tested on the tasks whose failures motivated them, plus two
    lead/feedback tasks to check that nothing else got worse."""
    tasks = ("T07", "T19", "T09", "T16")
    return [Arm("rules version", "instructions v1 (first version)", variant=PromptVariant(rules=1), tasks=tasks, repeats=3),
            Arm("rules version", "instructions v2 (+2 rules)", variant=PromptVariant(rules=2), tasks=tasks, repeats=3)]


SUITES = {"personas": _personas_suite, "prompt": _prompt_suite, "config": _config_suite,
          "iteration": _iteration_suite}


def plan(suite: str, factors=None, tasks=None, repeats=None) -> list:
    """The (arm, task id, repeat) runs of a suite, optionally narrowed."""
    runs = []
    for arm in SUITES[suite]():
        if factors and arm.factor not in factors and arm.factor != "baseline":
            continue
        for task_id in (t for t in arm.tasks if not tasks or t in tasks):
            for r in range(repeats or arm.repeats):
                runs.append((arm, task_id, r + 1))
    return runs


def estimate_cost(runs: list, with_judge: bool = True) -> float:
    """Rough upper estimate: 3 model calls of ~4,500 input and ~200 output tokens per run."""
    total = 0.0
    for arm, _, _ in runs:
        price_in, _, price_out = PRICES.get(arm.override.get("model", "gpt-5.4-mini"), PRICES["gpt-5.4-mini"])
        total += 3 * (4500 * price_in + 200 * price_out) / 1e6
        if with_judge:
            total += (3500 * PRICES["gpt-5.4-mini"][0] + 80 * PRICES["gpt-5.4-mini"][2]) / 1e6
    return total


# --------------------------------------------------------------------------- one run
class _PerThreadDict(threading.local):
    """tools._RECENT is a process-wide de-duplication memory; runs in parallel threads must not share it."""

    def __init__(self):
        self.data = {}

    def __contains__(self, key):
        return key in self.data

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value


_RECENT = _PerThreadDict()


def _sandbox() -> None:
    tools.LOG_DIR, tools.VERBOSE, tools._RECENT = SANDBOX_DIR, False, _RECENT


def run_one(arm: Arm, task_id: str, repeat: int, suite: str, run_id: str, use_judge: bool = True,
            use_llm_router: bool = False, client=None) -> tuple:
    """Run one task with one arm. Returns (result row, trace record)."""
    task = TASKS_BY_ID[task_id]
    _RECENT.data = {}
    agent, thread = build_graph(), uuid.uuid4().hex
    kw = dict(thread_id=thread, persona=arm.persona, variant=arm.variant, config_override=dict(arm.override),
              use_llm_router=use_llm_router, client=client)
    start = time.perf_counter()
    for earlier in task.setup:
        run_turn(agent, earlier, **kw)
    state = run_turn(agent, task.message, **kw)
    wall = time.perf_counter() - start
    checks = check_run(task, state)
    steps = [e for e in state["trace"] if e.get("type") in ("action", "final", "invalid", "grounding")]
    usage = state["usage"]
    row = {
        "run_id": run_id, "suite": suite, "factor": arm.factor, "arm": arm.label, "task": task.id,
        "category": task.category, "repeat": repeat, "persona": state["active_persona"],
        "expected_persona": task.expected_persona, "persona_match": state["active_persona"] == task.expected_persona,
        "prompt": arm.variant.short(), "override": json.dumps(arm.override, default=list),
        "model": next((e.get("model") for e in steps if e.get("model")), ""),
        "success": all(checks.values()), "score": round(sum(checks.values()) / len(checks), 3),
        "failed_checks": [k for k, v in checks.items() if not v],
        "model_calls": usage["calls"], "tool_calls": sum(e.get("type") == "observation" for e in state["trace"]),
        "tools_ok": [t for t, _ in successful_tools(state)],
        "format_errors": sum(e.get("type") == "invalid" for e in steps),
        "format_slips": sum(bool(e.get("format_note")) for e in steps),
        "grounding_checks": sum(e.get("type") == "grounding" for e in steps),
        "invented_observations": sum(bool(e.get("invented_observation")) for e in steps),
        "length_cutoffs": sum(e.get("finish_reason") == "length" for e in steps),
        "dropped_settings": sorted({s for e in steps for s in e.get("dropped_settings") or []}),
        "stop_reason": state["stop_reason"], "input_tokens": usage["input_tokens"],
        "cached_tokens": usage["cached_tokens"], "output_tokens": usage["output_tokens"],
        "cost_usd": usage["cost_usd"], "latency_s": round(wall, 2), "final": state["final"],
        **text_metrics(state["final"]),
    }
    if use_judge:
        try:
            row |= judge(task.message, state["final"], state["active_persona"], state, client=client)
        except QuotaExhausted:
            raise
        except Exception as exc:  # a judge failure must not lose the run
            row["judge_comment"] = f"judge failed: {type(exc).__name__}"
    trace = {"run_id": run_id, "suite": suite, "arm": arm.label, "task": task.id, "repeat": repeat,
             "message": task.message, "trace": state["trace"]}
    return row, trace


# --------------------------------------------------------------------------- a suite
def run_suite(suite: str, factors=None, tasks=None, repeats=None, use_judge=True, workers=4,
              use_llm_router=False, client=None, log=print) -> pd.DataFrame:
    """Run every planned run of a suite, write the results, and return the per-run table."""
    _sandbox()
    runs = plan(suite, factors, tasks, repeats)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows, traces, errors = [], [], 0
    log(f"{suite}: {len(runs)} runs, about ${estimate_cost(runs, use_judge):.2f} at most, {workers} in parallel")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(run_one, arm, t, r, suite, run_id, use_judge, use_llm_router, client): (arm, t, r)
                   for arm, t, r in runs}
        for i, fut in enumerate(as_completed(futures), 1):
            arm, task_id, repeat = futures[fut]
            try:
                row, trace = fut.result()
            except QuotaExhausted as exc:
                log(f"Stopping: the API quota is used up ({exc})")
                for f in futures:
                    f.cancel()
                break
            except Exception as exc:
                errors += 1
                row = {"run_id": run_id, "suite": suite, "factor": arm.factor, "arm": arm.label, "task": task_id,
                       "repeat": repeat, "success": False, "score": 0.0, "error": f"{type(exc).__name__}: {exc}"[:300]}
                trace = None
            rows.append(row)
            if trace:
                traces.append(trace)
            mark = "ok " if row.get("success") else ("ERR" if row.get("error") else "-- ")
            log(f"[{i:>3}/{len(runs)}] {mark} {arm.label[:34]:<34} {task_id} #{repeat}"
                + (f"  failed: {', '.join(row.get('failed_checks', []))[:70]}" if row.get("failed_checks") else "")
                + (f"  {row['error'][:90]}" if row.get("error") else ""))
    _write_jsonl(RESULTS_DIR / f"{suite}_runs.jsonl", rows)
    _write_jsonl(RESULTS_DIR / f"{suite}_traces.jsonl", traces)
    df = pd.DataFrame(rows)
    summary = summarize(df, suite)
    write_report(summary, suite)
    log(f"Done: {len(rows)} runs ({errors} errors), ${df.get('cost_usd', pd.Series(dtype=float)).sum():.3f} agent "
        f"cost. Results in {RESULTS_DIR}")
    return df


def _write_jsonl(path, records) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def rejudge(suite: str, workers: int = 4, client=None, log=print) -> pd.DataFrame:
    """Score the saved answers of a suite again with the current judge (the agent is not re-run)."""
    runs_path, traces_path = RESULTS_DIR / f"{suite}_runs.jsonl", RESULTS_DIR / f"{suite}_traces.jsonl"
    rows = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    traces = {}
    for line in traces_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            t = json.loads(line)
            traces[(t["run_id"], t["arm"], t["task"], t["repeat"])] = t
    todo = [r for r in rows if r.get("final") is not None and not r.get("error")]
    log(f"{suite}: re-judging {len(todo)} saved answers")

    def score(row):
        trace = traces.get((row["run_id"], row["arm"], row["task"], row["repeat"]), {})
        return row, judge(TASKS_BY_ID[row["task"]].message, row["final"], row["persona"], trace, client=client)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, fut in enumerate(as_completed([pool.submit(score, r) for r in todo]), 1):
            try:
                row, scores = fut.result()
            except QuotaExhausted as exc:
                log(f"Stopping: the API quota is used up ({exc})")
                break
            except Exception as exc:
                log(f"judge failed: {type(exc).__name__}: {exc}"[:120])
                continue
            row.update(scores)
            if i % 20 == 0:
                log(f"  {i}/{len(todo)}")
    _write_jsonl(runs_path, rows)
    df = pd.DataFrame(rows)
    write_report(summarize(df, suite), suite)
    log(f"{suite}: scores, tables and charts updated")
    return df


def load_runs(suite: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{suite}_runs.jsonl"
    return pd.read_json(path, lines=True) if path.exists() else pd.DataFrame()


# --------------------------------------------------------------------------- comparing arms
METRICS = {  # column -> (label, how to aggregate)
    "success": ("success rate", "mean"), "score": ("checks passed", "mean"),
    "persona_match": ("persona as expected", "mean"),
    "persona_fidelity": ("judge: persona", "mean"), "helpfulness": ("judge: helpful", "mean"),
    "grounding": ("judge: grounded", "mean"),
    "model_calls": ("model calls", "mean"), "format_errors": ("format errors", "sum"),
    "format_slips": ("format slips", "sum"), "grounding_checks": ("grounding checks", "sum"),
    "invented_observations": ("invented observations", "sum"), "length_cutoffs": ("cut off by max tokens", "sum"),
    "answer_words": ("words", "mean"), "lexical_variety": ("lexical variety", "mean"),
    "repeated_trigrams": ("repeated trigrams", "mean"),
    "cost_usd": ("cost per run ($)", "mean"), "latency_s": ("latency (s)", "mean"),
}


def summarize(df: pd.DataFrame, suite: str) -> pd.DataFrame:
    """One row per arm (in suite order) with the metrics above, plus consistency across repeats."""
    if df.empty:
        return df
    order = {a.label: i for i, a in enumerate(SUITES[suite]())}
    agg = {c: how for c, (_, how) in METRICS.items() if c in df}
    df = df.copy()
    for c in agg:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    out = df.groupby(["factor", "arm"], sort=False).agg({**agg, "task": "count"}).rename(columns={"task": "runs"})
    if "error" in df:
        out["errors"] = df.groupby(["factor", "arm"], sort=False)["error"].apply(lambda s: s.notna().sum())
    if "repeat" in df and df["repeat"].max() > 1 and "final" in df:
        groups = df.dropna(subset=["final"]).groupby(["factor", "arm", "task"], sort=False)
        div = groups["final"].apply(lambda s: diversity(list(s)) if len(s) > 1 else None)
        same = groups["final"].apply(lambda s: len({numbers_in(x) for x in s}) == 1 if len(s) > 1 else None)
        out["answer diversity"] = div.groupby(level=[0, 1], sort=False).mean()
        out["same numbers every time"] = same.dropna().astype(float).groupby(level=[0, 1], sort=False).mean()
    out = out.reset_index()
    out["_order"] = out["arm"].map(order)
    out = out.sort_values("_order").drop(columns="_order")
    out.insert(0, "suite", suite)
    out.to_csv(RESULTS_DIR / f"{suite}_summary.csv", index=False)
    return out


def _fmt(value, column: str) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "–"
    if column in ("success", "score", "persona_match", "same numbers every time"):
        return f"{value:.0%}"
    if column == "cost_usd":
        return f"{value:.4f}"
    if isinstance(value, float):
        return f"{value:.2f}" if abs(value) < 100 else f"{value:.0f}"
    return str(value)


def factor_tables(summary: pd.DataFrame) -> dict:
    """{factor: table} where each table includes the suite's shared baseline row, if any."""
    base = summary[summary["factor"] == "baseline"]
    tables = {}
    for factor in summary["factor"].unique():
        if factor == "baseline":
            continue
        part = summary[summary["factor"] == factor]
        tables[factor] = pd.concat([base, part]) if not base.empty else part
    return tables or {"baseline": base}


def write_report(summary: pd.DataFrame, suite: str) -> None:
    """Markdown tables (one per factor) and a chart per factor."""
    if summary.empty:
        return
    lines = [f"# {suite.title()} experiments\n", f"Generated {datetime.now():%d %B %Y %H:%M}.\n"]
    for factor, table in factor_tables(summary).items():
        cols = ["arm", "runs"] + [c for c in list(METRICS) + ["answer diversity", "same numbers every time", "errors"]
                                  if c in table and table[c].notna().any()]
        names = {c: METRICS[c][0] if c in METRICS else c for c in cols}
        lines += [f"\n## {factor.title()}\n", "| " + " | ".join(names[c] for c in cols) + " |",
                  "|" + "---|" * len(cols)]
        lines += ["| " + " | ".join(_fmt(row[c], c) for c in cols) + " |" for _, row in table.iterrows()]
        plot_factor(table, suite, factor)
    (RESULTS_DIR / f"{suite}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


BLUE, INK, MUTED, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def plot_factor(table: pd.DataFrame, suite: str, factor: str, ncols: int | None = None, path=None, title: bool = True):
    """Small multiples: one horizontal bar panel per metric, one row per arm (single hue, values labelled).

    By default the panels sit side by side (a wide strip for the notebook); ncols arranges them in a grid that
    fits a portrait page (used by the PDF report). Returns the path of the saved image."""
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [c for c in ("success", "persona_fidelity", "grounding", "answer diversity", "lexical_variety",
                          "cost_usd", "latency_s") if c in table and table[c].notna().any()]
    if factor in ("temperature", "top_p", "penalties"):
        panels = [c for c in ("success", "persona_fidelity", "answer diversity", "lexical_variety", "repeated_trigrams")
                  if c in table and table[c].notna().any()]
    if not panels:
        return None
    labels = list(table["arm"])
    cols = min(ncols or len(panels), len(panels))
    rows = math.ceil(len(panels) / cols)
    if ncols:
        size = (7.4, rows * (0.22 * len(labels) + 0.8) + (0.35 if title else 0.1))
    else:
        size = (2.6 * len(panels) + 2.2, 0.42 * len(labels) + 1.3)
    fig, grid = plt.subplots(rows, cols, figsize=size, facecolor=SURFACE, squeeze=False)
    y = list(range(len(labels)))
    for i, ax in enumerate(grid.flat):
        if i >= len(panels):
            ax.axis("off")
            continue
        col = panels[i]
        vals = pd.to_numeric(table[col], errors="coerce").fillna(0).tolist()
        ax.set_facecolor(SURFACE)
        ax.barh(y, vals, color=BLUE, height=0.55)
        pct = col in ("success", "same numbers every time")
        top = 5 if col in ("persona_fidelity", "grounding", "helpfulness") else (1 if pct else max(vals or [1]) * 1.3 or 1)
        ax.set_xlim(0, top)
        for yi, v in zip(y, vals):
            text = f"{v:.0%}" if pct else (f"{v:.4f}" if col == "cost_usd" else f"{v:.2f}")
            ax.text(v + top * 0.02, yi, text, va="center", fontsize=7.5 if ncols else 8, color=INK)
        ax.set_title(METRICS.get(col, (col,))[0], fontsize=8.5 if ncols else 9, color=INK, loc="left")
        ax.tick_params(colors=MUTED, labelsize=7 if ncols else 8, length=0)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        if i % cols == 0:
            ax.set_yticks(y, labels, fontsize=7.5 if ncols else 8, color=INK)
        else:
            ax.set_yticks(y, [""] * len(y))
        ax.invert_yaxis()
    if title:
        fig.suptitle(f"{suite.title()} experiments: {factor}", x=0.01, ha="left", fontsize=10 if ncols else 11,
                     color=INK)
    fig.tight_layout()
    path = path or CHARTS_DIR / f"{suite}_{factor.replace(' ', '_')}.png"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200 if ncols else 150, facecolor=SURFACE)
    plt.close(fig)
    return path

"""Display helpers for the notebook: persona cards, model-support tables, ReAct traces, experiment tables and charts."""
from __future__ import annotations

import html
import json

import pandas as pd

from . import experiments as ex
from .core import EXPERIMENTS_DIR, request_kwargs
from .personas import PERSONAS

# Which columns each comparison shows (the full set is in experiments/results/<suite>_summary.csv).
COLUMNS = {
    "routing": ["success", "persona_match", "persona_fidelity", "helpfulness", "model_calls", "answer_words",
                "cost_usd", "latency_s"],
    "fidelity": ["success", "persona_fidelity", "helpfulness", "grounding", "answer_words", "lexical_variety",
                 "latency_s"],
    "prompt": ["success", "score", "persona_fidelity", "grounding", "model_calls", "format_slips",
               "grounding_checks", "cost_usd", "latency_s"],
    "temperature": ["success", "persona_fidelity", "answer diversity", "same numbers every time", "answer_words",
                    "lexical_variety", "repeated_trigrams"],
    "top_p": ["success", "persona_fidelity", "answer diversity", "answer_words", "lexical_variety",
              "repeated_trigrams"],
    "penalties": ["success", "persona_fidelity", "answer diversity", "answer_words", "lexical_variety",
                  "repeated_trigrams"],
    "max tokens": ["success", "helpfulness", "persona_fidelity", "length_cutoffs", "answer_words", "cost_usd"],
    "model": ["success", "persona_fidelity", "helpfulness", "grounding", "model_calls", "format_slips",
              "answer_words", "cost_usd", "latency_s"],
    "reasoning effort": ["success", "helpfulness", "model_calls", "output_tokens", "cost_usd", "latency_s"],
    "stop sequence": ["success", "invented_observations", "format_errors", "format_slips", "cost_usd"],
    "rules version": ["success", "score", "persona_fidelity", "helpfulness", "grounding", "model_calls", "cost_usd"],
}
LABELS = dict({k: v[0] for k, v in ex.METRICS.items()}, output_tokens="output tokens")
PERCENT = {"success", "score", "persona_match", "same numbers every time"}


def styled(df: pd.DataFrame):
    """Left-aligned table without the index, the style every notebook table uses."""
    left = [{"selector": "th", "props": [("text-align", "left")]}]
    return df.style.hide(axis="index").set_properties(**{"text-align": "left"}).set_table_styles(left)


def _fmt(col, v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "–"
    if col in PERCENT:
        return f"{v:.0%}"
    if col == "cost_usd":
        return f"${v:.4f}"
    if col in ("latency_s",):
        return f"{v:.1f}"
    if isinstance(v, float):
        return f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}"
    return str(v)


def summary(suite: str) -> pd.DataFrame:
    """Per-arm summary of a suite from its saved runs (output tokens added for the effort comparison)."""
    df = ex.load_runs(suite)
    if df.empty:
        raise FileNotFoundError(f"No saved results for '{suite}'. Run: python scripts/run_experiments.py --suite {suite}")
    s = ex.summarize(df, suite)
    if "output_tokens" in df:
        tokens = df.groupby(["factor", "arm"], sort=False)["output_tokens"].mean().rename("output_tokens")
        s = s.merge(tokens.reset_index(), on=["factor", "arm"], how="left")
    return s


def table(suite: str, factor: str):
    """A compact comparison table for one factor (with the suite's shared baseline row, if any)."""
    tables = ex.factor_tables(summary(suite))
    t = tables[factor]
    key = "prompt" if suite == "prompt" else factor
    cols = [c for c in COLUMNS.get(key, list(ex.METRICS)) if c in t and t[c].notna().any()]
    out = pd.DataFrame({"setup": t["arm"], "runs": t["runs"]})
    for c in cols:
        out[LABELS.get(c, c)] = [_fmt(c, v) for v in t[c]]
    return styled(out)


def chart_path(suite: str, factor: str):
    return ex.CHARTS_DIR / f"{suite}_{factor.replace(' ', '_')}.png"


def rebuild_reports(suites=("personas", "prompt", "config", "iteration")) -> list:
    """Regenerate the markdown tables and charts of every suite from the saved runs (no API calls)."""
    done = []
    for suite in suites:
        df = ex.load_runs(suite)
        if not df.empty:
            ex.write_report(ex.summarize(df, suite), suite)
            done.append(f"{suite}: {len(df)} runs")
    return done


# --------------------------------------------------------------------------- personas and model support
def persona_table():
    rows = []
    for p in PERSONAS.values():
        c = p.config
        rows.append({"persona": f"{p.name}, {p.title}", "task": p.task, "tone": p.traits["tone"],
                     "model": c.model, "temperature": f"{c.temperature:g}", "top_p": f"{c.top_p:g}",
                     "presence / frequency penalty": f"{c.presence_penalty or 0} / {c.frequency_penalty or 0}",
                     "max tokens": c.max_completion_tokens, "reasoning effort": c.reasoning_effort,
                     "settings dropped": ", ".join(request_kwargs(c)[1]) or "none"})
    return styled(pd.DataFrame(rows))


def capability_table():
    """Which generation settings each model accepted in scripts/probe_models.py (single settings and combos)."""
    single = json.loads((EXPERIMENTS_DIR / "model_capabilities.json").read_text(encoding="utf-8"))
    rows = {}
    for model, probes in single.items():
        rows[model] = {k: ("yes" if v.get("ok") else "no") for k, v in probes.items() if k != "baseline"}
    combos_path = EXPERIMENTS_DIR / "model_capabilities_combos.json"
    if combos_path.exists():
        for model, probes in json.loads(combos_path.read_text(encoding="utf-8")).items():
            for k, v in probes.items():
                rows.setdefault(model, {})[k] = "yes" if v.get("ok") else "no"
    return pd.DataFrame(rows).fillna("–")


# --------------------------------------------------------------------------- traces
_COLORS = {"Thought": "#2a78d6", "Action": "#b06000", "Observation": "#0b7a3e", "Final Answer": "#0b0b0b",
           "note": "#6b6a65"}


def trace_html(state: dict, max_obs: int = 400) -> str:
    """A ReAct trace as HTML: persona choice, then Thought / Action / Observation steps, then the answer."""
    def line(label, text, color=None):
        return (f'<div style="margin:2px 0"><b style="color:{color or _COLORS.get(label, "#333")}">{label}:</b> '
                f'<span style="white-space:pre-wrap">{html.escape(str(text))}</span></div>')

    parts = []
    for e in state["trace"]:
        t = e["type"]
        if t == "persona":
            p = PERSONAS[e["persona"]]
            parts.append(line("persona", f"{p.name}, {p.title} — {e['reason']} (route {e['route']})", _COLORS["note"]))
        elif t in ("action", "final", "invalid", "grounding"):
            if e.get("thought"):
                parts.append(line("Thought", e["thought"]))
            if t == "action":
                parts.append(line("Action", f"{e['tool']}  {json.dumps(e['args'], ensure_ascii=False)}"))
            elif t == "invalid":
                parts.append(line("format error (sent back)", e["error"], _COLORS["note"]))
            elif t == "grounding":
                parts.append(line("grounding check (sent back)", e["final"][:200], _COLORS["note"]))
            elif e.get("format_note"):
                parts.append(line("format slip", e["format_note"], _COLORS["note"]))
        elif t == "observation":
            text = json.dumps(e["result"], ensure_ascii=False)
            parts.append(line("Observation", text[:max_obs] + ("…" if len(text) > max_obs else "")))
    parts.append(line("Final Answer", state["final"]))
    u = state["usage"]
    parts.append(line("usage", f"{u['calls']} model calls, {u['input_tokens']} in / {u['output_tokens']} out tokens, "
                               f"~${u['cost_usd']:.4f}, stop: {state['stop_reason']}", _COLORS["note"]))
    return '<div style="font-family:ui-monospace,monospace;font-size:12.5px;line-height:1.45">' + "".join(parts) + "</div>"


def saved_trace(suite: str, arm: str, task: str, repeat: int = 1) -> dict:
    """A trace from the experiment logs, in the same shape trace_html() expects."""
    runs = {(r["arm"], r["task"], r["repeat"]): r for r in ex.load_runs(suite).to_dict("records")}
    path = ex.RESULTS_DIR / f"{suite}_traces.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        t = json.loads(line)
        if (t["arm"], t["task"], t["repeat"]) == (arm, task, repeat):
            r = runs[(arm, task, repeat)]
            usage = {"calls": r["model_calls"], "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
                     "cost_usd": r["cost_usd"]}
            return {"trace": t["trace"], "final": r["final"], "usage": usage, "stop_reason": r["stop_reason"]}
    raise KeyError((suite, arm, task, repeat))

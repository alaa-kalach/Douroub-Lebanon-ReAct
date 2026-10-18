"""Offline tests for the evaluation checks and the experiment runner (no API calls).

    python -m pytest tests_experiments.py -q
"""
import json

import pandas as pd
import pytest

from douroub import evaluation as ev
from douroub import experiments as ex
from douroub import tools
from douroub.personas import choose_persona
from douroub.router import route_with_rules
from tests_react import FakeClient

JUDGE = '{"persona_fidelity": 4, "helpfulness": 5, "grounding": 5, "comment": "Precise and grounded."}'


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(ex, "CHARTS_DIR", tmp_path / "charts")
    monkeypatch.setattr(ex, "SANDBOX_DIR", tmp_path / "sandbox")
    old = tools.LOG_DIR, tools.VERBOSE, tools._RECENT
    yield
    tools.LOG_DIR, tools.VERBOSE, tools._RECENT = old


def test_task_ids_are_unique_and_core_tasks_exist():
    assert len(ev.TASKS_BY_ID) == len(ev.TASKS) == 20
    assert set(ev.CORE_TASKS) <= set(ev.TASKS_BY_ID)


@pytest.mark.parametrize("task", ev.TASKS, ids=lambda t: t.id)
def test_auto_mode_matches_the_expected_persona_of_every_task(task):
    assert choose_persona(task.message, route_with_rules(task.message))[0] == task.expected_persona


def _state(final, observations=(), saved=(), stop="answered"):
    trace = [{"type": "observation", "tool": t, "args": a, "status": s} for t, a, s in observations]
    return {"final": final, "trace": trace, "saved_ids": list(saved), "stop_reason": stop}


def test_checks_pass_for_a_good_answer_and_name_what_failed():
    task = ev.TASKS_BY_ID["T01"]
    good = _state("D04 + D08 for 3: (55 + 60) x 3 = 345 USD. Both run in November.",
                  [("budget_calculator", {}, "success"), ("season_check", {}, "success")])
    assert all(ev.check_run(task, good).values())
    bad = _state("About $350, and your booking is confirmed. Also try D99.", [("budget_calculator", {}, "error")])
    failed = {k for k, v in ev.check_run(task, bad).items() if not v}
    assert {"uses budget_calculator", "no booking confirmed", "only real experience IDs"} <= failed


def test_reference_ids_must_come_from_a_tool_and_tool_arguments_are_checked():
    task = ev.TASKS_BY_ID["T09"]
    invented = _state("Saved as INQ-20261017-ABC123.", [("record_customer_interest", {"email": "lina.saad@example.com"}, "success")])
    assert not ev.check_run(task, invented)["only real reference IDs"]
    real = _state("Saved as INQ-20261017-ABC123.", [("record_customer_interest", {"email": "lina.saad@example.com"}, "success")],
                  saved=["INQ-20261017-ABC123"])
    assert all(ev.check_run(task, real).values())


def test_no_tool_task_and_text_metrics():
    task = ev.TASKS_BY_ID["T12"]
    assert not ev.check_run(task, _state("Sure: def sort(x): ...", [("search_experiences", {}, "success")]))["no tool"]
    m = ev.text_metrics("the cedar the cedar the cedar forest")
    assert m["answer_words"] == 7 and m["lexical_variety"] == pytest.approx(3 / 7, abs=1e-3) and m["repeated_trigrams"] > 0
    assert ev.diversity(["same", "same"]) == 0.0 and ev.diversity(["abc", "xyz"]) == 1.0
    assert ev.numbers_in("Total $345 for 3, D04") == ("$345",)


def test_judge_parses_scores():
    client = FakeClient("Sure.\n" + JUDGE)
    out = ev.judge("How much?", "345 USD", "planner", {"trace": []}, client=client)
    assert out["persona_fidelity"] == 4 and out["grounding"] == 5 and "Rami" in client.requests[0]["messages"][0]["content"]


def test_plans_and_estimates():
    runs = ex.plan("prompt")
    assert len(runs) == 6 * len(ev.CORE_TASKS)
    only = ex.plan("config", factors=["model"], tasks=["T01"])
    assert {a.factor for a, _, _ in only} == {"model"} and len(only) == 6
    assert 0 < ex.estimate_cost(runs) < 5


def test_run_one_scores_a_run_without_touching_the_real_logs(tmp_path):
    ex._sandbox()
    client = FakeClient('Thought: total\nAction: budget_calculator\nAction Input: {"experience_ids": ["D04", "D08"], "group_size": 3}',
                        'Thought: season\nAction: season_check\nAction Input: {"experience_id": "D04", "when": "November"}',
                        "Thought: done\nFinal Answer: (55 + 60) x 3 = 345 USD; both run in November.", JUDGE)
    arm = ex.Arm("model", "gpt-5.4-mini", override={"model": "gpt-5.4-mini"}, tasks=("T01",))
    row, trace = ex.run_one(arm, "T01", 1, "config", "r1", use_judge=True, client=client)
    assert row["success"] and row["persona"] == "planner" and row["tool_calls"] == 2 and row["persona_fidelity"] == 4
    assert row["model_calls"] == 3 and row["cost_usd"] > 0 and trace["trace"][0]["type"] == "persona"
    assert tools.LOG_DIR == ex.SANDBOX_DIR and tools.VERBOSE is False


def test_setup_turns_run_in_the_same_conversation():
    ex._sandbox()
    client = FakeClient("Final Answer: Lovely choice! What is your name and email?",
                        'Thought: save\nAction: record_customer_interest\nAction Input: {"name": "Omar Haddad", '
                        '"email": "omar.haddad@example.com", "group_size": 2, "message": "Batroun sunset kayak (D20)"}',
                        "Final Answer: Saved, Khaled will email you.")
    row, _ = ex.run_one(ex.Arm("routing", "Auto"), "T19", 1, "personas", "r1", use_judge=False, client=client)
    assert row["success"], row["failed_checks"]
    assert len(client.requests[1]["messages"]) > len(client.requests[0]["messages"])  # the setup turn is in history


def test_summary_tables_and_charts(tmp_path):
    rows = []
    for arm, factor in (("baseline: CoT, one-shot, system", "baseline"), ("zero-shot", "examples"),
                        ("few-shot (4 examples)", "examples")):
        for task in ("T01", "T05"):
            for rep in (1, 2):
                rows.append({"factor": factor, "arm": arm, "task": task, "repeat": rep, "success": rep == 1,
                             "score": 0.9, "persona_fidelity": 4, "grounding": 5, "model_calls": 3, "format_errors": 0,
                             "cost_usd": 0.005, "latency_s": 3.2, "final": f"Total 345 USD {task} {rep}",
                             "answer_words": 5, "lexical_variety": 1.0, "repeated_trigrams": 0.0})
    ex.RESULTS_DIR.mkdir(parents=True)
    summary = ex.summarize(pd.DataFrame(rows), "prompt")
    assert list(summary["arm"])[0].startswith("baseline") and summary["success"].iloc[0] == 0.5
    assert summary["same numbers every time"].iloc[0] == 1.0
    ex.write_report(summary, "prompt")
    md = (ex.RESULTS_DIR / "prompt_summary.md").read_text(encoding="utf-8")
    assert "## Examples" in md and "baseline" in md and "50%" in md
    assert (ex.CHARTS_DIR / "prompt_examples.png").exists()


def test_run_suite_end_to_end_with_errors_logged(monkeypatch):
    calls = []

    def fake_run_one(arm, task_id, repeat, suite, run_id, *a, **k):
        calls.append(task_id)
        if task_id == "T02":
            raise ValueError("model said no")
        return ({"run_id": run_id, "suite": suite, "factor": arm.factor, "arm": arm.label, "task": task_id,
                 "repeat": repeat, "success": True, "score": 1.0, "final": "ok", "cost_usd": 0.001,
                 "latency_s": 1.0}, {"run_id": run_id, "trace": []})

    monkeypatch.setattr(ex, "run_one", fake_run_one)
    df = ex.run_suite("prompt", factors=["examples"], tasks=["T01", "T02"], workers=2, log=lambda *_: None)
    assert len(df) == 3 * 2 and df["error"].notna().sum() == 3
    saved = [json.loads(l) for l in (ex.RESULTS_DIR / "prompt_runs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(saved) == 6 and (ex.RESULTS_DIR / "prompt_summary.md").exists()


def test_rejudge_rescores_saved_answers_with_the_full_tool_results():
    ex.RESULTS_DIR.mkdir(parents=True)
    long_result = {"status": "success", "results": [{"id": "D18", "note": "x" * 3000, "group_size": "1-15 people"}]}
    run = {"run_id": "r1", "suite": "config", "factor": "model", "arm": "gpt-5.4-mini", "task": "T06", "repeat": 1,
           "persona": "storyteller", "final": "D18 takes 1-15 people.", "success": True, "persona_fidelity": 1}
    trace = {"run_id": "r1", "arm": "gpt-5.4-mini", "task": "T06", "repeat": 1,
             "trace": [{"type": "observation", "result": long_result}]}
    (ex.RESULTS_DIR / "config_runs.jsonl").write_text(json.dumps(run) + "\n", encoding="utf-8")
    (ex.RESULTS_DIR / "config_traces.jsonl").write_text(json.dumps(trace) + "\n", encoding="utf-8")
    client = FakeClient(JUDGE)
    df = ex.rejudge("config", workers=1, client=client, log=lambda *_: None)
    assert df["persona_fidelity"].iloc[0] == 4
    assert "1-15 people" in client.requests[0]["messages"][0]["content"]   # beyond the old 1,500-character cut


def test_empty_optional_tool_arguments_are_ignored():
    tools.VERBOSE = False
    args = {"query": "Byblos", "interests": [], "category": "", "region": "", "max_price_per_person": None}
    assert tools.execute_tool("search_experiences", args)["status"] == "success"


def test_rules_v2_adds_two_rules_and_the_old_suites_stay_on_v1():
    from douroub.instructions import RULES_V1, RULES_V2, build_instructions
    from douroub.react import PromptVariant, build_prompt
    from douroub.personas import PERSONAS
    assert len(RULES_V2) == len(RULES_V1) + 2 and set(RULES_V1) < set(RULES_V2)
    assert "give the visitor the reference ID" in build_instructions()
    assert "give the visitor the reference ID" not in build_instructions(version=1)
    system, _ = build_prompt(PERSONAS["guardian"], PromptVariant(rules=1))
    assert "a date is optional" not in system
    assert PromptVariant(rules=1).short().endswith("rules v1") and "rules" not in PromptVariant().short()
    for suite in ("personas", "prompt", "config"):
        assert {arm.variant.rules for arm in ex.SUITES[suite]()} == {1}
    labels = [arm.variant.rules for arm in ex.SUITES["iteration"]()]
    assert labels == [1, 2] and len(ex.plan("iteration")) == 24


def test_report_helpers_build_tables_from_saved_runs():
    from douroub import report
    rows = [{"factor": f, "arm": a, "task": "T01", "repeat": 1, "success": True, "score": 1.0, "model_calls": 2,
             "cost_usd": 0.002, "latency_s": 3.0, "final": "345", "persona_fidelity": 5, "output_tokens": 100}
            for f, a in (("model", "gpt-5.4-nano"), ("model", "gpt-5.4-mini"))]
    ex.RESULTS_DIR.mkdir(parents=True)
    (ex.RESULTS_DIR / "config_runs.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    t = report.table("config", "model").data
    assert list(t["setup"]) == ["gpt-5.4-nano", "gpt-5.4-mini"] and t["success rate"].iloc[0] == "100%"
    assert "settings dropped" in report.persona_table().data and "none" in set(report.persona_table().data["settings dropped"])
    html = report.trace_html({"trace": [{"type": "persona", "persona": "planner", "reason": "r", "route": "X"}],
                              "final": "<b>", "usage": {"calls": 1, "input_tokens": 1, "output_tokens": 1,
                                                       "cost_usd": 0.0}, "stop_reason": "answered"})
    assert "Rami" in html and "&lt;b&gt;" in html

"""Offline tests for the Gradio app (no API calls, no browser).

    python -m pytest tests_ui.py -q
"""
import pytest

from douroub import ui
from douroub.agent import build_graph, run_turn
from tests_react import FakeClient


def _state():
    client = FakeClient('Thought: totals\nAction: budget_calculator\nAction Input: {"experience_ids": ["D04"], "group_size": 2}',
                        "Thought: done\nFinal Answer: $110 for two.")
    return run_turn(build_graph(), "How much is D04 for 2?", persona="planner", use_llm_router=False, client=client)


def test_trace_panels_show_persona_steps_and_usage():
    panels = ui.trace_panels(_state(), done=True)
    titles = [p["metadata"]["title"] for p in panels]
    assert "Rami" in titles[0] and "Step 1: calculated the total" in titles[1] and "ready to answer" in titles[2]
    assert "**Observation:**" in panels[1]["content"] and panels[1]["metadata"]["status"] == "done"
    assert "model calls" in panels[0]["metadata"]["log"]


def test_trace_panels_while_running_end_with_a_pending_panel():
    state = {"trace": [{"type": "persona", "persona": "host", "reason": "general question", "route": "GENERAL_QUESTION"},
                       {"type": "action", "tool": "search_experiences", "args": {"region": "Chouf"}, "thought": "look"}]}
    panels = ui.trace_panels(state, done=False)
    assert panels[1]["metadata"]["status"] == "pending" and panels[-1]["metadata"]["title"].endswith("Thinking…")


def test_guard_refusal_is_shown_as_a_safety_check():
    state = {"trace": [{"type": "action", "tool": "record_customer_interest", "args": {}},
                       {"type": "observation", "status": "error", "result": {"status": "error", "error": "details_not_from_visitor"}}]}
    assert "safety check" in ui.trace_panels(state, done=True)[0]["metadata"]["title"]


def test_chat_without_a_key_answers_politely(monkeypatch):
    monkeypatch.setattr(ui, "has_api_key", lambda: False)
    history, box, thread = list(ui.chat("Hello", [], None, "auto", True, "one", "system", True))[-1]
    assert history[0] == {"role": "user", "content": "Hello"} and "not configured" in history[-1]["content"]
    assert box == "" and thread


def test_persona_cards_show_every_persona_and_its_settings():
    html = ui._persona_cards_html()
    for p in ui.PERSONAS.values():
        assert p.name in html and p.config.model in html
    assert "Why these settings" in html


def test_page_builds():
    demo = ui.build_demo()
    assert demo is not None and "Auto" in ui.PERSONA_CHOICES[0][0] and len(ui.PERSONA_CHOICES) == 5


def test_pdf_report_builds_from_saved_results(tmp_path):
    from douroub import experiments as ex
    from douroub.pdf_report import build
    if ex.load_runs("prompt").empty:
        pytest.skip("no saved experiment results")
    from pypdf import PdfReader
    out = build(tmp_path / "report.pdf")
    text = "".join(page.extract_text() for page in PdfReader(out).pages)
    assert len(PdfReader(out).pages) >= 8 and "Headline findings" in text and "Reflection" in text


def test_prompt_style_settings_reach_the_agent_and_the_trace():
    v = ui._variant(False, "few", "message")
    assert (v.thought, v.shots, v.placement) == (False, "few", "message")
    assert ui.style_label(v) == "no chain of thought · few-shot · instructions inside the message"
    panels = ui.trace_panels(_state(), done=True, style=ui.style_label(ui._variant(True, "one", "system")))
    assert "**Prompt style:** chain of thought · one-shot · instructions in the system prompt" in panels[0]["content"]


def test_only_one_prompt_style_is_tested_at_a_time():
    # all off: the baseline
    assert ui.effective_style(False, False, False, "few", False, "message") == (True, "one", "system")
    # few-shot on: only the examples change
    assert ui.effective_style(False, False, True, "few", False, "message") == (True, "few", "system")
    # switching placement on turns the others off and changes only the placement
    out = ui.switch_style("place", True, False, "few", "message")
    assert [out[0]["value"], out[2]["value"], out[4]["value"]] == [False, False, True]
    assert [out[1]["visible"], out[3]["visible"], out[5]["visible"]] == [False, False, True]
    assert out[6:9] == (True, "one", "message") and "instruction placement" in out[9]
    # switching it off again restores the baseline
    out = ui.switch_style("place", False, False, "few", "message")
    assert out[6:9] == (True, "one", "system") and "default" in out[9]

"""Offline tests for the ReAct agent: parser, prompts, personas, tools, and the LangGraph loop.

    python -m pytest tests_react.py -q

No API calls: the model is replaced by a scripted fake client that returns prepared replies.
"""
from types import SimpleNamespace as NS

import json

import pytest

from douroub import agent, core, tools
from douroub.instructions import RULES, build_instructions, check_conflicts
from douroub.personas import PERSONAS, choose_persona
from douroub.react import PromptVariant, build_prompt, format_observation, parse_step
from douroub.router import Route


class FakeClient:
    """Plays back prepared model replies and records every request."""

    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        text = self.replies.pop(0)
        return NS(choices=[NS(message=NS(content=text))],
                  usage=NS(prompt_tokens=1000, completion_tokens=40, prompt_tokens_details=NS(cached_tokens=0)))


@pytest.fixture(autouse=True)
def isolated_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "LOG_DIR", tmp_path)
    tools._RECENT.clear()
    return tmp_path


def run(client, message, persona="host", thread="t", graph=None, **kw):
    graph = graph or agent.build_graph()
    return agent.run_turn(graph, message, thread_id=thread, persona=persona, use_llm_router=False,
                          client=client, **kw)


# ----------------------------------------------------------------------------------------------- parser
@pytest.mark.parametrize("text,tool,args", [
    ('Thought: x\nAction: search_experiences\nAction Input: {"region": "Chouf"}', "search_experiences", {"region": "Chouf"}),
    ('Action: search_experiences\nAction Input: ```json\n{"region": "Chouf"}\n```', "search_experiences", {"region": "Chouf"}),
    ("Action: search_experiences\nAction Input: {'region': 'Chouf'}", "search_experiences", {"region": "Chouf"}),
    ('Action: season_check({"experience_id": "D20", "when": "May"})', "season_check", {"experience_id": "D20", "when": "May"}),
    ('**Thought:** x\n**Action:** `budget_calculator`\n**Action Input:** {"experience_ids": ["D04"]}', "budget_calculator", {"experience_ids": ["D04"]}),
    ("Thought: list all\nAction: search_experiences\nAction Input: {}", "search_experiences", {}),
])
def test_parser_detects_actions_in_common_formats(text, tool, args):
    step = parse_step(text)
    assert step.kind == "action" and step.tool == tool and step.args == args


def test_parser_cuts_off_an_invented_observation():
    step = parse_step('Thought: x\nAction: search_experiences\nAction Input: {"region": "Byblos"}\n'
                      'Observation: {"total_matches": 99}\nThought: great\nFinal Answer: 99 options!')
    assert step.kind == "action" and step.invented_observation and step.args == {"region": "Byblos"}


@pytest.mark.parametrize("text,kind,final", [
    ("Thought: I know this.\nFinal Answer: Ahla w sahla!", "final", "Ahla w sahla!"),
    ("Action: Final Answer\nAction Input: Hello there", "final", "Hello there"),
    ("Just a plain reply with no markers.", "final", "Just a plain reply with no markers."),
])
def test_parser_detects_final_answers(text, kind, final):
    step = parse_step(text)
    assert step.kind == kind and step.final == final


@pytest.mark.parametrize("text,error", [
    ("Thought: hmm\nAction: book_now\nAction Input: {}", "Unknown tool"),
    ("Action: search_experiences\nAction Input: {region: Chouf", "not closed"),
    ("Action: search_experiences\nAction Input: region=Chouf", "JSON object"),
    ("Thought: I am still thinking", "No Action or Final Answer"),
    ("Final Answer:", "empty"),
])
def test_parser_explains_invalid_steps(text, error):
    step = parse_step(text)
    assert step.kind == "invalid" and error in step.error


def test_plain_text_answer_is_accepted_but_flagged():
    assert "format" in parse_step("Hello!").error


def test_observation_is_compact_and_truncated():
    obs = format_observation({"status": "success", "text": "x" * 10000}, limit=200)
    assert obs.startswith("Observation: {") and obs.endswith("(truncated)") and len(obs) < 260


# ----------------------------------------------------------------------------------------------- prompts
def test_prompt_contains_persona_instructions_tools_and_knowledge():
    system, prefix = build_prompt(PERSONAS["planner"])
    for part in ["# PERSONA: Rami", "# INSTRUCTIONS", "## Worked example", "# TOOLS", "budget_calculator",
                 "season_check", "# BUSINESS KNOWLEDGE", "Today's date"]:
        assert part in system, part
    assert prefix == ""


def test_prompt_variants():
    no_cot = build_prompt(PERSONAS["host"], PromptVariant(thought=False))[0]
    assert "Thought:" not in no_cot.split("# TOOLS")[0]
    zero = build_prompt(PERSONAS["host"], PromptVariant(shots="zero"))[0]
    few = build_prompt(PERSONAS["host"], PromptVariant(shots="few"))[0]
    assert "Worked example" not in zero and "More examples" in few
    system, prefix = build_prompt(PERSONAS["host"], PromptVariant(placement="message"))
    assert "# INSTRUCTIONS" not in system and "# INSTRUCTIONS" in prefix and prefix.endswith("# VISITOR MESSAGE\n")
    bare = build_prompt(PERSONAS["host"], PromptVariant(instructions=False))[0]
    assert "## Rules" not in bare and "Action Input" in bare


# ----------------------------------------------------------------------------------------------- instructions
def test_instructions_have_single_purpose_rules_io_and_a_worked_example():
    block = build_instructions()
    assert "## Input you receive" in block and "## Output you must return" in block and "## Worked example" in block
    for rule in RULES:  # one purpose each: short, at most two sentences
        assert len(rule) < 220 and rule.count(". ") <= 1, rule
    example = block.split("## Worked example")[1]
    assert "Action: budget_calculator" in example and "Observation:" in example and "Final Answer:" in example


def test_personas_do_not_conflict_with_instructions():
    for p in PERSONAS.values():
        assert check_conflicts(p.prompt) == [], p.key
    assert check_conflicts("Feel free to guess the price if unsure.") == ["feel free to guess"]


# ----------------------------------------------------------------------------------------------- personas
def test_four_personas_with_different_tasks_and_configs():
    assert set(PERSONAS) == {"host", "planner", "storyteller", "guardian"}
    assert len({p.task for p in PERSONAS.values()}) == 4
    assert len({p.config for p in PERSONAS.values()}) == 4
    for p in PERSONAS.values():
        assert set(p.traits) == {"tone", "vocabulary", "response style", "values"}
        assert p.name in p.prompt
        for setting in p.config_rationale:   # every explained setting is actually set
            assert getattr(p.config, setting) is not None, (p.key, setting)
    assert PERSONAS["planner"].config.temperature < PERSONAS["host"].config.temperature < PERSONAS["storyteller"].config.temperature


@pytest.mark.parametrize("message,route,expected", [
    ("Plan 2 days in the north under $120", Route("GENERAL_QUESTION"), "planner"),
    ("How much is D04 for 3 people?", Route("GENERAL_QUESTION"), "planner"),
    ("Tell me about Byblos", Route("GENERAL_QUESTION"), "storyteller"),
    ("Is D10 wheelchair accessible?", Route("FEEDBACK_UNKNOWN"), "guardian"),
    ("I want to book D03, I'm Lina, lina@example.com", Route("LEAD_COLLECTION"), "host"),
    ("Hi! What does Douroub do?", Route("GENERAL_QUESTION"), "host"),
])
def test_auto_mode_picks_the_persona_for_the_task(message, route, expected):
    assert choose_persona(message, route)[0] == expected


# ----------------------------------------------------------------------------------------------- model settings
def test_unsupported_settings_are_dropped_and_reported():
    kwargs, dropped = core.request_kwargs(core.LLMConfig(model="gpt-5.4-mini", temperature=0.2, stop=("Observation:",)))
    assert kwargs["temperature"] == 0.2 and "stop" not in kwargs and dropped == ["stop"]
    kwargs, dropped = core.request_kwargs(core.LLMConfig(model="gpt-4.1-mini", stop=("Observation:",), reasoning_effort="low"))
    assert kwargs["stop"] == ["Observation:"] and dropped == ["reasoning_effort"]


def test_reasoning_effort_low_disables_sampling_settings():
    kwargs, dropped = core.request_kwargs(core.LLMConfig(model="gpt-5.4-mini", temperature=0.1, top_p=0.5,
                                                         reasoning_effort="low"))
    assert "temperature" not in kwargs and set(dropped) == {"temperature", "top_p"}
    kwargs, dropped = core.request_kwargs(core.LLMConfig(model="gpt-6-luna", temperature=0.5, presence_penalty=0.6,
                                                         reasoning_effort="none"))
    assert kwargs["temperature"] == 0.5 and dropped == ["presence_penalty"]


@pytest.mark.parametrize("key", list(PERSONAS))
def test_every_persona_setting_is_actually_sent(key):
    assert core.request_kwargs(PERSONAS[key].config)[1] == []


def test_cost_estimate_uses_cached_price():
    assert core.estimate_cost("gpt-6-luna", 1_000_000, 0, 0) == pytest.approx(0.10)
    assert core.estimate_cost("gpt-6-luna", 1_000_000, 1_000_000, 1_000_000) == pytest.approx(0.51)


# ----------------------------------------------------------------------------------------------- new tools
def test_budget_calculator_and_season_check():
    r = tools.budget_calculator(["D04", "d8"], 3, budget_per_person=100)
    assert r["total_for_group_usd"] == 345 and r["within_budget_per_person"] is False
    assert r["calculation"] == "(55 + 60) x 3 = 345 USD"
    assert tools.budget_calculator(["D99"])["error"] == "unknown_experience"
    assert tools.season_check("D20", "21 November")["runs_that_month"] is False
    assert tools.season_check("D19", "January")["runs_that_month"] is True
    assert tools.season_check("D20", "someday")["error"] == "invalid_date"
    assert len(tools.TOOL_SPECS) == 7 and "season_check:" in tools.tool_catalog_text()


# ----------------------------------------------------------------------------------------------- the ReAct loop
def test_grounding_check_sends_back_catalog_facts_quoted_without_a_tool():
    client = FakeClient("Byblos is lovely. D04 costs $55 per person.",
                        'Thought: check the catalog\nAction: search_experiences\nAction Input: {"region": "Byblos"}',
                        "Thought: done\nFinal Answer: D04 Heritage Through Time, $55 per person.")
    out = run(client, "Tell me about Byblos", persona="storyteller")
    assert [e["type"] for e in out["trace"]] == ["persona", "grounding", "action", "observation", "final"]
    assert "System check" in client.requests[1]["messages"][-1]["content"]
    assert out["final"].startswith("D04") and out["usage"]["calls"] == 3


def test_grounding_check_insists_on_a_tool_at_most_twice_and_ignores_answers_without_catalog_facts():
    client = FakeClient("Final Answer: D04 is $55.", "Final Answer: I should check the catalog first.",
                        "Final Answer: D04 is $55.")
    out = run(client, "Price of D04?")
    assert out["final"] == "D04 is $55." and [e["type"] for e in out["trace"]].count("grounding") == 2
    notes = [m["content"] for m in client.requests[2]["messages"] if "System check" in m["content"]]
    assert len(notes) == 2 and "still have not called a tool" in notes[1] and "never mention it" in notes[0]
    out = run(FakeClient("Final Answer: Ahla w sahla! How can I help?"), "Hi")
    assert out["usage"]["calls"] == 1


def test_final_answer_control_characters_are_removed():
    out = run(FakeClient("Final Answer: Hello\r there\x1b[2K"), "Hi")
    assert out["final"] == "Hello there[2K"


def test_loop_runs_action_observation_then_answer():
    client = FakeClient('Thought: totals\nAction: budget_calculator\nAction Input: {"experience_ids": ["D04"], "group_size": 2}',
                        "Thought: done\nFinal Answer: $110 for two.")
    out = run(client, "How much is D04 for 2?", persona="planner")
    assert out["final"] == "$110 for two." and out["stop_reason"] == "answered"
    assert [e["type"] for e in out["trace"]] == ["persona", "action", "observation", "final"]
    second = client.requests[1]["messages"]
    assert second[-1]["role"] == "user" and second[-1]["content"].startswith("Observation: ")
    assert '"total_for_group_usd":110' in second[-1]["content"] and "(Next: 'Thought:'" in second[-1]["content"]
    assert client.requests[0]["temperature"] == 0.1 and client.requests[0]["model"] == "gpt-5.4-mini"
    assert out["usage"]["calls"] == 2


def test_format_errors_are_sent_back_and_corrected():
    client = FakeClient("Thought: let me think", "Final Answer: Ahla w sahla!")
    out = run(client, "hi")
    assert out["final"] == "Ahla w sahla!" and [e["type"] for e in out["trace"]][1:] == ["invalid", "final"]
    assert "Format error" in client.requests[1]["messages"][-1]["content"]


def test_step_limit_forces_an_answer():
    loop = ['Action: search_experiences\nAction Input: {"region": "R%d"}' % i for i in range(agent.MAX_STEPS)]
    client = FakeClient(*loop, "Final Answer: Here is what I found.")
    out = run(client, "anything in the north?")
    assert out["stop_reason"] == "step_limit" and out["final"] == "Here is what I found."
    assert "Step limit reached" in client.requests[-1]["messages"][-1]["content"]


def test_repeated_action_is_not_run_twice():
    act = 'Action: search_experiences\nAction Input: {"region": "Chouf"}'
    client = FakeClient(act, act, "Final Answer: Four options in the Chouf.")
    out = run(client, "Chouf?")
    obs = [e for e in out["trace"] if e["type"] == "observation"]
    assert obs[0]["status"] == "success" and obs[1]["result"]["error"] == "repeated_action"


def test_invented_email_is_refused_and_memory_keeps_real_ids(isolated_logs):
    graph = agent.build_graph()
    client = FakeClient(
        'Action: record_customer_interest\nAction Input: {"name": "Rana", "email": "rana@example.com", "group_size": 3, "message": "D04 Heritage Through Time"}',
        "Final Answer: What email should the team use?",
        'Action: record_customer_interest\nAction Input: {"name": "Rana", "email": "rana.k@example.com", "group_size": 3, "message": "D04 Heritage Through Time"}',
        "Final Answer: Saved!")
    first = run(client, "Book D04 for 3, I'm Rana", graph=graph, thread="lead")
    assert first["trace"][2]["result"]["error"] == "details_not_from_visitor" and first["saved_ids"] == []
    second = run(client, "rana.k@example.com", graph=graph, thread="lead")
    assert len(second["saved_ids"]) == 1 and second["saved_ids"][0].startswith("INQ-")
    assert len(second["messages"]) == 4                      # both turns are in the conversation memory
    assert second["saved_ids"][0] in client.requests[2]["messages"][0]["content"] or \
           "No inquiry" in client.requests[2]["messages"][0]["content"]


def test_feedback_for_a_general_question_is_refused():
    client = FakeClient('Action: record_feedback\nAction Input: {"question": "restaurants in Byblos?"}',
                        "Final Answer: Byblos is known for seafood by the old port.")
    out = run(client, "Any restaurant suggestions in Byblos?")
    assert out["trace"][2]["result"]["error"] == "not_feedback"


def test_auto_persona_and_overrides_reach_the_request():
    client = FakeClient("Final Answer: ok")
    out = run(client, "Tell me about Byblos", persona="auto", config_override={"temperature": 0.9},
              variant=PromptVariant(thought=False))
    assert out["active_persona"] == "storyteller" and client.requests[0]["temperature"] == 0.9
    assert client.requests[0]["presence_penalty"] == 0.6
    assert "# PERSONA: Layla" in client.requests[0]["messages"][0]["content"]


def test_graph_shape():
    g = agent.build_graph().get_graph()
    assert {"prepare", "reason", "act", "force_answer", "finalize"} <= set(g.nodes)


# ----------------------------------------------------------------------------------------------- group size (v3)
def test_an_inquiry_without_the_number_of_people_is_not_saved(isolated_logs):
    client = FakeClient('Action: record_customer_interest\nAction Input: {"name": "Sami Kay", "email": "sami@example.com", "message": "D05 on 12 Dec"}',
                        "Final Answer: How many people will join?")
    out = run(client, "I'm Sami Kay, sami@example.com, D05 on 12 Dec please")
    obs = out["trace"][2]["result"]
    assert obs["status"] == "error" and out["saved_ids"] == []
    assert "group_size" in json.dumps(obs)


def test_a_number_of_people_the_visitor_never_gave_is_refused(isolated_logs):
    client = FakeClient('Action: record_customer_interest\nAction Input: {"name": "Sami Kay", "email": "sami@example.com", "group_size": 2, "message": "D05"}',
                        "Final Answer: How many people will join?")
    out = run(client, "I'm Sami Kay, sami@example.com, I'd like D05")
    obs = out["trace"][2]["result"]
    assert obs["error"] == "details_not_from_visitor" and "number of people" in obs["missing"]


@pytest.mark.parametrize("n,text,ok", [
    (3, "we are 3", True), (2, "for two people", True), (2, "my sister and me", True), (1, "just me", True),
    (4, "a family of four", True), (2, "D05 on 12 Dec, budget $200", False), (5, "D05 please", False),
    (12, "12 dec", True),
])
def test_group_size_must_come_from_the_visitor(n, text, ok):
    assert tools.group_size_from_visitor(n, text) is ok


def test_rules_v3_require_the_number_of_people():
    from douroub.instructions import RULES_V2, RULES_V3
    assert len(RULES_V3) == len(RULES_V2) and "the number of people" in build_instructions()
    assert "the number of people" not in build_instructions(version=2)

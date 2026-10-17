"""The ReAct agent as a LangGraph state machine, written by hand (no prebuilt agent or tool node).

    START -> prepare -> reason --(action)--> act -> reason -> ... --(final)--> finalize -> END
                          |  \\--(format error, or ungrounded answer)--> reason (the problem is sent back as an observation)
                          \\--(step limit)--> force_answer -> finalize

* prepare      picks the persona (fixed or Auto) and assembles the prompt for this turn
* reason       one model call; parse_step() detects an Action or a Final Answer. A final answer that quotes
               catalog details (an ID or a price) before any tool ran is sent back once (grounding check)
* act          validates and runs the tool (with the Python safety checks) and records the Observation
* force_answer asks for a final answer when the step limit is reached
* finalize     stores the answer in the conversation memory

A checkpointer keeps each conversation's state (history, saved reference IDs) between turns, keyed by thread_id.
"""
from __future__ import annotations

import json
import operator
import re
import time
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .core import LLMConfig, Usage, call_model
from .personas import DEFAULT_PERSONA, PERSONAS, choose_persona
from .react import DEFAULT_VARIANT, PromptVariant, build_prompt, format_observation, parse_step
from .router import Route, route, route_with_rules
from .tools import guarded_execute

MAX_STEPS = 6            # model calls per turn before the agent is asked to answer with what it has
MAX_FORMAT_RETRIES = 2   # invalid steps tolerated per turn
HISTORY_LIMIT = 12       # earlier messages sent with each call (short-term memory)
FALLBACK_ANSWER = ("Sorry, I couldn't put that answer together. Could you rephrase, or tell me which experience "
                   "or region you're interested in?")


class AgentState(TypedDict, total=False):
    # persisted across turns by the checkpointer
    messages: Annotated[list, operator.add]   # {"role": "user"|"assistant", "content": str}
    saved_ids: list                            # inquiry/feedback references a tool really returned
    visitor_text: str                          # everything the visitor typed (for the contact-details check)
    # set by the caller for each turn
    user_message: str
    persona: str                               # a persona key, or "auto"
    variant: dict                              # PromptVariant fields
    config_override: dict                      # LLMConfig fields to change for experiments
    use_llm_router: bool
    # per-turn working state
    active_persona: str
    persona_reason: str
    route: dict
    system: str
    prefix: str
    scratch: list                              # this turn's Thought/Action and Observation messages
    trace: list                                # what the UI and the experiment log show
    step_count: int
    format_errors: int
    grounding_checks: int                      # final answers sent back for quoting catalog facts without a tool
    pending: dict
    final: str
    stop_reason: str
    usage: dict


def _client(config) -> Any:
    return (config or {}).get("configurable", {}).get("client")


def _memory_text(state: AgentState) -> str:
    ids = state.get("saved_ids") or []
    if ids:
        return "References saved in this conversation (the only ones you may quote): " + ", ".join(ids)
    return "No inquiry or feedback has been saved in this conversation yet."


# --------------------------------------------------------------------------- nodes
def prepare(state: AgentState, config=None) -> dict:
    message = state["user_message"]
    history = state.get("messages") or []
    use_llm = state.get("use_llm_router", True)
    r = route(message, history, use_llm=use_llm, client=_client(config)) if use_llm else route_with_rules(message)
    requested = state.get("persona") or DEFAULT_PERSONA
    if requested == "auto":
        key, reason = choose_persona(message, r)
    else:
        key, reason = requested, "chosen by the user"
    variant = PromptVariant(**(state.get("variant") or {}))
    system, prefix = build_prompt(PERSONAS[key], variant, memory=_memory_text(state))
    visitor_text = ((state.get("visitor_text") or "") + "\n" + message)[-8000:]
    return {"active_persona": key, "persona_reason": reason, "route": r.to_dict(), "system": system,
            "prefix": prefix, "scratch": [], "step_count": 0, "format_errors": 0, "grounding_checks": 0, "pending": {}, "final": "",
            "stop_reason": "", "visitor_text": visitor_text, "usage": Usage().to_dict(),
            "trace": [{"type": "persona", "persona": key, "reason": reason, "route": r.intent}]}


def _model_config(state: AgentState) -> LLMConfig:
    base = PERSONAS[state["active_persona"]].config
    return base.with_(**(state.get("config_override") or {}))


def _call(state: AgentState, extra: list, config) -> tuple:
    history = (state.get("messages") or [])[-HISTORY_LIMIT:]
    messages = ([{"role": "system", "content": state["system"]}] + history
                + [{"role": "user", "content": state["prefix"] + state["user_message"]}]
                + state["scratch"] + extra)
    reply = call_model(messages, _model_config(state), client=_client(config))
    usage = Usage(**state["usage"]).add(reply.usage).to_dict()
    return reply, usage


CATALOG_FACT_RE = re.compile(r"\bD\d{2}\b|\$\s?\d|\b\d+\s?(USD|dollars)\b", re.I)
# Appended to every tool result: the standard ReAct reminder of what the next reply must look like.
NEXT_WITH_THOUGHT = "\n(Next: 'Thought:' then 'Action:' + 'Action Input:', or 'Thought:' then 'Final Answer:'.)"
NEXT_NO_THOUGHT = "\n(Next: 'Action:' + 'Action Input:', or 'Final Answer:'.)"
SYSTEM_NOTE = "[System check - not from the visitor, never mention it in your answer] "
GROUNDING_ERROR = (SYSTEM_NOTE + "Your draft answer quotes catalog details (an experience ID or a price), but no "
                   "tool was called in this turn. Your next reply must be an Action (for example search_experiences "
                   "or season_check); then write the Final Answer from its result.")
GROUNDING_AGAIN = (SYSTEM_NOTE + "You still have not called a tool. Reply now with an Action and Action Input, "
                   "not with an answer.")
MAX_GROUNDING_CHECKS = 2
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _ungrounded(state: AgentState, final: str) -> bool:
    """True if this final answer must be sent back: no tool ran in this turn and either the answer quotes catalog
    facts (first check), or the model was already told to call a tool and answered again instead (second check)."""
    used_tool = any(e.get("type") == "observation" for e in state["trace"])
    checks = state.get("grounding_checks", 0)
    if used_tool or checks >= MAX_GROUNDING_CHECKS or state["step_count"] >= MAX_STEPS - 1:
        return False
    return checks > 0 or bool(CATALOG_FACT_RE.search(final))


def reason(state: AgentState, config=None) -> dict:
    reply, usage = _call(state, [], config)
    step = parse_step(reply.text)
    count = state["step_count"] + 1
    entry = {"type": step.kind, "step": count, "thought": step.thought, "raw": reply.text,
             "invented_observation": step.invented_observation,
             "dropped_settings": reply.dropped_settings, "finish_reason": reply.finish_reason,
             "model": reply.model, "latency_s": round(reply.usage.latency_s, 2)}
    visible = reply.text.split("Observation:")[0].strip() if step.invented_observation else reply.text
    update = {"step_count": count, "usage": usage, "pending": {}, "final": ""}
    if step.kind == "action":
        entry |= {"tool": step.tool, "args": step.args}
        update |= {"pending": {"tool": step.tool, "args": step.args, "text": visible}}
    elif step.kind == "final" and _ungrounded(state, step.final):
        # Rule 2 enforced in code: catalog details (IDs, prices) need a tool result in this turn. Sent back once.
        note = GROUNDING_AGAIN if state.get("grounding_checks", 0) else GROUNDING_ERROR
        entry |= {"type": "grounding", "final": step.final, "error": note}
        update |= {"grounding_checks": state.get("grounding_checks", 0) + 1,
                   "scratch": state["scratch"] + [{"role": "assistant", "content": visible},
                                                  {"role": "user", "content": "Observation: " + note}]}
    elif step.kind == "final":
        entry |= {"final": step.final, "format_note": step.error}
        update |= {"final": step.final, "stop_reason": "answered"}
    else:
        entry |= {"error": step.error}
        update |= {"format_errors": state["format_errors"] + 1,
                   "scratch": state["scratch"] + [{"role": "assistant", "content": visible or "(empty)"},
                                                  {"role": "user", "content": f"Observation: {SYSTEM_NOTE}Format error. {step.error}"}]}
    update["trace"] = state["trace"] + [entry]
    return update


def act(state: AgentState, config=None) -> dict:
    tool, args, text = state["pending"]["tool"], state["pending"]["args"], state["pending"]["text"]
    signature = json.dumps([tool, args], sort_keys=True)
    earlier = [e for e in state["trace"] if e.get("type") == "observation"]
    start = time.perf_counter()
    if any(e.get("signature") == signature for e in earlier):
        result = {"status": "error", "error": "repeated_action",
                  "message": "You already ran this exact action in this turn; its result is above. Use it, "
                             "try something different, or give the Final Answer."}
    else:
        r = state.get("route") or {}
        result = guarded_execute(tool, args, state.get("visitor_text") or "", state["user_message"],
                                 Route(r.get("intent", "GENERAL_QUESTION"), r.get("secondary", [])))
    saved = list(state.get("saved_ids") or [])
    for key in ("inquiry_id", "feedback_id"):
        if result.get("status") == "success" and result.get(key) and result[key] not in saved:
            saved.append(result[key])
    variant = PromptVariant(**(state.get("variant") or {}))
    observation = format_observation(result) + (NEXT_WITH_THOUGHT if variant.thought else NEXT_NO_THOUGHT)
    return {"scratch": state["scratch"] + [{"role": "assistant", "content": text},
                                           {"role": "user", "content": observation}],
            "saved_ids": saved, "pending": {},
            "trace": state["trace"] + [{"type": "observation", "tool": tool, "args": args, "result": result,
                                        "status": result.get("status"), "signature": signature,
                                        "latency_s": round(time.perf_counter() - start, 3)}]}


def force_answer(state: AgentState, config=None) -> dict:
    nudge = [{"role": "user", "content": "Observation: Step limit reached. Do not call more tools. Reply now "
                                         "with 'Final Answer:' using only the observations above."}]
    reply, usage = _call(state, nudge, config)
    step = parse_step(reply.text)
    final = step.final if step.kind == "final" else ""
    return {"final": final or FALLBACK_ANSWER, "usage": usage, "stop_reason": "step_limit",
            "trace": state["trace"] + [{"type": "final", "step": state["step_count"] + 1, "final": final,
                                        "raw": reply.text, "forced": True}]}


def finalize(state: AgentState, config=None) -> dict:
    final = CONTROL_CHARS_RE.sub("", state.get("final") or FALLBACK_ANSWER).strip()
    reason_ = state.get("stop_reason") or ("format_errors" if not state.get("final") else "answered")
    return {"final": final, "stop_reason": reason_,
            "messages": [{"role": "user", "content": state["user_message"]},
                         {"role": "assistant", "content": final}]}


# --------------------------------------------------------------------------- edges
def after_reason(state: AgentState) -> str:
    if state.get("final"):
        return "finalize"
    if state.get("pending"):
        return "act" if state["step_count"] < MAX_STEPS else "force_answer"
    if state["format_errors"] > MAX_FORMAT_RETRIES:
        return "finalize"
    return "reason" if state["step_count"] < MAX_STEPS else "force_answer"


def after_act(state: AgentState) -> str:
    return "reason" if state["step_count"] < MAX_STEPS else "force_answer"


def build_graph(checkpointer=None):
    graph = StateGraph(AgentState)
    for name, fn in (("prepare", prepare), ("reason", reason), ("act", act), ("force_answer", force_answer),
                     ("finalize", finalize)):
        graph.add_node(name, fn)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "reason")
    graph.add_conditional_edges("reason", after_reason,
                                {"act": "act", "reason": "reason", "finalize": "finalize",
                                 "force_answer": "force_answer"})
    graph.add_conditional_edges("act", after_act, {"reason": "reason", "force_answer": "force_answer"})
    graph.add_edge("force_answer", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer if checkpointer is not None else MemorySaver())


def run_turn(agent, message: str, thread_id: str = "default", persona: str = "auto",
             variant: PromptVariant = DEFAULT_VARIANT, config_override: dict | None = None,
             use_llm_router: bool = True, client=None) -> dict:
    """Answer one visitor message. Returns the final state (final answer, persona, trace, usage...)."""
    payload = {"user_message": message, "persona": persona, "variant": variant.__dict__,
               "config_override": config_override or {}, "use_llm_router": use_llm_router}
    cfg = {"configurable": {"thread_id": thread_id, "client": client}, "recursion_limit": 4 * MAX_STEPS + 10}
    return agent.invoke(payload, cfg)

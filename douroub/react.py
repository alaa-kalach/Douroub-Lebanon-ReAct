"""The ReAct text protocol: building the prompt, and parsing each model step into an action or a final answer.

The model writes plain text in the format the instructions define:

    Thought: ...
    Action: <tool name>
    Action Input: {"json": "arguments"}

or `Final Answer: ...`. `parse_step()` is the "logic to detect when the model suggests an action": it finds the
action and its arguments, the final answer, or a format problem, and it cuts off any Observation the model
tried to write itself (the newer OpenAI models do not accept a stop sequence, so the parser must do this).
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from datetime import date

from .core import load_summary
from .instructions import build_instructions
from .personas import Persona
from .tools import TOOL_FUNCTIONS, tool_catalog_text


# --------------------------------------------------------------------------- prompt variants
@dataclass(frozen=True)
class PromptVariant:
    """How the prompt is assembled (the prompt-engineering experiments).

    thought:      True = a "Thought:" line before every step (chain of thought); False = actions only.
    shots:        "zero" (no examples), "one" (the worked example, default), "few" (worked example + 3 more).
    placement:    "system" = persona and instructions in the system message; "message" = the system message
                  holds only the persona, and the instructions travel inside the visitor's message.
    instructions: False drops the instructions block entirely (to measure what it changes).
    rules:        which version of the rules (1 = as first written, 2 = with two rules added after the experiments).
    """
    thought: bool = True
    shots: str = "one"
    placement: str = "system"
    instructions: bool = True
    rules: int = 2                 # instructions version: 1 = before the experiment-driven fixes, 2 = after

    def short(self) -> str:
        bits = ["CoT" if self.thought else "no-CoT", f"{self.shots}-shot", f"in-{self.placement}"]
        if not self.instructions:
            bits.append("no-instructions")
        elif self.rules != 2:
            bits.append(f"rules v{self.rules}")
        return " / ".join(bits)


DEFAULT_VARIANT = PromptVariant()


def _context_block(today: date | None) -> str:
    today = today or date.today()
    return f"# CONTEXT\nToday's date is {today:%A, %d %B %Y}."


def _instructions_block(variant: PromptVariant) -> str:
    if not variant.instructions:
        # Minimal protocol only, so the loop can still run: no rules, no examples.
        return ("# INSTRUCTIONS\nUse the tools when needed. Reply with 'Action: <tool>' and 'Action Input: <JSON>', "
                "or with 'Final Answer: <reply>'.")
    return "# INSTRUCTIONS\n" + build_instructions(with_thought=variant.thought, shots=variant.shots,
                                                       version=variant.rules)


def build_prompt(persona: Persona, variant: PromptVariant = DEFAULT_VARIANT, today: date | None = None,
                 memory: str = "") -> tuple:
    """Return (system_message, message_prefix).

    With placement="system" everything is in the system message and the prefix is empty. With
    placement="message" the system message holds only the persona and the rest is prefixed to the visitor's
    message, which is the "system prompt vs in-message instructions" experiment."""
    persona_block = f"# PERSONA: {persona.name}, {persona.title}\n{persona.prompt}"
    rest = [_instructions_block(variant), "# TOOLS\n" + tool_catalog_text(), _context_block(today)]
    if memory:
        rest.append("# CONVERSATION MEMORY\n" + memory)
    rest.append("# BUSINESS KNOWLEDGE\n" + load_summary())
    if variant.placement == "message":
        return persona_block, "\n\n".join(rest) + "\n\n# VISITOR MESSAGE\n"
    return "\n\n".join([persona_block] + rest), ""


# --------------------------------------------------------------------------- parsing a step
@dataclass
class Step:
    """One parsed model step."""
    kind: str                      # "action", "final", or "invalid"
    thought: str = ""
    tool: str = ""
    args: dict = field(default_factory=dict)
    final: str = ""
    error: str = ""                # why the step is invalid (sent back to the model as an observation)
    invented_observation: bool = False   # the model wrote its own "Observation:" (cut off and ignored)
    raw: str = ""


_LABEL = r"(?im)^\s*\**\s*{}\s*\**\s*:\s*\**\s*"  # also **Action:** and **Action**:
THOUGHT_RE = re.compile(_LABEL.format("thought"))
ACTION_RE = re.compile(_LABEL.format("action"))
INPUT_RE = re.compile(_LABEL.format(r"action\s*input"))
FINAL_RE = re.compile(_LABEL.format(r"final\s*answer"))
OBSERVATION_RE = re.compile(_LABEL.format("observation"))
FINAL_ALIASES = {"final answer", "finish", "final", "answer", "none", "respond"}


def _parse_json_args(text: str) -> tuple:
    """(args, error). Accepts a JSON object, also inside ``` fences, or a Python-style dict with single quotes."""
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start == -1:
        return ({}, "") if text in ("", "{}", "none", "None") else ({}, "Action Input must be a JSON object")
    depth, end = 0, None
    for i, ch in enumerate(text[start:], start):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            end = i + 1
            break
    if end is None:
        return {}, "Action Input JSON is not closed with '}'"
    blob = text[start:end]
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(blob)
            if isinstance(value, dict):
                return value, ""
        except (ValueError, SyntaxError, TypeError):
            continue
    return {}, "Action Input is not valid JSON"


def _section(text: str, start_re, stop_res) -> str:
    """Text after the first match of start_re, up to the next label in stop_res."""
    m = start_re.search(text)
    if not m:
        return ""
    rest = text[m.end():]
    cut = min((r.search(rest).start() for r in stop_res if r.search(rest)), default=len(rest))
    return rest[:cut].strip()


def parse_step(text: str) -> Step:
    """Turn one model reply into an action, a final answer, or an invalid step with a fixable error."""
    raw = text or ""
    step = Step(kind="invalid", raw=raw)
    obs = OBSERVATION_RE.search(raw)
    if obs:  # everything from an invented Observation onwards is ignored
        step.invented_observation = True
        raw = raw[:obs.start()]
    step.thought = _section(raw, THOUGHT_RE, [ACTION_RE, FINAL_RE, INPUT_RE])

    action_m, final_m = ACTION_RE.search(raw), FINAL_RE.search(raw)
    if action_m and (not final_m or action_m.start() < final_m.start()):
        line = raw[action_m.end():].split("\n", 1)[0].strip().strip("`*").strip()
        inline = re.match(r"^([A-Za-z_][\w\-]*)\s*\((.*)\)\s*$", line, re.S)  # Action: tool({...})
        tool = (inline.group(1) if inline else line).strip().lower().replace(" ", "_").replace("-", "_")
        if tool.replace("_", " ") in FINAL_ALIASES:   # "Action: Final Answer"
            final = _section(raw, INPUT_RE, [ACTION_RE]) or (final_m and raw[final_m.end():].strip()) or ""
            if final:
                step.kind, step.final = "final", final.strip().strip('"')
                return step
        if tool not in TOOL_FUNCTIONS:
            step.error = (f"Unknown tool '{line[:40]}'. Use exactly one of: {', '.join(TOOL_FUNCTIONS)}, "
                          "or reply with 'Final Answer: ...'.")
            return step
        args_text = inline.group(2) if inline else _section(raw, INPUT_RE, [ACTION_RE, FINAL_RE, THOUGHT_RE])
        args, error = _parse_json_args(args_text)
        if error:
            step.error = f"{error}. Write 'Action Input:' followed by one JSON object on one line."
            return step
        step.kind, step.tool, step.args = "action", tool, args
        return step

    if final_m:
        step.kind, step.final = "final", raw[final_m.end():].strip()
        if not step.final:
            step.kind, step.error = "invalid", "The Final Answer is empty."
        return step

    if raw.strip() and not THOUGHT_RE.search(raw):
        # Plain text with no markers: the model answered directly. Accept it, but flag the format slip.
        step.kind, step.final, step.error = "final", raw.strip(), "format: answered without 'Final Answer:'"
        return step
    step.error = ("No Action or Final Answer found. Reply with 'Action:' and 'Action Input:', "
                  "or with 'Final Answer:'.")
    return step


def format_observation(result: dict, limit: int = 6000) -> str:
    """The tool result as the model sees it: compact JSON, cut at `limit` characters."""
    text = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    if len(text) > limit:
        text = text[:limit] + '..." (truncated)'
    return "Observation: " + text

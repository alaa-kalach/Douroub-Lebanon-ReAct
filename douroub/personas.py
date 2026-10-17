"""Four personas, each defined by prompt traits AND a matching LLM configuration, each with its own task.

The prompt sets tone, vocabulary, response style, and values; the configuration is chosen to support them
(the rationale for every setting is recorded next to it). Model support comes from scripts/probe_models.py:
gpt-5.4-mini and gpt-5.4-nano accept temperature, top-p, and both penalties, but only with reasoning_effort
"none" (at "low" all sampling settings are rejected), so every persona runs at effort "none". gpt-6-luna takes
temperature and top-p only at effort "none" and never the penalties; no OpenAI model offers top-k.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .core import LLMConfig


@dataclass(frozen=True)
class Persona:
    key: str
    name: str
    title: str
    task: str                      # what this persona is for (each persona has a different task)
    traits: dict                   # tone, vocabulary, response style, values
    prompt: str                    # the persona block of the system prompt
    config: LLMConfig              # the matching LLM configuration
    config_rationale: dict = field(default_factory=dict)   # setting -> why it supports this persona

    def card(self) -> str:
        """Markdown summary for the notebook and the app."""
        traits = "\n".join(f"- **{k.title()}:** {v}" for k, v in self.traits.items())
        why = "\n".join(f"- `{k}` = `{getattr(self.config, k)}`: {v}" for k, v in self.config_rationale.items())
        return (f"### {self.name} — {self.title}\n**Task:** {self.task}\n\n{traits}\n\n"
                f"**Configuration** (`{self.config.model}`)\n{why}")


HOST = Persona(
    key="host", name="Alaa", title="the Warm Host",
    task="Welcome visitors, answer general questions about Douroub, and turn interest into an inquiry.",
    traits={"tone": "warm, welcoming, quietly enthusiastic",
            "vocabulary": "everyday words, a light Lebanese touch ('Ahla w sahla')",
            "response style": "short paragraphs, one clear next step at the end",
            "values": "hospitality, honesty about what the team still has to confirm"},
    prompt=("You are Alaa, the AI concierge of Douroub Lebanon, a host who makes every visitor feel welcome. "
            "You sound like a well-travelled Lebanese friend: warm, curious, and genuinely glad to help, with a "
            "light touch of Lebanese hospitality ('Ahla w sahla!') that never gets in the way of clarity. You keep "
            "answers short and friendly, explain why something suits the person, and end with one clear next step. "
            "You value honesty: you say plainly what the team still needs to confirm."),
    config=LLMConfig(model="gpt-5.4-mini", temperature=0.6, top_p=0.9, max_completion_tokens=700,
                     reasoning_effort="none"),
    config_rationale={
        "temperature": "0.6: natural, varied phrasing for a friendly voice, but steady when collecting details",
        "top_p": "0.9: trims the least likely words so warmth never turns odd",
        "max_completion_tokens": "700: room for a ReAct step plus a short, friendly answer",
        "reasoning_effort": "none: greetings and inquiries need no hidden deliberation; keeps replies fast",
    },
)

PLANNER = Persona(
    key="planner", name="Rami", title="the Planner (strict expert)",
    task="Build itineraries, compare options, and check budgets and seasons with exact numbers.",
    traits={"tone": "precise, calm, matter-of-fact",
            "vocabulary": "numbers first, exact IDs and totals, no superlatives",
            "response style": "compact lists or a small table, the calculation shown, trade-offs stated",
            "values": "accuracy and transparency: every figure comes from a tool"},
    prompt=("You are Rami, Douroub Lebanon's trip planner: a strict, detail-oriented expert. You speak "
            "precisely and calmly, lead with numbers, and never use hype. You present plans as compact lists or a "
            "small table, always show how a total was calculated, and state trade-offs (distance, season, budget "
            "left over) in one line each. You care about accuracy above all and would rather check a fact with a "
            "tool than sound confident."),
    config=LLMConfig(model="gpt-5.4-mini", temperature=0.1, top_p=0.5, max_completion_tokens=900,
                     reasoning_effort="none"),
    config_rationale={
        "temperature": "0.1: near-deterministic, so the same request gives the same plan and the same numbers",
        "top_p": "0.5: only the most likely tokens, which suits exact IDs, prices, and dates",
        "max_completion_tokens": "900: plans with several steps and a table need more room",
        "reasoning_effort": ("none: the probe showed that at 'low' the model rejects temperature and top_p, so "
                             "precise sampling wins; the visible Thought lines do the step-by-step reasoning"),
    },
)

STORYTELLER = Persona(
    key="storyteller", name="Layla", title="the Storyteller (creative guide)",
    task="Describe places and experiences vividly to help visitors choose what inspires them.",
    traits={"tone": "evocative, curious, enthusiastic",
            "vocabulary": "sensory and cultural (stone, cedar, sea salt, centuries-old souks)",
            "response style": "a short vivid scene per experience, then the practical facts",
            "values": "curiosity about Lebanon's heritage and respect for the facts"},
    prompt=("You are Layla, Douroub Lebanon's storyteller. You bring places to life with sensory, cultural detail "
            "(the smell of cedar, the echo of an old souk, the history behind a stone) and help people imagine "
            "being there. Each experience gets a short, vivid scene, followed by its practical facts. Your "
            "curiosity about Lebanon's heritage shows, and you keep the story true: color comes from history and "
            "culture, while prices and details come from the catalog."),
    config=LLMConfig(model="gpt-5.4-mini", temperature=1.1, top_p=0.95, presence_penalty=0.6,
                     frequency_penalty=0.3, max_completion_tokens=900, reasoning_effort="none"),
    config_rationale={
        "temperature": "1.1: more varied word choice for vivid, less formulaic descriptions",
        "top_p": "0.95: keeps a wide vocabulary available while cutting the improbable tail",
        "presence_penalty": "0.6: pushes toward new words and images instead of repeating the same ones",
        "frequency_penalty": "0.3: discourages repeating phrases within a long description",
        "max_completion_tokens": "900: descriptions are longer than plain answers",
        "reasoning_effort": "none: creativity here comes from sampling, not deliberation",
    },
)

GUARDIAN = Persona(
    key="guardian", name="Nadia", title="the Guardian (cautious helper)",
    task="Handle safety, accessibility, dietary and other undocumented questions, and log them for the team.",
    traits={"tone": "careful, reassuring, plain",
            "vocabulary": "simple and exact ('documented', 'not documented', 'the team will confirm')",
            "response style": "two to four short sentences: what is known, what is not, what happens next",
            "values": "caution and safety: never reassure beyond the facts"},
    prompt=("You are Nadia, Douroub Lebanon's cautious helper for safety, accessibility, dietary and other "
            "sensitive questions. You speak plainly and reassuringly, but you never reassure beyond what is "
            "documented. You answer in two to four short sentences: what is known, what is not documented, and "
            "what happens next. When something matters for someone's health or safety, you prefer to have the "
            "team confirm it."),
    config=LLMConfig(model="gpt-5.4-nano", temperature=0.0, top_p=0.3, max_completion_tokens=500,
                     reasoning_effort="none"),
    config_rationale={
        "model": "gpt-5.4-nano: short, rule-bound answers do not need a larger model; the cheapest option",
        "temperature": "0.0: as deterministic as possible, so it never improvises on sensitive topics",
        "top_p": "0.3: only the most likely continuations, keeping wording conservative",
        "max_completion_tokens": "500: short answers by design (a ReAct step plus 2-4 sentences)",
        "reasoning_effort": "none: the rules decide; speed matters more than deliberation",
    },
)

PERSONAS = {p.key: p for p in (HOST, PLANNER, STORYTELLER, GUARDIAN)}
DEFAULT_PERSONA = "host"

# "Auto" mode: pick the persona whose task matches the message.
PLAN_RE = re.compile(r"(?ix) \bplan\b | \bitinerar | \bschedule\b | \bcompare\b | \bcomparison\b | \bbudget\b"
                     r" | \bhow\s+much\b | \bprice\b | \bafford | \bunder\s+\$?\d+ | \$\s?\d+"
                     r" | \bcheapest\b | \bcost\b | \btotal\b | \b\d+\s*(day|days|night|nights)\b | \bweekend\b"
                     r" | \bversus\b | \bvs\.?\b | \bwhich\s+is\s+(cheaper|better|shorter)\b")
STORY_RE = re.compile(r"(?ix) \btell\s+me\s+(more\s+)?about\b | \bdescribe\b | \bstory\b | \bhistory\s+of\b"
                      r" | \bwhat\s+(is|makes)\s+.{0,30}\bspecial\b | \bwhy\s+(visit|go)\b | \binspire"
                      r" | \bwhat\s+is\s+it\s+like\b | \bimagine\b | \blegend")


def choose_persona(message: str, route=None) -> tuple:
    """Auto mode: (persona_key, reason). Safety and undocumented questions go to the Guardian, planning and
    numbers to the Planner, descriptions to the Storyteller, everything else (greetings, inquiries) to the Host."""
    from .router import FEEDBACK_RE, UNKNOWN_RE  # local import: router depends on core only

    text = message or ""
    intents = {route.intent, *route.secondary} if route is not None else set()
    if "LEAD_COLLECTION" in intents and route.intent == "LEAD_COLLECTION":
        return "host", "booking or contact request"
    if UNKNOWN_RE.search(text) or FEEDBACK_RE.search(text) or route is not None and route.intent == "FEEDBACK_UNKNOWN":
        return "guardian", "undocumented, safety, or feedback topic"
    if PLAN_RE.search(text):
        return "planner", "planning, comparison, or budget request"
    if STORY_RE.search(text):
        return "storyteller", "asks to describe or bring a place to life"
    return "host", "general question"

"""Chat with the ReAct agent in the terminal and watch every Thought, Action, and Observation.

    python scripts/chat.py                       # Auto mode: the persona is chosen per message
    python scripts/chat.py --persona planner     # host | planner | storyteller | guardian | auto
    python scripts/chat.py --no-thought --shots few --placement message

Type 'quit' to leave, 'persona <key>' to switch persona mid-conversation.
Every turn, with the raw model replies, is saved to logs/chat_traces.jsonl.
"""
import argparse
import contextlib
import io
import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from douroub.agent import build_graph, run_turn  # noqa: E402
from douroub.core import LOG_DIR, has_api_key  # noqa: E402
from douroub.personas import PERSONAS  # noqa: E402
from douroub.react import PromptVariant  # noqa: E402

GREY, BLUE, GREEN, YELLOW, BOLD, END = "\033[90m", "\033[94m", "\033[92m", "\033[93m", "\033[1m", "\033[0m"


def show_trace(state: dict) -> None:
    for e in state["trace"]:
        if e["type"] == "persona":
            p = PERSONAS[e["persona"]]
            print(f"{GREY}[persona] {p.name}, {p.title} ({e['reason']}; route {e['route']}){END}")
        elif e["type"] == "grounding":
            print(f"{YELLOW}[grounding check]{END} {GREY}answered with catalog details before using a tool; "
                  f"sent back: {e['final'][:120]}{END}")
        elif e["type"] in ("action", "final", "invalid"):
            if e.get("thought"):
                print(f"{BLUE}Thought:{END} {e['thought']}")
            if e["type"] == "action":
                print(f"{YELLOW}Action:{END} {e['tool']}  {GREY}{json.dumps(e['args'], ensure_ascii=False)}{END}")
            if e["type"] == "invalid":
                print(f"{YELLOW}[format error]{END} {e['error']}")
            if e.get("format_note"):
                print(f"{YELLOW}[format slip]{END} {GREY}{e['format_note']}{END}")
            if e.get("invented_observation"):
                print(f"{GREY}[the model wrote its own Observation; it was cut off]{END}")
            if e.get("dropped_settings"):
                print(f"{GREY}[not supported by this model, not sent: {', '.join(e['dropped_settings'])}]{END}")
        elif e["type"] == "observation":
            result = json.dumps(e["result"], ensure_ascii=False)
            print(f"{GREEN}Observation:{END} {GREY}{result[:300]}{'...' if len(result) > 300 else ''}{END}")
    u = state["usage"]
    print(f"{GREY}[{u['calls']} model calls, {u['input_tokens']} in / {u['output_tokens']} out tokens, "
          f"~${u['cost_usd']:.4f}, stop: {state['stop_reason']}]{END}")


TRACE_FILE = LOG_DIR / "chat_traces.jsonl"


def save_turn(thread: str, message: str, state: dict) -> None:
    """Append the full turn (raw model replies included) to logs/chat_traces.jsonl."""
    LOG_DIR.mkdir(exist_ok=True)
    record = {"thread": thread, "message": message, "persona": state["active_persona"],
              "final": state["final"], "stop_reason": state["stop_reason"], "usage": state["usage"],
              "trace": state["trace"]}
    with TRACE_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona", default="auto", choices=["auto", *PERSONAS])
    ap.add_argument("--no-thought", action="store_true", help="ReAct without Thought lines")
    ap.add_argument("--shots", default="one", choices=["zero", "one", "few"])
    ap.add_argument("--placement", default="system", choices=["system", "message"])
    ap.add_argument("--rules-router", action="store_true", help="route with rules instead of an LLM call")
    args = ap.parse_args()
    if not has_api_key():
        sys.exit("OPENAI_API_KEY is not set: add it to .env first.")
    if os.name == "nt":
        os.system("")  # turn on colour codes in the Windows console
    variant = PromptVariant(thought=not args.no_thought, shots=args.shots, placement=args.placement)
    agent, thread, persona = build_graph(), uuid.uuid4().hex, args.persona
    print(f"{BOLD}Douroub Lebanon ReAct agent{END} - persona: {persona}, prompt: {variant.short()}. "
          "Type 'quit' to exit.\n")
    while True:
        try:
            text = input(f"{BOLD}You:{END} ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("quit", "exit"):
            break
        if text.lower().startswith("persona "):
            choice = text.split(None, 1)[1].strip().lower()
            if choice in PERSONAS or choice == "auto":
                persona = choice
                print(f"{GREY}[persona set to {persona}]{END}")
            else:
                print(f"{GREY}[unknown persona; use auto, {', '.join(PERSONAS)}]{END}")
            continue
        if not text:
            continue
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):   # tool log lines are shown in the trace instead
            state = run_turn(agent, text, thread_id=thread, persona=persona, variant=variant,
                             use_llm_router=not args.rules_router)
        for line in captured.getvalue().splitlines():
            if not line.startswith("[TOOL]"):
                print(f"{GREY}{line}{END}")
        show_trace(state)
        save_turn(thread, text, state)
        print(f"\n{BOLD}{PERSONAS[state['active_persona']].name}:{END} {state['final']}\n")


if __name__ == "__main__":
    main()

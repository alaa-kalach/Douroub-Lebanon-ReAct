"""The Gradio app: chat with the ReAct agent (Auto mode or a chosen persona), watch every Thought, Action and
Observation as it happens, compare the four personas on one question, meet the personas, and see how it works.
Experiment results are not shown in the app: they are in the PDF report (docs/experiment_report.pdf), which the
"How it works" page offers as a download.

Run it with `python app.py` (or `python -m douroub.ui`).
"""
from __future__ import annotations

import json
import time
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

warnings.filterwarnings("ignore", message="IProgress not found")
import gradio as gr  # noqa: E402

from .agent import build_graph, run_turn, stream_turn  # noqa: E402
from .core import ROOT, QuotaExhausted, has_api_key  # noqa: E402
from .instructions import build_instructions  # noqa: E402
from .personas import PERSONAS, choose_persona  # noqa: E402
from .react import PromptVariant  # noqa: E402
from .router import route_with_rules  # noqa: E402
from .tools import TOOL_SPECS  # noqa: E402
from .ui_assets import CEDAR_PATH, HEAD, ICON_PATHS, THEME  # noqa: E402

APP_TITLE = "Douroub Lebanon: four specialists, one ReAct agent"
MAX_MESSAGE_CHARS = 2000
FALLBACK = ("Sorry, something went wrong on my side. Please try again in a moment, or email the team and Khaled "
            "will help you directly.")
EXAMPLES = [
    "How much would D04 and D08 cost for 3 of us, and do both run in November?",
    "Tell me about Byblos",
    "Is D10 wheelchair accessible?",
    "Plan a 2-day trip in the Chouf for 2 people, under $100 each.",
    "Can we do the Batroun sunset kayak in December?",
    "Quelles expériences culinaires proposez-vous à Beyrouth ?",
]
PERSONA_CHOICES = [("Auto (picks per message)", "auto")] + [
    (f"{p.name} · {p.title.replace('the ', '').split(' (')[0]}", key) for key, p in PERSONAS.items()]

# The agent graph is compiled once; its checkpointer keeps each browser session's conversation by thread_id.
AGENT = build_graph()

# --------------------------------------------------------------------------- trace panels
EXTRA_ICONS = {
    "persona": '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "thought": ('<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 '
                '1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/>'),
    "budget_calculator": ('<rect width="16" height="20" x="4" y="2" rx="2"/><line x1="8" x2="16" y1="6" y2="6"/>'
                          '<path d="M16 10h.01"/><path d="M12 10h.01"/><path d="M8 10h.01"/><path d="M12 14h.01"/>'
                          '<path d="M8 14h.01"/><path d="M12 18h.01"/><path d="M8 18h.01"/>'
                          '<line x1="16" x2="16" y1="14" y2="18"/>'),
    "season_check": '<path d="M8 2v4"/><path d="M16 2v4"/><rect width="18" height="18" x="3" y="4" rx="2"/><path d="M3 10h18"/>',
}
ICONS = {**ICON_PATHS, **EXTRA_ICONS}
TOOL_TITLES = {"search_experiences": "searched the catalog", "compare_experiences": "compared experiences",
               "create_itinerary": "planned an itinerary", "record_customer_interest": "saved the inquiry",
               "record_feedback": "logged it for the team", "budget_calculator": "calculated the total",
               "season_check": "checked the season"}


def icon(name: str, tone: str = "") -> str:
    """A classed span (Gradio's sanitiser keeps it); the CSS draws the icon as a mask so it follows the theme."""
    return f'<span class="db-ico db-ico-{name}{" db-ico-" + tone if tone else ""}" aria-hidden="true"></span>'


def _icon_css() -> str:
    rules = []
    for name, paths in ICONS.items():
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" '
               f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{paths}</svg>')
        uri = "data:image/svg+xml," + quote(svg)
        rules.append(f".db-ico-{name} {{ -webkit-mask-image: url('{uri}'); mask-image: url('{uri}'); }}")
    return "\n".join(rules)


def _short(result: dict, limit: int = 700) -> str:
    text = json.dumps(result, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + " …"


def _panel(title: str, content: str, pending: bool = False, log: str = "", duration: float | None = None) -> dict:
    meta = {"title": title, "status": "pending" if pending else "done"}
    if log:
        meta["log"] = log
    if duration and duration >= 0.05:
        meta["duration"] = round(float(duration), 2)
    return {"role": "assistant", "content": content or " ", "metadata": meta}


def trace_panels(state: dict, done: bool, style: str = "") -> list:
    """The ReAct trace as collapsible chat panels: who answers (and with which prompt style), then one panel
    per step."""
    trace = state.get("trace") or []
    panels, step, last_action = [], 0, None
    for e in trace:
        t = e["type"]
        if t == "persona":
            p = PERSONAS[e["persona"]]
            how = "picked by Auto mode" if e["reason"] != "chosen by the user" else "chosen in the settings"
            panels.append(_panel(f"{icon('persona')}{p.name}, {p.title} is answering",
                                 f"{how}: {e['reason']} (intent: `{e['route']}`)\n\n*{p.config.short()}*"
                                 + (f"\n\n**Prompt style:** {style}" if style else ""),
                                 log=p.config.model))
        elif t == "action":
            step += 1
            body = (f"**Thought:** {e['thought']}\n\n" if e.get("thought") else "")
            body += f"**Action:** `{e['tool']}`\n\n**Action Input:** `{json.dumps(e['args'], ensure_ascii=False)}`"
            name = e["tool"] if e["tool"] in ICONS else "search_experiences"
            last_action = _panel(f"{icon(name)}Step {step}: {TOOL_TITLES.get(e['tool'], e['tool'])}", body,
                                 pending=True, log=e["tool"], duration=e.get("latency_s"))
            panels.append(last_action)
        elif t == "observation" and last_action is not None:
            ok = e.get("status") == "success"
            last_action["content"] += f"\n\n**Observation:** `{_short(e['result'])}`"
            last_action["metadata"]["status"] = "done"
            if not ok:
                err = e["result"].get("error", "error")
                last_action["metadata"]["title"] = (f"{icon('approved')}Step {step}: stopped by a safety check"
                                                    if err in ("details_not_from_visitor", "not_feedback")
                                                    else f"{icon('error', 'warn')}Step {step}: tool error ({err})")
            last_action = None
        elif t == "invalid":
            step += 1
            panels.append(_panel(f"{icon('error', 'warn')}Step {step}: format error, sent back to the model",
                                 f"{e['error']}\n\n**Model wrote:** {e.get('raw', '')[:400]}"))
        elif t == "grounding":
            step += 1
            panels.append(_panel(f"{icon('approved')}Step {step}: grounding check, look it up first",
                                 "The draft quoted catalog details before any tool ran, so it was sent back:\n\n> "
                                 + e["final"][:500].replace("\n", "\n> ")))
        elif t == "final" and (e.get("thought") or e.get("format_note") or e.get("forced")):
            step += 1
            body = (f"**Thought:** {e['thought']}" if e.get("thought") else "")
            if e.get("format_note"):
                body += f"\n\n*Format slip: {e['format_note']}*"
            if e.get("forced"):
                body += "\n\n*Step limit reached: answered from the observations so far.*"
            panels.append(_panel(f"{icon('thought')}Step {step}: ready to answer", body))
    if not done:
        panels.append(_panel(f"{icon('thought')}Thinking…", "Waiting for the model.", pending=True))
    elif state.get("usage"):
        u = state["usage"]
        if panels:
            panels[0]["metadata"]["log"] = (f"{u['calls']} model calls · {u['input_tokens'] + u['output_tokens']:,} "
                                            f"tokens · ${u['cost_usd']:.4f}")
    return panels


# --------------------------------------------------------------------------- chat
SHOT_NAMES = {"zero": "zero-shot", "one": "one-shot", "few": "few-shot"}


def _variant(thought: bool, shots: str, placement: str = "system") -> PromptVariant:
    return PromptVariant(thought=bool(thought), shots=shots or "one", placement=placement or "system")


def style_label(variant: PromptVariant) -> str:
    """'chain of thought · one-shot · instructions in the system prompt', for the trace and the compare tab."""
    return " · ".join(["chain of thought" if variant.thought else "no chain of thought",
                       SHOT_NAMES.get(variant.shots, variant.shots),
                       "instructions in the system prompt" if variant.placement == "system"
                       else "instructions inside the message"])


BASELINE = PromptVariant()   # chain of thought, one-shot, instructions in the system prompt
BASELINE_NOTE = ("Every reply uses the default style: **with** chain of thought, **one-shot**, instructions in the "
                 "**system prompt**. To test a style, change **one** of them; changing one puts the others back to "
                 "their default.")
STYLE_NAMES = {"cot": "chain of thought", "shots": "examples", "place": "instruction placement"}


def effective_style(cot_on, cot_val, shots_on, shots_val, place_on, place_val) -> tuple:
    """(thought, shots, placement): the baseline, with only the switched-on style changed."""
    thought = bool(cot_val) if cot_on else BASELINE.thought
    shots = (shots_val or BASELINE.shots) if shots_on else BASELINE.shots
    placement = (place_val or BASELINE.placement) if place_on else BASELINE.placement
    return thought, shots, placement


def style_status(active, variant) -> str:
    if not active or variant is None:
        return "**Now:** the default style (nothing changed)."
    return f"**Now testing {STYLE_NAMES[active]}:** {style_label(variant)}."


def switch_style(which: str, on: bool, cot_val, shots_val, place_val):
    """One style on at a time: switching one on turns the other two off and shows only its choices."""
    states = {k: (on and k == which) for k in ("cot", "shots", "place")}
    t, sh, pl = effective_style(states["cot"], cot_val, states["shots"], shots_val, states["place"], place_val)
    active = which if on else None
    return (gr.update(value=states["cot"]), gr.update(visible=states["cot"]),
            gr.update(value=states["shots"]), gr.update(visible=states["shots"]),
            gr.update(value=states["place"]), gr.update(visible=states["place"]),
            t, sh, pl, style_status(active, _variant(t, sh, pl)))


def change_choice(cot_on, cot_val, shots_on, shots_val, place_on, place_val):
    """A different choice for the active style."""
    t, sh, pl = effective_style(cot_on, cot_val, shots_on, shots_val, place_on, place_val)
    active = "cot" if cot_on else "shots" if shots_on else "place" if place_on else None
    return t, sh, pl, style_status(active, _variant(t, sh, pl))


def chat(message, history, thread, persona, thought, shots, placement, show_steps):
    """Generator: shows the visitor's message, then the ReAct steps as they happen, then the answer."""
    text = (message or "").strip()
    thread = thread or uuid.uuid4().hex
    history = list(history or [])
    if not text:
        yield history, "", thread
        return
    history.append({"role": "user", "content": text})
    if len(text) > MAX_MESSAGE_CHARS:
        history.append({"role": "assistant", "content": f"Could you shorten that to under {MAX_MESSAGE_CHARS} characters?"})
        yield history, "", thread
        return
    if not has_api_key():
        history.append({"role": "assistant", "content": "The assistant is not configured yet (OPENAI_API_KEY is missing)."})
        yield history, "", thread
        return
    yield history + [_panel(f"{icon('thought')}Thinking…", "Choosing a persona.", pending=True)], "", thread
    variant = _variant(thought, shots, placement)
    try:
        for node, state in stream_turn(AGENT, text, thread_id=thread, persona=persona or "auto", variant=variant):
            done = node == "done"
            panels = trace_panels(state, done, style_label(variant)) if show_steps else ([] if done else [
                _panel(f"{icon('thought')}Thinking…", " ", pending=True)])
            final = [{"role": "assistant", "content": state["final"]}] if done else []
            yield history + panels + final, "", thread
    except QuotaExhausted:
        yield history + [{"role": "assistant", "content": "The assistant has reached its usage limit for now. "
                                                           "Please try again later."}], "", thread
    except Exception as exc:  # the visitor always gets an answer
        print(f"[APP] {type(exc).__name__}: {exc}")
        yield history + [{"role": "assistant", "content": FALLBACK}], "", thread


def new_thread():
    return uuid.uuid4().hex


# --------------------------------------------------------------------------- compare personas
def compare(question, thought, shots, placement="system"):
    """All four personas answer the same question in parallel (separate conversations)."""
    text = (question or "").strip()[:MAX_MESSAGE_CHARS]
    keys = list(PERSONAS)
    if not text:
        yield ["*Type a question above.*"] * len(keys) + [""]
        return
    auto = PERSONAS[choose_persona(text, route_with_rules(text))[0]]
    variant = _variant(thought, shots, placement)
    note = (f"Auto mode would send this to **{auto.name}, {auto.title}**. Prompt style for all four: "
            f"{style_label(variant)} (change it in the Chat tab's settings).")
    if not has_api_key():
        yield ["*OPENAI_API_KEY is missing.*"] * len(keys) + [note]
        return
    yield ["*Thinking…*"] * len(keys) + [note]

    def one(key):
        start = time.perf_counter()
        try:
            state = run_turn(AGENT, text, thread_id=f"compare-{uuid.uuid4().hex}", persona=key, variant=variant)
        except Exception as exc:
            return f"*Failed: {type(exc).__name__}*"
        u, wall = state["usage"], time.perf_counter() - start
        tools_used = [e["tool"] for e in state["trace"] if e["type"] == "observation"]
        words = len(state["final"].split())
        stats = (f"{u['calls']} model calls · tools: {', '.join(tools_used) or 'none'} · {words} words · "
                 f"${u['cost_usd']:.4f} · {wall:.1f} s")
        return f"{state['final']}\n\n---\n<small>{stats}</small>"

    with ThreadPoolExecutor(max_workers=len(keys)) as pool:
        answers = list(pool.map(one, keys))
    yield answers + [note]


# --------------------------------------------------------------------------- page
REPORT_PDF = ROOT / "docs" / "experiment_report.pdf"
_CEDAR_URI = ("data:image/svg+xml," + quote(
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 212"><path fill="#1E4D3F" fill-opacity="0.07" d="{CEDAR_PATH}"/></svg>'))

CSS = """
.gradio-container { max-width: 1320px !important; margin: 0 auto !important; }
.gradio-container::after { content: ""; position: fixed; right: -90px; bottom: -40px; width: 480px; height: 420px;
    background: url('CEDAR_URI') no-repeat right bottom / contain; pointer-events: none; z-index: 0; }
.html-container:has(.db-header) { padding: 0 !important; }
/* header */
.db-header { background: #1E4D3F; color: #F4F7F3; border-radius: 16px; overflow: hidden; }
.db-flag { display: flex; flex-direction: column; height: 10px; }
.db-flag span { flex: 1; background: #ED1C24; }
.db-flag span:nth-child(2) { flex: 2; background: #FFFFFF; }
.db-header-body { display: flex; align-items: center; justify-content: space-between; gap: 24px;
                  padding: 14px 24px 16px; flex-wrap: wrap; }
.db-title { font-family: 'Reem Kufi', 'IBM Plex Sans Arabic', sans-serif; font-weight: 600; margin: 0;
            font-size: 28px; line-height: 1.15; color: #FFFFFF; }
.db-title .db-ar { font-size: 22px; color: #CFE1D6; margin-left: 10px; }
.db-lede { margin: 4px 0 0; font-size: 14px; color: #D8E6DD; max-width: 640px; }
.db-chips { display: flex; gap: 8px; flex-wrap: wrap; }
.db-chip { background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14); border-radius: 999px;
           padding: 5px 12px; font-size: 13px; color: #E6F0EA; white-space: nowrap; }
.db-chip b { color: #FFFFFF; }
/* pages */
#db-tabs > .tab-wrapper button, #db-tabs button[role="tab"] { font-size: 15px; font-weight: 500; }
.db-intro { color: #4B5D55; margin: 2px 0 6px; }
.db-small, .db-small p { font-size: 12.5px !important; color: #4B5D55; margin: 0 0 4px; }
.db-section-title { margin: 6px 0 2px; }
#db-chat { height: calc(100vh - 420px) !important; min-height: 360px; }
#db-settings { gap: 10px; padding: 14px 16px !important; }
#db-settings .db-settings-title { margin: 0 0 4px; font-size: 15px; font-weight: 600; }
.message-content p, .message-content li { unicode-bidi: plaintext; text-align: start; }
.db-ico { display: inline-block; width: 16px; height: 16px; margin-right: 7px; vertical-align: -3px;
          background-color: #2F7658; -webkit-mask: no-repeat center / contain; mask: no-repeat center / contain; }
.db-ico.db-ico-warn { background-color: #B5562F; }
/* cards */
.db-card { border: 1px solid #D7E1DA !important; border-radius: 14px !important; background: #FFFFFF; }
.db-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
@media (max-width: 800px) { .db-grid { grid-template-columns: 1fr; } }
.db-persona { border: 1px solid #D7E1DA; border-radius: 14px; padding: 16px 18px; background: #FFFFFF; }
.db-persona h3 { margin: 0; font-size: 18px; }
.db-persona .db-role { margin: 0 0 8px; color: #2F7658; font-weight: 500; }
.db-persona dl { margin: 0; display: grid; grid-template-columns: 92px 1fr; gap: 4px 10px; font-size: 13.5px; }
.db-persona dt { color: #66786F; }
.db-persona dd { margin: 0; }
.db-persona ul { margin: 6px 0 0; padding-left: 18px; font-size: 13px; color: #354640; }
.db-persona code { font-size: 12px; }
.db-steps { counter-reset: step; display: grid; gap: 10px; margin: 0; padding: 0; list-style: none; }
.db-steps li { border: 1px solid #D7E1DA; border-radius: 12px; padding: 10px 14px 10px 48px; position: relative;
               background: #FFFFFF; font-size: 14px; }
.db-steps li::before { counter-increment: step; content: counter(step); position: absolute; left: 14px; top: 10px;
                       width: 22px; height: 22px; border-radius: 50%; background: #1E4D3F; color: #FFF;
                       font-size: 12.5px; display: grid; place-items: center; }
#db-compare-row { gap: 14px; align-items: stretch; }
#db-compare-row .db-card { padding: 6px 14px !important; }
@media (prefers-color-scheme: dark) {
  .db-ico { background-color: #7FB399; }
  .db-ico.db-ico-warn { background-color: #E08A65; }
  .db-card, .db-persona, .db-steps li { background: #152520 !important; border-color: #253A33 !important; }
  .db-persona dt, .db-intro { color: #93A39A; }
  .db-persona ul { color: #C3CEC7; }
}
@media (max-width: 900px) {
  .gradio-container { overflow-x: hidden; }
  .gradio-container::after { display: none; }
  #db-tabs .tab-wrapper, #db-tabs .tab-container { flex-wrap: wrap; min-width: 0; height: auto; }
  #db-tabs .tab-container button { font-size: 14px; padding: 6px 10px; }
  .db-title { font-size: 23px; }
  .db-title .db-ar { font-size: 18px; }
  #db-chat { height: 60vh !important; }
  .db-header-body { padding: 12px 16px; }
}
""".replace("CEDAR_URI", _CEDAR_URI) + "\n" + _icon_css()

HEADER_HTML = f"""
<header class="db-header">
  <div class="db-flag" aria-hidden="true"><span></span><span></span><span></span></div>
  <div class="db-header-body">
    <div>
      <h1 class="db-title">Douroub Lebanon<span class="db-ar" lang="ar" dir="rtl">دروب لبنان</span></h1>
      <p class="db-lede">Four AI specialists, one reasoning-and-acting loop. Ask about experiences, prices, seasons
      or a trip plan, in English, Arabic or French.</p>
    </div>
    <div class="db-chips">{"".join(f'<span class="db-chip"><b>{p.name}</b> · {p.title.replace("the ", "").split(" (")[0]}</span>' for p in PERSONAS.values())}</div>
  </div>
</header>
"""


def _persona_cards_html() -> str:
    cards = []
    for p in PERSONAS.values():
        traits = "".join(f"<dt>{k.capitalize()}</dt><dd>{v}</dd>" for k, v in p.traits.items())
        why = "".join(f"<li><code>{k}</code> {v.split(': ', 1)[-1] if ': ' in v else v} "
                      f"(<b>{getattr(p.config, k)}</b>)</li>" for k, v in p.config_rationale.items())
        cards.append(f'<article class="db-persona"><h3>{p.name}</h3><p class="db-role">{p.title}</p>'
                     f'<dl><dt>Task</dt><dd>{p.task}</dd>{traits}<dt>Model</dt><dd><code>{p.config.model}</code></dd></dl>'
                     f'<p style="margin:10px 0 0;font-size:13px;font-weight:600">Why these settings</p><ul>{why}</ul>'
                     f'</article>')
    return f'<div class="db-grid">{"".join(cards)}</div>'


LOOP_HTML = """
<ol class="db-steps">
  <li><b>Prepare.</b> The message is classified, Auto mode picks the persona, and the prompt is assembled: persona,
      instructions, tool descriptions, today's date, what was saved so far, and the business knowledge.</li>
  <li><b>Reason.</b> One model call. The model writes <code>Thought</code> and <code>Action</code> /
      <code>Action Input</code>, or a <code>Final Answer</code>, as plain text; a hand-written parser reads it.</li>
  <li><b>Act.</b> Safety checks run first (no invented email, nothing logged that is not real feedback), then the
      tool. Its result goes back to the model as an <code>Observation</code>, and the loop returns to step 2.</li>
  <li><b>Check.</b> A malformed step, or an answer quoting prices before any tool ran, is sent back to the model
      instead of reaching the visitor. After 6 steps the agent must answer from what it has.</li>
  <li><b>Finish.</b> The answer is shown and the conversation is saved, so the next message (to any persona)
      remembers it.</li>
</ol>
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE, fill_height=True) as demo:
        thread = gr.State(None)
        gr.HTML(HEADER_HTML, padding=False)
        with gr.Tabs(elem_id="db-tabs"):
            # ---------------------------------------------------------------- 1. chat
            with gr.Tab("Chat", id="chat"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3, min_width=300):
                        chatbot = gr.Chatbot(show_label=False, elem_id="db-chat", buttons=["copy"],
                                             placeholder="### Ahla w sahla!\nAsk about an experience, a price, a "
                                                         "season, or a plan. Pick an example below to start.")
                        with gr.Row():
                            box = gr.Textbox(placeholder="Ask about an experience, a price, or a plan…",
                                             show_label=False, max_lines=6, scale=8, autofocus=True)
                            send = gr.Button("Send", variant="primary", scale=1, min_width=90)
                        gr.Examples(EXAMPLES, inputs=box, label="Examples")
                    with gr.Column(scale=1, min_width=260, elem_id="db-settings", elem_classes="db-card"):
                        gr.Markdown("Conversation settings", elem_classes="db-settings-title")
                        persona = gr.Radio(PERSONA_CHOICES, value="auto", label="Who answers",
                                           info="Auto picks the right specialist for each message.")
                        show_steps = gr.Checkbox(value=True, label="Show the reasoning steps",
                                                 info="Thought, Action and Observation under each answer.")
                        with gr.Accordion("Prompt style", open=True):
                            gr.Markdown(BASELINE_NOTE, elem_classes="db-small")
                            cot_on = gr.Checkbox(False, label="Change chain of thought (default: with)")
                            cot_val = gr.Radio([("With", True), ("Without", False)], value=False, visible=False,
                                               show_label=False, info="With: a Thought line before every step.")
                            shots_on = gr.Checkbox(False, label="Change the examples (default: one-shot)")
                            shots_val = gr.Radio([("Zero-shot", "zero"), ("One-shot", "one"), ("Few-shot", "few")],
                                                 value="few", visible=False, show_label=False,
                                                 info="No examples, the worked example, or it plus three more.")
                            place_on = gr.Checkbox(False, label="Change where instructions go (default: system prompt)")
                            place_val = gr.Radio([("System prompt", "system"), ("In the message", "message")],
                                                 value="message", visible=False, show_label=False,
                                                 info="Where the instructions block is sent.")
                            style_now = gr.Markdown(style_status(None, None), elem_classes="db-small")
                        thought = gr.State(BASELINE.thought)
                        shots = gr.State(BASELINE.shots)
                        placement = gr.State(BASELINE.placement)
                        clear = gr.Button("New conversation", variant="secondary")
            # ---------------------------------------------------------------- 2. compare
            with gr.Tab("Compare personas", id="compare"):
                gr.Markdown("Ask one question and see how each persona answers it, with its own prompt and "
                            "configuration but the same tools and facts. Each persona answers in a separate "
                            "conversation.", elem_classes="db-intro")
                with gr.Row():
                    question = gr.Textbox(value="Tell me about Byblos", show_label=False, scale=8)
                    go = gr.Button("Ask all four", variant="primary", scale=1, min_width=130)
                gr.Examples(["Tell me about Byblos", "Is D10 wheelchair accessible?",
                             "How much would D04 and D08 cost for 3 of us?", "Hi! What does Douroub do?"],
                            inputs=question, label="Examples")
                auto_note = gr.Markdown()
                outs, people = [], list(PERSONAS.values())
                for pair in (people[:2], people[2:]):
                    with gr.Row(elem_id="db-compare-row", equal_height=True):
                        for p in pair:
                            with gr.Column(min_width=280, elem_classes="db-card"):
                                gr.Markdown(f"#### {p.name}, {p.title}\n<small>{p.config.short()}</small>")
                                outs.append(gr.Markdown("*The answer will appear here.*"))
            # ---------------------------------------------------------------- 3. personas
            with gr.Tab("Meet the personas", id="personas"):
                gr.Markdown("Each persona has its own task, a voice defined in its prompt, and an LLM configuration "
                            "chosen to support that voice. All four follow the same instructions and use the same "
                            "tools.", elem_classes="db-intro")
                gr.HTML(_persona_cards_html())
            # ---------------------------------------------------------------- 4. how it works
            with gr.Tab("How it works", id="how"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        gr.Markdown("### The ReAct loop", elem_classes="db-section-title")
                        gr.HTML(LOOP_HTML)
                        gr.Markdown("### Tools", elem_classes="db-section-title")
                        gr.Markdown("\n".join(f"- **`{t['name']}`**: {t['description'].split('. ')[0]}."
                                              for t in TOOL_SPECS))
                        with gr.Accordion("The instructions block (shared by every persona)", open=False):
                            gr.Code(build_instructions(), language="markdown", show_label=False)
                    with gr.Column(scale=4):
                        gr.Image(str(ROOT / "docs" / "react_graph.png"), show_label=False, interactive=False,
                                 buttons=["fullscreen"])
                        gr.Markdown("### Experiment report", elem_classes="db-section-title")
                        gr.Markdown("297 logged runs compared the personas, prompt styles (chain of thought, "
                                    "examples, instruction placement) and LLM settings (temperature, top-p, "
                                    "penalties, max tokens, model, reasoning effort). The findings, tables, charts "
                                    "and reflection are in the report.")
                        if REPORT_PDF.exists():
                            gr.DownloadButton("Download the report (PDF)", value=str(REPORT_PDF), variant="primary")
                        else:
                            gr.Markdown("*Build it with `python scripts/build_report.py`.*")

        switches = [cot_on, cot_val, shots_on, shots_val, place_on, place_val]
        style_outputs = [cot_on, cot_val, shots_on, shots_val, place_on, place_val, thought, shots, placement, style_now]
        for key, box_ in (("cot", cot_on), ("shots", shots_on), ("place", place_on)):
            box_.input(lambda on, c, sh, pl, k=key: switch_style(k, on, c, sh, pl),
                       [box_, cot_val, shots_val, place_val], style_outputs)
        for radio in (cot_val, shots_val, place_val):
            radio.input(change_choice, switches, [thought, shots, placement, style_now])
        inputs = [box, chatbot, thread, persona, thought, shots, placement, show_steps]
        box.submit(chat, inputs, [chatbot, box, thread])
        send.click(chat, inputs, [chatbot, box, thread])
        clear.click(lambda: ([], new_thread()), None, [chatbot, thread])
        chatbot.clear(new_thread, None, thread)
        go.click(compare, [question, thought, shots, placement], outs + [auto_note])
        question.submit(compare, [question, thought, shots, placement], outs + [auto_note])
    return demo


LAUNCH_OPTIONS = {"theme": THEME, "css": CSS, "head": HEAD, "footer_links": ["gradio"]}

if __name__ == "__main__":
    build_demo().launch(**LAUNCH_OPTIONS)

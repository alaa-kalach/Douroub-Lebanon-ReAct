"""The experiment report as a PDF: method, every comparison (finding, table, chart), the best combination, and the
reflection. Built from the saved results in experiments/results/, so it never calls the API.

    python scripts/build_report.py          # writes docs/experiment_report.pdf
"""
from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from . import experiments as ex
from . import report
from .core import ROOT
from .evaluation import CORE_TASKS, TASKS
from .instructions import RULES_V1, RULES_V2
from .personas import PERSONAS

OUT = ROOT / "docs" / "experiment_report.pdf"
CEDAR, INK, MUTED, RULE, SOFT = (colors.HexColor(c) for c in ("#1E4D3F", "#14332B", "#52514e", "#C3CEC7", "#EEF2EC"))
PAGE_W, PAGE_H = A4
MARGIN = 1.7 * cm
WIDTH = PAGE_W - 2 * MARGIN

# --------------------------------------------------------------------------- text
HEADLINES = [
    "<b>The instructions block matters most.</b> Removing it dropped success from 94% to 62%; two rules written "
    "from observed failures then raised success on those tasks from 58% to 92%.",
    "<b>Personas behave differently where it counts.</b> Auto mode sent every message to the persona built for it; "
    "the guardian logged praise the generalist host only thanked the visitor for.",
    "<b>Reasoning effort 'low' is a bad trade here.</b> Same success at 2.8x the cost, and it silently disables "
    "temperature and top-p on these models.",
    "<b>Temperature barely changes correctness.</b> 100% success from 0 to 1.6; it changes wording and which numbers "
    "are mentioned. Repetition penalties clearly cut repeated phrases (8.6% to 2.0%).",
    "<b>Smaller can be enough.</b> gpt-5.4-nano matched gpt-5.4-mini at about a third of the cost; gpt-4o-mini failed "
    "half the tasks by skipping tools.",
]

SECTIONS = [  # (section title, intro, [(suite, factor, heading, finding)])
    ("Personas", "Four personas share one ReAct loop; each has its own task, prompt traits and configuration.", [
        ("personas", "routing", "Auto mode (four personas) vs one persona for everything",
         "Overall success is a wash: 90% vs 95% here and the opposite (90% vs 85%) in an earlier run of the same "
         "suite, i.e. one task either way. The difference that held in both runs is task-specific: given praise for "
         "a tour, the host alone thanked the visitor but never logged it, while Auto mode sent it to Nadia, whose task "
         "is logging. Auto mode chose the intended persona for 100% of messages."),
        ("personas", "fidelity", "Each persona on the same four questions",
         "The personas are clearly distinct: Rami writes the longest answers (lists and calculations), Nadia the "
         "shortest. Alaa scores 5/5 for persona fidelity on every question; Rami scores lowest (4.25), including 4/5 "
         "on his own budget question, as the judge reads his numbers-first style as plain."),
    ]),
    ("Prompt engineering", "The same 8 core tasks, each run twice, with one change from the baseline at a time "
     "(baseline: chain of thought, one worked example, instructions in the system prompt).", [
        ("prompt", "chain of thought", "With vs without chain of thought",
         "Without the Thought lines, success is unchanged (94%) and answers are about 1 s faster, but format slips "
         "rise from 2 to 10: the Thought line keeps the model inside the ReAct format."),
        ("prompt", "examples", "Zero-shot vs one-shot vs few-shot",
         "Few-shot is the only variant at 100%. One of its examples shows the guardian quoting the feedback reference "
         "after logging, and the model copied that behaviour, which the rules did not ask for. Zero-shot reached 88%. "
         "The cost is the longest prompt ($0.0035 per run vs $0.0027)."),
        ("prompt", "placement", "Instructions in the system prompt vs in the message",
         "Moving the instructions into the visitor's message gave the fewest model calls (2.1), the fewest grounding "
         "checks (2 vs 9) and the fastest answers (3.6 s) at the same success rate: instructions next to the question "
         "are followed more closely."),
        ("prompt", "instructions", "With vs without the instructions block",
         "The largest effect in the study: success drops from 94% to 62%. Without the rules, Rami skipped the season "
         "check, the March 5 answer never stated the year, and the guardian once did not log an undocumented "
         "question. Format slips rose from 2 to 12, because the block also carries the output format."),
    ]),
    ("LLM configuration", "Each setting varied on the tasks where it matters, with the personas' own settings as "
     "the reference point.", [
        ("config", "temperature", "Temperature (0 to 1.6, three repeats each)",
         "Success stayed at 100% at every temperature and the judge's persona score rose slightly (4.33 to 4.67). "
         "At 1.6 only 33% of tasks gave the same numbers in all three repeats, against 67% at lower temperatures: "
         "temperature changes which figures are mentioned, not the totals ($345 was right in every run)."),
        ("config", "top_p", "Top-p (on the storyteller)",
         "All variants succeed; lower top-p gives shorter answers (184 vs 218 words) with less repetition. The effect "
         "is smaller than temperature's. Top-k is not offered by any OpenAI model, so top-p is the nucleus control."),
        ("config", "penalties", "Presence and frequency penalties (on the storyteller)",
         "The clearest sampling effect: repeated three-word phrases fall from 8.6% with no penalties to 4.8% with "
         "Layla's settings and 2.0% with high penalties, and answers get shorter. High penalties cost a little "
         "helpfulness and grounding, so Layla's moderate values are the balance."),
        ("config", "max tokens", "Maximum output tokens",
         "A 150-token limit cut off all three answers mid-sentence and helpfulness fell from 5.0 to 2.3, even though "
         "the checks passed because the key numbers came early. 400 tokens was already enough."),
        ("config", "model", "Model type (same prompt, same tasks)",
         "gpt-5.4-nano matches gpt-5.4-mini at about a third of the cost; gpt-6-luna is the cheapest and tersest; "
         "gpt-4o-mini is the weakest (50%), answering from general knowledge instead of using tools; gpt-6.1-sol costs "
         "2.6x mini and is almost 3x slower for no gain."),
        ("config", "reasoning effort", "Reasoning effort: none vs low",
         "Effort 'low' gives the same success but uses 1.7x the output tokens, 2.8x the cost and 2 s more per answer, "
         "and the temperature and top-p settings are dropped (the API rejects them above effort 'none')."),
        ("config", "stop sequence", "Stop sequence vs parser only",
         "On gpt-4.1-mini, one of the few models that accepts a stop sequence: no invented observations either way. "
         "Across all runs no model ever wrote its own Observation, so the parser's cut-off is a safety net."),
    ]),
    ("Instructions iteration", "Two failures repeated across every suite: the visitor was rarely given the feedback "
     "reference (task T07), and with experience, name and email known the agent asked for a date instead of saving "
     "the inquiry (task T19). Version 2 adds one single-purpose rule for each.", [
        ("iteration", "rules version", "Instructions v1 vs v2 (three repeats per task)",
         "Success went from 58% to 92%: T07 from 0/3 to 2/3, T19 from 1/3 to 3/3, and the two control tasks stayed "
         "at 3/3. The remaining failure gave the reference but did not say the information is undocumented."),
    ]),
]

BEST = [
    ("Personas", "Auto mode with four personas", "right persona 100% of the time; the guardian logs what the host skipped"),
    ("Reasoning style", "Chain of thought (Thought lines)", "same success, 5x fewer format slips"),
    ("Examples", "Few-shot", "the only variant at 100%; examples teach behaviours the rules do not state"),
    ("Instructions", "Version 2, block kept", "62% without the block; 58% to 92% from v1 to v2 on targeted tasks"),
    ("Placement", "System prompt (in-message close behind)", "in-message was faster with fewer calls, equal success"),
    ("Model", "gpt-5.4-mini; gpt-5.4-nano for the guardian", "nano matches mini at a third of the cost"),
    ("Reasoning effort", "none, with per-persona sampling", "'low' costs 2.8x and drops temperature and top-p"),
    ("Penalties", "Moderate, storyteller only", "repetition 8.6% to 4.8% without losing quality"),
    ("Max tokens", "400 or more", "150 cut every answer"),
]

REFLECTION = [
    ("Which persona gave the most helpful or natural results?",
     "Alaa, the warm host, was the most natural: 5/5 persona fidelity on every question she was given, short friendly "
     "answers, and on her own she passed 95% of the full test set. 'Most helpful' depended on the task, which is the "
     "case for personas with different jobs. Rami was the most trustworthy for numbers: he always showed the "
     "calculation, and the budget total was right in all 43 runs of that task. Nadia reliably logged praise and "
     "undocumented questions that the generalist host only acknowledged. Layla was the least natural at first: her "
     "early answers were one vivid sentence followed by a fact sheet, and 91% of her first drafts answered from the "
     "background text before searching the catalog (4% for Rami), which is why the grounding check exists. Her "
     "repetition penalties do their job, roughly halving repeated phrases."),
    ("Which prompt and configuration combination performed best?",
     "Chain of thought, few-shot examples and the version 2 instructions block in the system prompt, with every "
     "persona on reasoning effort 'none' so its sampling settings actually apply, gpt-5.4-mini for the three main "
     "personas and gpt-5.4-nano for the guardian. Two surprises: temperature barely affected correctness (100% at every "
     "value from 0 to 1.6), and reasoning effort 'low' was strictly worse here, costing 2.8x for the same success while "
     "silently disabling temperature and top-p."),
    ("How well did the agent reason and use tools?",
     "Well, with two clear patterns. Every turn ended with an answer, the right tool was used for every total "
     "(budget_calculator in all 43 runs of the budget task, always $345), and no model ever invented an observation. "
     "Format errors were rare (8 in 297 runs) and always recovered. The weak spot was grounding: the business summary "
     "in the prompt already lists prices and seasons, so in 45% of runs the first draft answered from it before calling "
     "a tool, and the grounding check sent it back (92% of those runs then succeeded). The other weakness was "
     "over-caution, such as asking for a date it did not need instead of saving an inquiry."),
    ("Did the instructions block change the agent's behaviour in a noticeable way? How?",
     "Yes, more than any other change. Without it, success fell from 94% to 62%, and every failure maps to a missing "
     "rule: Rami gave the right total but skipped the season check, the March 5 answer never stated the year, and the "
     "guardian once did not log an undocumented accessibility question. Format slips rose from 2 to 12 because the block "
     "carries the output format. The block was improved the same way: two rules written from observed failures raised "
     "success on those tasks from 58% to 92%."),
    ("What were the biggest challenges in implementation?",
     "First, the API is not uniform: top-k does not exist, newer models reject stop and max_tokens, and sampling "
     "settings are rejected whenever reasoning effort is above 'none'. Every model was probed, unsupported settings are "
     "dropped and reported, and a test fails if any persona's setting would be dropped. Second, parsing free text "
     "robustly without a stop sequence (bold labels, fences, inline calls, plain answers, invented observations). Third, "
     "a fix that backfired: the first grounding check was read by the model as the visitor speaking, until it was "
     "reworded as an internal check that must be followed by an action. Fourth, evaluating the evaluator: two early "
     "'agent failures' were a too-strict check and a judge that saw only part of a tool result. Fifth, small samples: "
     "with 8 to 20 runs per variant one task moves a rate by 5 to 12 points, so small differences were treated as noise."),
]

LIMITATIONS = [
    "Sample sizes are small (4 to 20 runs per variant): the large effects are clear, the small ones are not.",
    "The judge (gpt-5.4-mini) is generous and related to the agent; scores cluster at 4.5 to 5. A judge from another "
    "model family and blind pairwise comparisons would separate variants better.",
    "Grounding costs an extra model call in 45% of runs; keeping prices out of the business summary would force a tool "
    "call and could remove that step.",
    "Auto mode picks the persona with rules; an LLM classifier would cover phrasings the patterns miss.",
]


# --------------------------------------------------------------------------- building blocks
def _styles() -> dict:
    base = getSampleStyleSheet()
    s = {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=22, leading=26,
                                textColor=CEDAR, alignment=TA_LEFT, spaceAfter=4),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], fontSize=12, leading=16, textColor=MUTED,
                                   spaceAfter=12),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=15, leading=19,
                             textColor=CEDAR, spaceBefore=6, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=15,
                             textColor=INK, spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["Normal"], fontSize=9.5, leading=13.5, textColor=INK, spaceAfter=6),
        "small": ParagraphStyle("small", parent=base["Normal"], fontSize=8, leading=10.5, textColor=MUTED),
        "cell": ParagraphStyle("cell", parent=base["Normal"], fontSize=7.6, leading=9.4, textColor=INK),
        "head": ParagraphStyle("head", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=7, leading=8.6,
                               textColor=MUTED),
        "q": ParagraphStyle("q", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=10, leading=14,
                            textColor=CEDAR, spaceBefore=8, spaceAfter=3),
    }
    s["bullet"] = ParagraphStyle("bullet", parent=s["body"], leftIndent=12, bulletIndent=2)
    return s


ST = _styles()


def _clean(text) -> str:
    """Escape for reportlab and replace glyphs the standard fonts do not have."""
    text = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for a, b in (("→", "to"), ("≥", ">="), ("✓", "yes"), ("–", "-")):
        text = text.replace(a, b)
    return text


ARABIC_GLOSS = {"شو في نشاطات بالبقاع؟": "(in Lebanese Arabic) What activities are there in the Bekaa?"}


def _latin(text: str) -> str:
    """The standard PDF fonts have no Arabic glyphs: show an English gloss instead."""
    return ARABIC_GLOSS.get(text, text) if any("\u0600" <= ch <= "\u06ff" for ch in text) else text


def table(df: pd.DataFrame, widths=None, first_col_ratio: float = 0.22) -> Table:
    """A compact, wrapped table: bold header, light rules, first column wider."""
    head = [Paragraph(_clean(c), ST["head"]) for c in df.columns]
    rows = [[Paragraph(_clean(v), ST["cell"]) for v in r] for r in df.astype(str).values.tolist()]
    n = len(df.columns)
    if widths is None:
        first = WIDTH * first_col_ratio
        widths = [first] + [(WIDTH - first) / max(1, n - 1)] * (n - 1)
    t = Table([head] + rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, 0), 1.0, RULE),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, colors.HexColor("#E3EAE5")),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def chart(suite: str, factor: str, folder: Path):
    """The comparison chart, re-drawn as a grid that fits the page width."""
    summary = report.summary(suite)
    data = ex.factor_tables(summary)[factor]
    path = ex.plot_factor(data, suite, factor, ncols=4, path=folder / f"{suite}_{factor.replace(' ', '_')}.png",
                          title=False)
    if not path:
        return None
    from PIL import Image as PILImage

    w, h = PILImage.open(path).size
    return Image(str(path), width=WIDTH, height=WIDTH * h / w)


def _numbers() -> dict:
    runs = pd.concat([ex.load_runs(s) for s in ("personas", "prompt", "config", "iteration")], ignore_index=True)
    judge = runs["judge_cost_usd"].sum() if "judge_cost_usd" in runs else 0.0
    return {"runs": len(runs), "agent_cost": runs["cost_usd"].sum(), "judge_cost": judge,
            "sent_back": (runs["grounding_checks"] > 0).mean(), "all_runs": runs}


def _on_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(CEDAR)
    canvas.rect(0, PAGE_H - 0.45 * cm, PAGE_W, 0.45 * cm, stroke=0, fill=1)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 1.0 * cm, "Douroub Lebanon ReAct agent - experiment report")
    canvas.drawRightString(PAGE_W - MARGIN, 1.0 * cm, f"Page {doc.page}")
    canvas.restoreState()


# --------------------------------------------------------------------------- the report
def build(out: Path = OUT) -> Path:
    n = _numbers()
    story = [
        Paragraph("Douroub Lebanon ReAct agent", ST["title"]),
        Paragraph("Experiment report: personas, prompt engineering and LLM configuration", ST["subtitle"]),
    ]
    stats = [[Paragraph(f"<font size=16 color='#1E4D3F'><b>{v}</b></font><br/>{_clean(k)}", ST["small"])
              for k, v in (("logged runs", n["runs"]), ("test tasks", len(TASKS)), ("personas", len(PERSONAS)),
                           ("total cost (agent + judge)", f"${n['agent_cost'] + n['judge_cost']:.2f}"))]]
    st = Table(stats, colWidths=[WIDTH / 4] * 4)
    st.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), SOFT), ("TOPPADDING", (0, 0), (-1, -1), 8),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 8), ("LEFTPADDING", (0, 0), (-1, -1), 10)]))
    story += [st, Spacer(1, 12), Paragraph("Headline findings", ST["h1"])]
    story += [Paragraph(h, ST["bullet"], bulletText="•") for h in HEADLINES]
    story += [Spacer(1, 4), Paragraph(
        f"Report generated on {date.today():%d %B %Y} from the saved results in <i>experiments/results/</i>. "
        "Inquiries and feedback created during experiments went to a sandbox log.", ST["small"])]

    # ---- method
    story += [PageBreak(), Paragraph("1. What was tested", ST["h1"]), Paragraph(
        "A ReAct agent written by hand in LangGraph (prepare, reason, act, force_answer, finalize) helps visitors of "
        "Douroub Lebanon choose and plan experiences across Lebanon with seven tools. Four personas share the loop; "
        "each is defined by prompt traits and a matching LLM configuration:", ST["body"])]
    personas = report.persona_table().data[["persona", "task", "model", "temperature", "top_p",
                                            "presence / frequency penalty", "max tokens", "reasoning effort"]]
    story += [table(personas, widths=[3.0 * cm, 5.4 * cm, 1.9 * cm, 1.3 * cm, 1.0 * cm, 1.7 * cm, 1.2 * cm,
                                      WIDTH - 15.5 * cm]), Spacer(1, 8)]
    story += [Paragraph(
        "<b>Test set.</b> 20 tasks covering every job of the agent: prices, seasons, itineraries, comparisons, "
        "descriptions, accessibility and dietary questions, bookings with and without an email, a two-message booking "
        "that needs memory, praise, off-topic and prompt-injection requests, French and Arabic, and a date without a "
        "year. Suites that multiply many variants use an 8-task core set.", ST["body"])]
    tasks = pd.DataFrame([{"id": t.id, "category": t.category, "message": _latin(t.message),
                           "core": "yes" if t.id in CORE_TASKS else ""} for t in TASKS])
    story += [table(tasks, widths=[1.0 * cm, 2.2 * cm, WIDTH - 4.4 * cm, 1.2 * cm]), Spacer(1, 8)]
    story += [Paragraph(
        "<b>Scoring.</b> Automatic checks on every run: the tools that had to run successfully, facts the answer must "
        "contain (for example the total 345), and four honesty checks (no unknown experience IDs, no reference ID that "
        "a tool did not return, never 'booking confirmed', and the loop ended with an answer). <i>Success</i> means every "
        "check passed. An LLM judge (gpt-5.4-mini, temperature 0) scored persona fidelity, helpfulness and grounding "
        "from 1 to 5, given the persona and the full tool results. Every run also logs model calls, format errors and "
        "slips, grounding checks, tokens, cost and latency.", ST["body"])]

    # ---- results
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for number, (title, intro, items) in enumerate(SECTIONS, start=2):
            story += [PageBreak(), Paragraph(f"{number}. {title}", ST["h1"]), Paragraph(_clean(intro), ST["body"])]
            for suite, factor, heading, finding in items:
                block = [Paragraph(_clean(heading), ST["h2"]),
                         Paragraph("<b>Finding.</b> " + _clean(finding), ST["body"]),
                         table(report.table(suite, factor).data), Spacer(1, 4)]
                img = chart(suite, factor, folder)
                if img:
                    block.append(img)
                story.append(KeepTogether(block))
                if (suite, factor) == ("personas", "fidelity"):
                    runs = n["all_runs"]
                    g = (runs.assign(sent=runs["grounding_checks"] > 0).groupby("persona")["sent"]
                         .agg(["sum", "count", "mean"]))
                    g = pd.DataFrame({"persona": [PERSONAS[k].name for k in g.index], "runs sent back": g["sum"],
                                      "runs": g["count"], "share": [f"{v:.0%}" for v in g["mean"]]})
                    story.append(KeepTogether([
                        Paragraph("Grounding check by persona", ST["h2"]),
                        Paragraph("How often each persona's first draft quoted catalog facts before using a tool and "
                                  "was sent back, across all runs. Describing a place invites the storyteller to talk "
                                  "before looking anything up.", ST["body"]),
                        table(g, widths=[4 * cm, 4 * cm, 4 * cm, WIDTH - 12 * cm])]))
            if title == "LLM configuration":
                cap = report.capability_table().reset_index().rename(columns={"index": "setting"})
                story += [PageBreak(), Paragraph("Which settings each model accepts", ST["h2"]), Paragraph(
                    "Measured by sending every setting to every model (scripts/probe_models.py). Top-k does not exist "
                    "in the OpenAI API, and on the newer models sampling settings only work with reasoning effort "
                    "'none'.", ST["body"]), table(cap, first_col_ratio=0.30)]
            if title == "Instructions iteration":
                runs = ex.load_runs("iteration")
                pivot = runs.pivot_table(index="task", columns="arm", values="success", aggfunc="sum").astype(int)
                pivot = pivot.astype(str).add(" / 3").reset_index()
                added = [r for r in RULES_V2 if r not in RULES_V1]
                story += [KeepTogether([Paragraph("Per task", ST["h2"]), table(pivot, first_col_ratio=0.2),
                                        Spacer(1, 6), Paragraph("The two rules added in version 2:", ST["body"])]
                                       + [Paragraph(_clean(r), ST["bullet"], bulletText="+") for r in added]
                                + [Spacer(1, 4), Paragraph(
                                    "<b>Version 3 (after testing the app by hand).</b> Version 2's inquiry rule let "
                                    "the agent save an inquiry without asking how many people were coming, which the "
                                    "team needs to quote a price. Version 3 rewrites that rule, and the number of "
                                    "people is now a required argument of the inquiry tool, with a Python check that "
                                    "the visitor actually gave it.", ST["body"])])]

        # ---- conclusions
        story += [PageBreak(), Paragraph("6. The best combination for this use case", ST["h1"]),
                  table(pd.DataFrame(BEST, columns=["choice", "best setting", "evidence"]),
                        widths=[3.2 * cm, 5.6 * cm, WIDTH - 8.8 * cm])]
        story += [Spacer(1, 6), Paragraph("7. Reflection", ST["h1"])]
        for q, a in REFLECTION:
            story += [Paragraph(_clean(q), ST["q"]), Paragraph(_clean(a), ST["body"])]
        story += [Paragraph("8. Limitations", ST["h1"])]
        story += [Paragraph(_clean(t), ST["bullet"], bulletText="•") for t in LIMITATIONS]

        out.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=1.5 * cm,
                                bottomMargin=1.6 * cm, title="Douroub Lebanon ReAct agent: experiment report",
                                author="Douroub Lebanon")
        doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    return out

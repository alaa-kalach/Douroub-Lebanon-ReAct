"""Draw the ReAct state graph as docs/react_graph.png (the same nodes and edges as douroub.agent.build_graph)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, MUTED, SURFACE, LINE = "#0b0b0b", "#52514e", "#fcfcfb", "#8a8984"
FILL, ACCENT, TERMINAL = "#e8f0fb", "#2a78d6", "#f0efec"
OUT = Path(__file__).resolve().parent.parent / "docs" / "react_graph.png"

NODES = {  # name: (x, y, subtitle)
    "START": (0.0, 6.2, ""),
    "prepare": (0.0, 5.0, "route + pick persona + build prompt"),
    "reason": (0.0, 3.4, "1 model call → parse_step()"),
    "act": (4.7, 3.4, "guards → run tool → Observation"),
    "force_answer": (4.7, 1.6, "step limit: answer from observations"),
    "finalize": (0.0, 1.6, "save to memory (checkpointer)"),
    "END": (0.0, 0.4, ""),
}
W, H = 2.9, 0.78


def box(ax, name):
    x, y, sub = NODES[name]
    terminal = name in ("START", "END")
    w, h = (1.2, 0.5) if terminal else (W, H)
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                facecolor=TERMINAL if terminal else FILL,
                                edgecolor=ACCENT if name == "reason" else LINE, linewidth=1.8 if name == "reason" else 1))
    ax.text(x, y + (0.12 if sub else 0), name, ha="center", va="center", fontsize=11, color=INK, weight="bold",
            family="monospace")
    if sub:
        ax.text(x, y - 0.2, sub, ha="center", va="center", fontsize=8, color=MUTED)


def arrow(ax, start, end, label="", rad=0.0, dashed=False, label_xy=None):
    """An edge between two points on the box borders (never through a box)."""
    ax.add_patch(FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=14,
                                 color=LINE, linewidth=1.2, linestyle="--" if dashed else "-", shrinkA=0, shrinkB=1))
    if label:
        ax.text(*(label_xy or ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)), label, ha="center", va="center",
                fontsize=8.5, color=INK, bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE, edgecolor="none"))


def main():
    fig, ax = plt.subplots(figsize=(9, 7.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for n in NODES:
        box(ax, n)
    hw, hh = W / 2, H / 2
    arrow(ax, (0, 6.2 - 0.25), (0, 5.0 + hh))                                   # START -> prepare
    arrow(ax, (0, 5.0 - hh), (0, 3.4 + hh))                                     # prepare -> reason
    arrow(ax, (hw, 3.55), (4.7 - hw, 3.55), "Action + Action Input", dashed=True, label_xy=(2.35, 3.82))
    arrow(ax, (4.7 - hw, 3.25), (hw, 3.25), "Observation", dashed=True, label_xy=(2.35, 2.98))
    arrow(ax, (0, 3.4 - hh), (0, 1.6 + hh), "Final Answer", dashed=True, label_xy=(0, 2.5))
    arrow(ax, (1.1, 3.4 - hh), (4.7 - hw, 1.75), "step limit (6)", dashed=True, label_xy=(2.3, 2.45))
    arrow(ax, (4.7, 3.4 - hh), (4.7, 1.6 + hh), "step limit", dashed=True, label_xy=(4.7, 2.5))
    arrow(ax, (4.7 - hw, 1.6), (hw, 1.6))                                       # force_answer -> finalize
    arrow(ax, (0, 1.6 - hh), (0, 0.4 + 0.25))                                   # finalize -> END
    # reason -> reason: a format error or a failed grounding check is sent back as an observation
    arrow(ax, (-hw, 3.6), (-hw, 3.2), dashed=True, rad=1.4)
    ax.text(-2.45, 3.4, "format error or\ngrounding check\n(sent back)", ha="center", va="center", fontsize=8.5,
            color=INK)
    ax.text(-3.2, 7.35, "Douroub ReAct agent: LangGraph state machine", fontsize=12, color=INK, weight="bold")
    ax.text(-3.2, 7.0, "solid = always; dashed = conditional edge chosen by after_reason / after_act", fontsize=8.5,
            color=MUTED)
    ax.set_xlim(-3.3, 6.4)
    ax.set_ylim(0, 7.6)
    ax.axis("off")
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()

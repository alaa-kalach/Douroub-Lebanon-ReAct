"""Which OpenAI models accept which generation settings?

Step 0 of the ReAct project: before designing personas around temperature, top-p, penalties and so on, check
what each candidate model actually accepts through the Chat Completions API. Every probe is one tiny request
(a few output tokens), so the whole run costs a fraction of a cent.

    python scripts/probe_models.py                 # default candidate models
    python scripts/probe_models.py gpt-6-luna      # or name the models to probe
    python scripts/probe_models.py --combos        # sampling settings combined with reasoning effort

Results are printed as a table and saved to experiments/model_capabilities.json.
"""
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)

CANDIDATES = ["gpt-6-luna", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-6.1-sol", "gpt-4.1-mini", "gpt-4o-mini"]

# One setting at a time, each on top of the same minimal request. top_k is not part of the OpenAI API, so it is
# probed only to document that the API rejects it.
PROBES = {
    "baseline": {},
    "temperature=0.2": {"temperature": 0.2},
    "temperature=1.2": {"temperature": 1.2},
    "top_p=0.5": {"top_p": 0.5},
    "frequency_penalty=0.6": {"frequency_penalty": 0.6},
    "presence_penalty=0.6": {"presence_penalty": 0.6},
    "max_completion_tokens=20": {"max_completion_tokens": 20},
    "max_tokens=20": {"max_tokens": 20},
    "stop=['Observation:']": {"stop": ["Observation:"]},
    "seed=7": {"seed": 7},
    "reasoning_effort=none": {"reasoning_effort": "none"},
    "reasoning_effort=low": {"reasoning_effort": "low"},
    "top_k=40 (not an OpenAI setting)": {"extra_body": {"top_k": 40}},
}

PROMPT = [{"role": "user", "content": "Reply with exactly one word: pong"}]

# --combos: sampling settings together with a reasoning-effort value (some models accept sampling settings only
# when they are not reasoning, so single-setting probes are not enough).
COMBO_MODELS = ["gpt-6-luna", "gpt-5.4-mini", "gpt-5.4-nano"]
COMBO_SAMPLING = {"temperature=0.1": {"temperature": 0.1}, "temperature=1.1": {"temperature": 1.1},
                  "top_p=0.5": {"top_p": 0.5}, "presence_penalty=0.6": {"presence_penalty": 0.6},
                  "frequency_penalty=0.3": {"frequency_penalty": 0.3}}
COMBO_EFFORTS = ["none", "low"]


def probe(client, model: str, extra: dict) -> tuple:
    """Return (ok, detail): the reply text, or the API's error message shortened."""
    kwargs = {"model": model, "messages": PROMPT}
    if "max_completion_tokens" not in extra and "max_tokens" not in extra:
        kwargs["max_completion_tokens"] = 400  # room for models that think before answering
    kwargs.update(extra)
    try:
        resp = client.chat.completions.create(**kwargs)
        return True, (resp.choices[0].message.content or "").strip()[:20] or "(empty)"
    except Exception as exc:  # report every rejection, whatever its type
        msg = getattr(exc, "message", None) or str(exc)
        return False, f"{getattr(exc, 'status_code', '')} {msg}".strip()[:140]


def combos(client) -> None:
    """Probe each sampling setting together with each reasoning effort, for the models that support both."""
    results = {}
    for model in COMBO_MODELS:
        results[model] = {}
        for effort in COMBO_EFFORTS:
            for name, extra in COMBO_SAMPLING.items():
                ok, detail = probe(client, model, extra | {"reasoning_effort": effort})
                results[model][f"{name} + effort={effort}"] = {"ok": ok, "detail": detail}
                time.sleep(0.2)
        print(f"{model:14} probed")
    rows = list(next(iter(results.values())))
    print("\n" + "setting".ljust(36) + "".join(m.ljust(15) for m in COMBO_MODELS))
    for row in rows:
        print(row.ljust(36) + "".join(("yes" if results[m][row]["ok"] else "NO").ljust(15) for m in COMBO_MODELS))
    out = ROOT / "experiments" / "model_capabilities_combos.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {out.relative_to(ROOT)}")


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set: add it to .env first.")
    client = OpenAI(max_retries=0, timeout=60)
    if "--combos" in sys.argv:
        combos(client)
        return
    models = sys.argv[1:] or CANDIDATES
    results = {}
    for model in models:
        ok, detail = probe(client, model, {})
        results[model] = {"baseline": {"ok": ok, "detail": detail}}
        if not ok:  # unknown model or no access: skip its other probes
            print(f"{model:14} unavailable: {detail}")
            continue
        for name, extra in list(PROBES.items())[1:]:
            ok, detail = probe(client, model, extra)
            results[model][name] = {"ok": ok, "detail": detail}
            time.sleep(0.2)
        print(f"{model:14} probed")

    names = list(PROBES)
    available = [m for m in models if results[m]["baseline"]["ok"]]
    print("\n" + "setting".ljust(34) + "".join(m[:14].ljust(15) for m in available))
    for name in names:
        print(name.ljust(34) + "".join(("yes" if results[m][name]["ok"] else "NO").ljust(15) for m in available))

    print("\nWhy each setting was rejected:")
    for m in available:
        for name in names:
            if not results[m][name]["ok"]:
                print(f"  {m} / {name}: {results[m][name]['detail']}")

    out = ROOT / "experiments" / "model_capabilities.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

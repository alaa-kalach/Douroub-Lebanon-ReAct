"""Configuration, the OpenAI client, model calls with per-persona settings, and the business knowledge.

Every model call in the project goes through `call_model()`, which
  * applies a generation config (model, temperature, top-p, penalties, max tokens, reasoning effort, seed),
  * drops settings the chosen model does not accept (measured by scripts/probe_models.py) and reports them,
  * retries on rate limits, and
  * returns the text together with token usage and an estimated cost, so experiments can compare setups.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
ME_DIR = ROOT / "me"
LOG_DIR = ROOT / "logs"
EXPERIMENTS_DIR = ROOT / "experiments"

load_dotenv(ROOT / ".env", override=True)
API_KEY = os.getenv("OPENAI_API_KEY", "")
REQUEST_TIMEOUT_S = 60


def has_api_key() -> bool:
    return bool(API_KEY)


@lru_cache(maxsize=1)
def get_client():
    """OpenAI client. The SDK's own retries are off; call_model() retries rate limits itself."""
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    from openai import OpenAI

    return OpenAI(api_key=API_KEY, max_retries=0, timeout=REQUEST_TIMEOUT_S)


# --------------------------------------------------------------------------- generation configs
@dataclass(frozen=True)
class LLMConfig:
    """One LLM configuration. None means "use the model's default" (the setting is not sent)."""
    model: str = "gpt-5.4-mini"
    temperature: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    max_completion_tokens: int | None = 800
    reasoning_effort: str | None = None
    seed: int | None = None
    stop: tuple | None = None   # only some models accept a stop sequence (see experiments/model_capabilities.json)

    def with_(self, **changes) -> "LLMConfig":
        data = asdict(self) | changes
        return LLMConfig(**data)

    def short(self) -> str:
        """Compact label for tables, e.g. 'gpt-5.4-mini t=0.1 p=0.5 effort=low'."""
        parts = [self.model]
        for key, label in (("temperature", "t"), ("top_p", "p"), ("frequency_penalty", "fp"),
                           ("presence_penalty", "pp"), ("max_completion_tokens", "max"),
                           ("reasoning_effort", "effort")):
            value = getattr(self, key)
            if value is not None:
                parts.append(f"{label}={value}")
        if self.stop:
            parts.append("stop")
        return " ".join(parts)


# Settings each model rejects, measured with scripts/probe_models.py (17 October 2026). If the probe's JSON
# file exists it replaces this table, so a re-run of the probe updates the project automatically.
_MEASURED_REJECTIONS = {
    "gpt-6-luna": {"temperature", "top_p", "frequency_penalty", "presence_penalty", "stop"},
    "gpt-6.1-sol": {"temperature", "top_p", "frequency_penalty", "presence_penalty", "stop"},
    "gpt-5.4-mini": {"stop"},
    "gpt-5.4-nano": {"stop"},
    "gpt-4.1-mini": {"reasoning_effort"},
    "gpt-4o-mini": {"reasoning_effort"},
}
_PROBE_KEYS = {"temperature=0.2": "temperature", "top_p=0.5": "top_p", "frequency_penalty=0.6": "frequency_penalty",
               "presence_penalty=0.6": "presence_penalty", "stop=['Observation:']": "stop",
               "reasoning_effort=low": "reasoning_effort", "seed=7": "seed"}


@lru_cache(maxsize=1)
def model_rejections() -> dict:
    """{model: set of settings it rejects}, from experiments/model_capabilities.json when available."""
    path = EXPERIMENTS_DIR / "model_capabilities.json"
    if not path.exists():
        return _MEASURED_REJECTIONS
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for model, probes in data.items():
        if not probes.get("baseline", {}).get("ok"):
            continue
        out[model] = {setting for key, setting in _PROBE_KEYS.items()
                      if key in probes and not probes[key]["ok"]}
    return out | {m: s for m, s in _MEASURED_REJECTIONS.items() if m not in out}


_SAMPLING = ("temperature", "top_p", "presence_penalty", "frequency_penalty")
# Combined probes (scripts/probe_models.py --combos, 17 October 2026): sampling settings work only with
# reasoning_effort="none"; at "low" every model rejects all of them, and gpt-6-luna never takes penalties.
_MEASURED_COMBOS = {
    ("gpt-6-luna", "none"): {"temperature": True, "top_p": True, "presence_penalty": False, "frequency_penalty": False},
    **{(m, "none"): {s: True for s in _SAMPLING} for m in ("gpt-5.4-mini", "gpt-5.4-nano")},
    **{(m, "low"): {s: False for s in _SAMPLING} for m in ("gpt-6-luna", "gpt-5.4-mini", "gpt-5.4-nano")},
}


@lru_cache(maxsize=1)
def combo_rejections() -> dict:
    """{(model, reasoning_effort): {setting: accepted?}} from experiments/model_capabilities_combos.json.

    Some models accept sampling settings only at certain reasoning efforts, so these combined probes take
    precedence over the single-setting table for the settings they cover."""
    path = EXPERIMENTS_DIR / "model_capabilities_combos.json"
    if not path.exists():
        return _MEASURED_COMBOS
    out = {}
    for model, probes in json.loads(path.read_text(encoding="utf-8")).items():
        for label, result in probes.items():
            m = re.match(r"(\w+)=[^ ]+ \+ effort=(\w+)", label)
            if m:
                setting, effort = m.groups()
                accepted = out.setdefault((model, effort), {})
                accepted[setting] = accepted.get(setting, True) and result["ok"]
    return out


def request_kwargs(config: LLMConfig) -> tuple[dict, list]:
    """Turn a config into API arguments. Returns (kwargs, dropped) where dropped lists the settings the model
    does not accept, so callers can record them instead of failing."""
    rejected = set(model_rejections().get(config.model, set()))
    for setting, accepted in combo_rejections().get((config.model, config.reasoning_effort), {}).items():
        (rejected.discard if accepted else rejected.add)(setting)
    kwargs, dropped = {"model": config.model}, []
    for key in ("temperature", "top_p", "frequency_penalty", "presence_penalty", "max_completion_tokens",
                "reasoning_effort", "seed", "stop"):
        value = getattr(config, key)
        if value is None:
            continue
        if key in rejected:
            dropped.append(key)
            continue
        kwargs[key] = list(value) if key == "stop" else value
    return kwargs, dropped


# --------------------------------------------------------------------------- cost
# USD per million tokens (input, cached input, output) from OpenAI's model pages, October 2026. Used only to
# compare setups in the experiments; check https://developers.openai.com/api/docs/models for current prices.
PRICES = {
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-6.1-sol": (2.00, 0.10, 10.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
}


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0

    def add(self, other: "Usage") -> "Usage":
        for f in ("calls", "input_tokens", "cached_tokens", "output_tokens", "cost_usd", "latency_s"):
            setattr(self, f, getattr(self, f) + getattr(other, f))
        return self

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cost_usd"], d["latency_s"] = round(d["cost_usd"], 6), round(d["latency_s"], 2)
        return d


def estimate_cost(model: str, input_tokens: int, cached: int, output_tokens: int) -> float:
    price_in, price_cached, price_out = PRICES.get(model, (0.0, 0.0, 0.0))
    return ((input_tokens - cached) * price_in + cached * price_cached + output_tokens * price_out) / 1_000_000


@dataclass
class ModelReply:
    text: str
    usage: Usage = field(default_factory=Usage)
    dropped_settings: list = field(default_factory=list)
    model: str = ""
    finish_reason: str = ""        # "stop", or "length" when max_completion_tokens cut the reply off


class QuotaExhausted(RuntimeError):
    """No billing credit left, or a per-day limit: waiting will not help."""


def _suggested_wait(exc) -> float | None:
    m = re.search(r"try again in (\d+(?:\.\d+)?)\s*(ms|s)\b", str(exc), re.I)
    if not m:
        return None
    return float(m.group(1)) / (1000 if m.group(2).lower() == "ms" else 1)


def call_model(messages: list, config: LLMConfig, client=None, attempts: int = 4) -> ModelReply:
    """One chat completion with the given config. Retries rate limits; raises other errors."""
    client = client or get_client()
    kwargs, dropped = request_kwargs(config)
    start = time.perf_counter()
    for attempt in range(1, attempts + 1):
        try:
            resp = client.chat.completions.create(messages=messages, **kwargs)
            break
        except Exception as exc:
            if getattr(exc, "status_code", None) != 429:
                raise
            if "insufficient_quota" in str(exc) or re.search(r"(?i)per day|\bRPD\b|\bTPD\b", str(exc)):
                raise QuotaExhausted(str(exc)[:200]) from exc
            if attempt == attempts:
                raise
            wait = min(_suggested_wait(exc) or 5 * attempt, 60) + 0.5
            print(f"[RATE LIMIT] {config.model}: waiting {wait:.1f}s")
            time.sleep(wait)
    usage = Usage(calls=1, latency_s=time.perf_counter() - start)
    u = getattr(resp, "usage", None)
    if u is not None:
        details = getattr(u, "prompt_tokens_details", None)
        usage.input_tokens = int(getattr(u, "prompt_tokens", 0) or 0)
        usage.output_tokens = int(getattr(u, "completion_tokens", 0) or 0)
        usage.cached_tokens = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
        usage.cost_usd = estimate_cost(config.model, usage.input_tokens, usage.cached_tokens, usage.output_tokens)
    choice = resp.choices[0]
    text = (choice.message.content or "").strip()
    return ModelReply(text=text, usage=usage, dropped_settings=dropped, model=config.model,
                      finish_reason=str(getattr(choice, "finish_reason", "") or ""))


# --------------------------------------------------------------------------- business knowledge
SUMMARY_PATH = ME_DIR / "business_summary.txt"
CATALOG_PATH = ME_DIR / "experiences.csv"


@lru_cache(maxsize=1)
def load_summary() -> str:
    return SUMMARY_PATH.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=1)
def load_catalog() -> pd.DataFrame:
    return pd.read_csv(CATALOG_PATH).set_index("id", drop=False)

"""The agent's seven tools, with MCP-style schemas and validation, and the checks Python enforces whatever the
model decides.

    record_customer_interest, record_feedback   save an inquiry / an unanswered question or feedback for the team
    search_experiences, compare_experiences       find and compare catalog experiences
    create_itinerary                              build a day-by-day plan within budget, season and group size
    budget_calculator, season_check               exact totals and season checks, so the model never guesses them

Every tool is a plain Python function that returns a dict with a "status". The ReAct loop's `act` node calls
them through `guarded_execute()`: the safety checks first, then `execute_tool()`, which validates the arguments
against the tool's inputSchema before running it.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from collections import Counter
from datetime import date, datetime, timezone

from . import core
from .core import load_catalog
from .router import FEEDBACK_RE, UNKNOWN_RE

LOG_DIR = core.LOG_DIR  # tests and experiments point this somewhere else
VERBOSE = True          # print [TOOL] and [PUSH] lines (the experiment runner turns this off)

EMAIL_VALID_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}$")
PLACEHOLDERS = {"", "n/a", "na", "none", "null", "unknown", "test", "name", "your name", "user", "visitor",
                "customer", "guest", "anonymous", "-", "?"}
FEEDBACK_KINDS = ("unanswered_question", "complaint", "correction", "suggestion", "praise", "other")
_RECENT = {}  # (log file, dedupe key) -> record id; stops the same record being saved twice in one session


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(timezone.utc):%Y%m%d}-{uuid.uuid4().hex[:6].upper()}"


def _append_jsonl(filename: str, record: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def mask_email(email: str) -> str:
    local, _, domain = str(email).partition("@")
    return f"{local[:1]}***@{domain}" if domain else "***"


def push(text: str) -> None:
    """Notify the team: print a [PUSH] line (the record itself is already saved to a JSONL log file)."""
    if VERBOSE:
        print(f"[PUSH] {text}")


def _clean(text) -> str:
    return " ".join(str(text or "").split())


def record_customer_interest(email: str, name: str, message: str, group_size: int = None) -> dict:
    """Validate and save a booking inquiry (lead) to logs/inquiries.jsonl. The team needs the number of people
    to check availability and quote a price, so an inquiry without it is not saved."""
    email, name, message = _clean(email).lower(), _clean(name), _clean(message)[:2000]
    problems = {}
    try:
        group_size = int(group_size) if group_size is not None else None
    except (TypeError, ValueError):
        group_size = None
    if group_size is None or not 1 <= group_size <= 50:
        problems["group_size"] = "the number of people (1-50) is required: ask the visitor how many will join"
    if name.lower() in PLACEHOLDERS or len(name) < 2:
        problems["name"] = "the visitor's real name is required"
    elif len(name) > 100:
        problems["name"] = "name is too long (max 100 characters)"
    if not EMAIL_VALID_RE.match(email):
        problems["email"] = "a valid email address is required, for example name@example.com"
    if len(message) < 10 or len(message.split()) < 2:
        problems["message"] = "describe what the visitor wants (experience, dates, group size)"
    if problems:
        return {"status": "error", "error": "validation_failed", "fields": problems,
                "message": "Inquiry NOT saved. Ask the visitor for the missing or invalid details."}

    key = ("inquiries.jsonl", email, message.lower())
    if key in _RECENT:
        return {"status": "success", "inquiry_id": _RECENT[key], "duplicate": True,
                "message": "This inquiry was already recorded earlier in the conversation."}
    record = {"inquiry_id": _new_id("INQ"), "timestamp": _now_iso(), "name": name, "email": email,
              "group_size": group_size, "message": message, "status": "new"}
    try:
        _append_jsonl("inquiries.jsonl", record)
    except OSError as exc:
        return {"status": "error", "error": "storage_failed",
                "message": f"Inquiry could not be saved ({type(exc).__name__}). Do not tell the visitor it was saved."}
    _RECENT[key] = record["inquiry_id"]
    push(f"New inquiry {record['inquiry_id']} from {name} <{mask_email(email)}>, {group_size} people: {message[:150]}")
    return {"status": "success", "inquiry_id": record["inquiry_id"], "saved_at": record["timestamp"],
            "next_step": "Khaled from the team will reply by email to confirm availability, dates, and the "
                         "final price. This is an inquiry, not a confirmed reservation."}


def record_feedback(question: str, kind: str = "unanswered_question") -> dict:
    """Save an unanswered question or a piece of feedback to logs/feedback.jsonl."""
    question, kind = _clean(question)[:2000], _clean(kind).lower() or "unanswered_question"
    if len(question) < 5:
        return {"status": "error", "error": "validation_failed", "fields": {"question": "the question or feedback text is required"},
                "message": "Feedback NOT saved: provide the visitor's question or feedback in full."}
    if kind not in FEEDBACK_KINDS:
        return {"status": "error", "error": "validation_failed", "fields": {"kind": f"must be one of {list(FEEDBACK_KINDS)}"},
                "message": "Feedback NOT saved: invalid kind."}

    key = ("feedback.jsonl", kind, question.lower())
    if key in _RECENT:
        return {"status": "success", "feedback_id": _RECENT[key], "duplicate": True,
                "message": "This was already recorded earlier in the conversation."}
    record = {"feedback_id": _new_id("FB"), "timestamp": _now_iso(), "kind": kind, "question": question,
              "status": "open"}
    try:
        _append_jsonl("feedback.jsonl", record)
    except OSError as exc:
        return {"status": "error", "error": "storage_failed",
                "message": f"Feedback could not be saved ({type(exc).__name__}). Do not tell the visitor it was saved."}
    _RECENT[key] = record["feedback_id"]
    push(f"New {kind.replace('_', ' ')} {record['feedback_id']}: {question[:150]}")
    return {"status": "success", "feedback_id": record["feedback_id"], "saved_at": record["timestamp"],
            "next_step": "The team reviews every logged question and feedback and follows up when contact details are available."}


MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
          "october", "november", "december"]
REGION_ALIASES = {  # broad area names a visitor might use -> catalog regions
    "north": {"Tripoli", "Bcharre", "Qadisha Valley", "Batroun"},
    "north lebanon": {"Tripoli", "Bcharre", "Qadisha Valley", "Batroun"},
    "south": {"Sidon", "Tyre", "Jezzine"},
    "south lebanon": {"Sidon", "Tyre", "Jezzine"},
    "mount lebanon": {"Chouf", "Metn", "Keserwan", "Byblos"},
    "shouf": {"Chouf"},
    "jbeil": {"Byblos"},
    "saida": {"Sidon"},
    "sour": {"Tyre"},
    "zahle": {"Bekaa"},
    "baalbek": {"Bekaa"},
    "jounieh": {"Keserwan"},
    "qadisha": {"Qadisha Valley"},
    "bsharri": {"Bcharre"},
}
SYNONYMS = {"hike": "hiking", "hikes": "hiking", "trek": "hiking", "walks": "walking", "historic": "history",
            "historical": "history", "heritage": "history", "ancient": "history", "ruins": "archaeology",
            "archaeological": "archaeology", "beach": "coast", "seaside": "coast", "sea": "coast",
            "kids": "families", "children": "families", "family": "families", "outdoor": "outdoors",
            "culinary": "food", "eat": "food", "eating": "food", "cuisine": "food", "dishes": "food",
            "mountain": "nature", "mountains": "nature", "forest": "nature", "forests": "nature",
            "wines": "wine", "winery": "wine", "wineries": "wine", "craft": "crafts", "handmade": "crafts",
            "photos": "photography", "photo": "photography", "village": "village", "villages": "village",
            "relax": "relaxation", "relaxing": "relaxation", "cedar": "cedars", "kayak": "kayaking",
            "snowshoe": "snow", "snowshoeing": "snow", "winter": "snow", "market": "markets", "souk": "markets",
            "souks": "markets", "sweet": "food", "sweets": "food", "cooking": "cooking", "cook": "cooking"}
STOPWORDS = {"a", "an", "and", "the", "to", "of", "in", "on", "for", "with", "or", "i", "we", "my", "our",
             "like", "love", "want", "something", "anything", "trip", "tour", "tours", "experience",
             "experiences", "activity", "activities", "lebanon", "some", "any", "day", "do", "go", "see",
             "is", "are", "be", "near", "around", "good", "best", "nice", "interested", "into"}


def _tokens(text: str) -> set:
    words = re.findall(r"[a-z]+", str(text).lower())
    return {SYNONYMS.get(w, w) for w in words if w not in STOPWORDS and len(w) > 2}


def _parse_month(value) -> int | None:
    """Accept 1-12, 'July', 'jul'. Returns None when not given; raises ValueError when invalid."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)) or str(value).strip().isdigit():
        m = int(value)
        if 1 <= m <= 12:
            return m
        raise ValueError(f"month must be 1-12, got {value}")
    text = str(value).strip().lower()
    for i, name in enumerate(MONTHS, 1):
        if text == name or (len(text) >= 3 and name.startswith(text)):
            return i
    raise ValueError(f"unrecognised month: {value!r}")


def season_months(season: str) -> set:
    """Months an experience runs, parsed from the catalog's season text."""
    text = str(season).lower()
    if text.startswith("year-round"):
        return set(range(1, 13))
    names = "|".join(MONTHS)
    m = re.search(rf"({names})\s*[-–]\s*({names})", text)
    if not m:
        return set(range(1, 13))
    start, end = MONTHS.index(m.group(1)) + 1, MONTHS.index(m.group(2)) + 1
    return set(range(start, end + 1)) if start <= end else set(range(start, 13)) | set(range(1, end + 1))


def _region_matches(row_region: str, wanted: str) -> bool:
    wanted = wanted.strip().lower()
    if wanted in REGION_ALIASES:
        return row_region in REGION_ALIASES[wanted]
    return wanted in row_region.lower() or row_region.lower() in wanted


def _relevance(row, terms: set) -> tuple:
    """Keyword score: matches in tags/name/category count double, description and suitability once."""
    strong = _tokens(" ".join([row.name, row.category, row.region, row.interests.replace(";", " ")]))
    weak = _tokens(" ".join([row.description, row.suitability]))
    hits = sorted(t for t in terms if t in strong or t in weak)
    return sum(2 if t in strong else 1 for t in hits), hits


def get_row(exp_id: str):
    """One catalog row as a named tuple (row.name is the experience name, row.id its ID)."""
    return next(load_catalog().loc[[exp_id]].itertuples(index=False))


def experience_card(row) -> dict:
    """Documented facts for one experience, straight from the catalog (nothing inferred)."""
    return {
        "id": row.id, "name": row.name, "region": row.region, "category": row.category,
        "duration": row.duration_label, "duration_hours": int(row.duration_hours),
        "price_usd_per_person": int(row.price_usd), "group_size": f"{row.min_group}-{row.max_group} people",
        "season": row.season, "included": row.inclusions, "good_for": row.suitability,
        "description": row.description, "accessibility": row.accessibility_notes, "food": row.food_notes,
        "transport": "Not documented; must be confirmed by the team.",
        "availability": "Inquiry only; the team confirms availability by email.",
    }


def search_experiences(query: str = None, interests: list = None, category: str = None, region: str = None,
                       max_price_per_person: float = None, max_duration_hours: float = None,
                       group_size: int = None, month=None, limit: int = 5) -> dict:
    """Filter and rank catalog experiences. Hard filters exclude; keywords rank (and exclude if nothing matches)."""
    df = load_catalog()
    try:
        month_num = _parse_month(month)
    except ValueError as exc:
        return {"status": "error", "error": "invalid_month", "message": str(exc)}
    terms = _tokens(" ".join([query or ""] + list(interests or [])))
    excluded, matches = Counter(), []
    for row in df.itertuples():
        out = []
        if category and row.category.lower() != category.strip().lower():
            out.append("category")
        if region and not _region_matches(row.region, region):
            out.append("region")
        if max_price_per_person is not None and row.price_usd > max_price_per_person:
            out.append("price")
        if max_duration_hours is not None and row.duration_hours > max_duration_hours:
            out.append("duration")
        if group_size is not None and not (row.min_group <= group_size <= row.max_group):
            out.append("group_size")
        if month_num and month_num not in season_months(row.season):
            out.append("season")
        score, hits = _relevance(row, terms)
        if terms and not out and score == 0:
            out.append("keywords")
        if out:
            excluded.update(out)
            continue
        matches.append((score, int(row.price_usd), row, hits))
    matches.sort(key=lambda m: (-m[0], m[1]))
    limit = max(1, min(int(limit or 5), 20))
    results = [dict(experience_card(r), matched_keywords=hits) for _, _, r, hits in matches[:limit]]
    filters = {k: v for k, v in dict(query=query, interests=interests, category=category, region=region,
                                     max_price_per_person=max_price_per_person,
                                     max_duration_hours=max_duration_hours, group_size=group_size,
                                     month=MONTHS[month_num - 1].title() if month_num else None).items() if v}
    out = {"status": "success", "total_matches": len(matches), "returned": len(results), "results": results,
           "filters_applied": filters, "excluded_by_filter": dict(excluded)}
    if not matches:
        out["note"] = ("No catalog experience matches all filters. Say so honestly, suggest relaxing the filter "
                       "that excluded the most options, and if the visitor wants something we do not offer, "
                       "log it with record_feedback.")
    return out


def _normalize_ids(ids) -> list:
    """Uppercase, strip, and de-duplicate experience IDs, keeping order ('d4' -> 'D04')."""
    out = []
    for raw in ids or []:
        m = re.fullmatch(r"\s*[dD]\s*0*(\d{1,2})\s*", str(raw))
        norm = f"D{int(m.group(1)):02d}" if m else str(raw).strip().upper()
        if norm not in out:
            out.append(norm)
    return out


def compare_experiences(experience_ids: list, group_size: int = None) -> dict:
    """Side-by-side comparison of 2-6 catalog experiences using documented fields only."""
    catalog = load_catalog()
    ids = _normalize_ids(experience_ids)
    found = [i for i in ids if i in catalog.index]
    not_found = [i for i in ids if i not in catalog.index]
    if len(found) < 2:
        return {"status": "error", "error": "need_two_valid_ids", "not_found": not_found,
                "message": "Provide at least two valid experience IDs (D01-D20); use search_experiences to find them."}
    if len(found) > 6:
        return {"status": "error", "error": "too_many_ids", "message": "Compare at most 6 experiences at a time."}

    rows = []
    for i in found:
        r = get_row(i)
        card = experience_card(r)
        if group_size:
            card["fits_group_size"] = bool(r.min_group <= group_size <= r.max_group)
            card["group_total_usd"] = int(r.price_usd) * group_size
        rows.append(card)

    prices = {c["id"]: c["price_usd_per_person"] for c in rows}
    hours = {c["id"]: c["duration_hours"] for c in rows}
    lo, hi = min(prices.values()), max(prices.values())
    summary = {
        "cheapest": [i for i, p in prices.items() if p == lo],
        "most_expensive": [i for i, p in prices.items() if p == hi],
        "price_difference_per_person_usd": hi - lo,
        "shortest": [i for i, h in hours.items() if h == min(hours.values())],
        "longest": [i for i, h in hours.items() if h == max(hours.values())],
        "same_region": len({c["region"] for c in rows}) == 1,
        "same_category": len({c["category"] for c in rows}) == 1,
    }
    if group_size:
        summary["group_size"] = group_size
        summary["not_suitable_for_group_size"] = [c["id"] for c in rows if not c["fits_group_size"]]
    unknown = ["transport", "availability on specific dates"]
    if any(c["accessibility"].startswith("Not documented") for c in rows):
        unknown.append("accessibility (not documented for some or all options)")
    if any("not documented" in c["food"].lower() for c in rows):
        unknown.append("dietary suitability (not documented for some options)")
    return {"status": "success", "compared": rows, "summary": summary, "not_found": not_found,
            "unknown_for_all_options": unknown,
            "note": "Unknown is not the same as no: do not claim suitability that is not documented."}


# Broad areas used to keep each day geographically sensible (travel times themselves are not documented).
AREA_OF_REGION = {"Tripoli": "North", "Bcharre": "North", "Qadisha Valley": "North", "Batroun": "North",
                  "Byblos": "Central coast", "Keserwan": "Central coast", "Metn": "Central coast",
                  "Beirut": "Central coast", "Chouf": "Chouf and South", "Jezzine": "Chouf and South",
                  "Sidon": "Chouf and South", "Tyre": "Chouf and South", "Bekaa": "Bekaa"}


def area_of(region: str) -> str:
    return AREA_OF_REGION.get(region, region)


class _DayPlanner:
    """Each day has a Morning and an Afternoon slot. Half-day items take one slot, full-day items take
    both slots of one day, and the overnight stay takes the afternoon of day N plus the morning of day N+1."""

    def __init__(self, days: int, max_per_day: int):
        self.days = [{"Morning": None, "Afternoon": None} for _ in range(days)]
        self.max_per_day = max_per_day

    def _count(self, d: int) -> int:
        return len({e["id"] for e in self.days[d].values() if e})

    def _entry(self, row, slot: str, **extra) -> dict:
        return dict(slot=slot, id=row.id, name=row.name, region=row.region, category=row.category,
                    duration=row.duration_label, price_usd_per_person=int(row.price_usd), **extra)

    def place(self, row) -> str | None:
        """Place an experience; return None on success or the reason it does not fit."""
        n = len(self.days)
        if row.category == "Overnight":
            if n < 2:
                return "an overnight stay needs a trip of at least 2 days"
            for d in range(n - 1):
                a, b = self.days[d], self.days[d + 1]
                if (a["Afternoon"] is None and b["Morning"] is None
                        and self._count(d) < self.max_per_day and self._count(d + 1) < self.max_per_day):
                    a["Afternoon"] = self._entry(row, "Afternoon, overnight stay")
                    b["Morning"] = self._entry(row, "Morning, end of overnight stay", continued=True)
                    return None
            return "no free afternoon followed by a free morning"
        if int(row.day_slots) == 2:
            for d in range(n):
                if self.days[d]["Morning"] is None and self.days[d]["Afternoon"] is None:
                    entry = self._entry(row, "Full day")
                    self.days[d]["Morning"] = self.days[d]["Afternoon"] = entry
                    return None
            return "no completely free day left for a full-day experience"
        # Half-day: prefer a part-filled day in the same area, then an empty day, then any free slot.
        def free_slot(d):
            return next((s for s in ("Morning", "Afternoon") if self.days[d][s] is None), None)

        same_area = [d for d in range(n) if self._count(d) and free_slot(d) and self._count(d) < self.max_per_day
                     and any(e and area_of(e["region"]) == area_of(row.region) for e in self.days[d].values())]
        empty = [d for d in range(n) if self._count(d) == 0]
        other = [d for d in range(n) if free_slot(d) and self._count(d) < self.max_per_day]
        for d in same_area + empty + other:
            slot = free_slot(d)
            if slot:
                self.days[d][slot] = self._entry(row, slot)
                return None
        return "no free half-day slot left"

    def open_areas(self) -> set:
        """Areas of days that still have one free half-day slot (good places for a nearby activity)."""
        out = set()
        for d, day in enumerate(self.days):
            if self._count(d) == 1 and self._count(d) < self.max_per_day and None in day.values():
                out |= {area_of(e["region"]) for e in day.values() if e}
        return out

    def schedule(self) -> list:
        out = []
        for i, day in enumerate(self.days, 1):
            acts, seen = [], set()
            for slot in ("Morning", "Afternoon"):
                e = day[slot]
                if e and (e["id"], e["slot"]) not in seen:
                    seen.add((e["id"], e["slot"]))
                    acts.append(e)
            out.append({"day": i, "activities": acts})
        return out


def create_itinerary(days: int, budget_per_person: float = None, total_budget: float = None,
                     group_size: int = None, interests: list = None, region: str = None, month=None,
                     experience_ids: list = None, max_activities_per_day: int = 2) -> dict:
    """Suggest a day-by-day plan from the catalog that respects hard constraints, revising when needed."""
    catalog = load_catalog()
    if not 1 <= int(days) <= 7:
        return {"status": "error", "error": "invalid_days", "message": "days must be between 1 and 7."}
    days, max_per_day = int(days), max(1, min(int(max_activities_per_day or 2), 2))
    try:
        month_num = _parse_month(month)
    except ValueError as exc:
        return {"status": "error", "error": "invalid_month", "message": str(exc)}
    assumptions, revisions, warnings = [], [], []

    budget = budget_per_person
    if total_budget is not None:
        if not group_size:
            return {"status": "error", "error": "group_size_required",
                    "message": "A total budget needs the group size to convert it to a per-person budget. Ask the visitor."}
        per_person = total_budget / group_size
        budget = min(budget, per_person) if budget is not None else per_person
        assumptions.append(f"Total budget ${total_budget:g} for {group_size} people = ${per_person:.2f} per person.")
    if budget is not None and budget < 0:
        return {"status": "error", "error": "invalid_budget", "message": "Budget cannot be negative."}

    def hard_problems(row) -> list:
        probs = []
        if group_size and not (row.min_group <= group_size <= row.max_group):
            probs.append(f"group size {group_size} is outside its {row.min_group}-{row.max_group} range")
        if month_num and month_num not in season_months(row.season):
            probs.append(f"it does not run in {MONTHS[month_num - 1].title()} (season: {row.season})")
        return probs

    planner, chosen, spent = _DayPlanner(days, max_per_day), [], 0

    def try_add(row, label=None) -> bool:
        """Add one experience if it fits the budget and the schedule; record a revision only on success."""
        nonlocal spent
        if budget is not None and spent + row.price_usd > budget:
            return False
        if planner.place(row):
            return False
        chosen.append(row)
        spent += int(row.price_usd)
        if label:
            revisions.append(label.format(id=row.id, name=row.name, price=int(row.price_usd)))
        return True

    # Candidate pool for automatic selection: eligible, in the region, matching the interests.
    terms = _tokens(" ".join(interests or []))
    scores = {}
    for row in catalog.itertuples():
        if hard_problems(row) or (region and not _region_matches(row.region, region)):
            continue
        score, _ = _relevance(row, terms)
        if terms and score == 0:
            continue
        scores[row.id] = score
    pool = [get_row(i) for i in scores]

    def fill(candidates, label=None):
        remaining = [r for r in candidates if r.id not in {c.id for c in chosen}]
        while remaining:
            used = Counter(c.category for c in chosen)
            near = planner.open_areas()
            remaining.sort(key=lambda r: (-(scores.get(r.id, 0) - 2 * used[r.category]
                                            + (1.5 if area_of(r.region) in near else 0)), r.price_usd))
            for r in remaining:
                if try_add(r, label):
                    remaining.remove(r)
                    break
            else:
                break

    requested = _normalize_ids(experience_ids)
    not_found = [i for i in requested if i not in catalog.index]
    if requested:
        selected = []
        for i in [i for i in requested if i in catalog.index]:
            row = get_row(i)
            probs = hard_problems(row)
            if probs:
                revisions.append(f"Removed {i} {row.name}: {'; '.join(probs)}.")
            else:
                selected.append(row)
        if budget is not None:
            while selected and sum(r.price_usd for r in selected) > budget:
                drop = max(selected, key=lambda r: r.price_usd)
                selected.remove(drop)
                revisions.append(f"Removed {drop.id} {drop.name} (${int(drop.price_usd)} per person) so the "
                                 f"plan stays within the ${budget:g} per-person budget.")
        for row in selected:
            if not try_add(row):
                probe = _DayPlanner(days, max_per_day)  # explain why it did not fit on an empty schedule
                reason = probe.place(row) or "no free slot left after the other requested experiences"
                revisions.append(f"Could not schedule {row.id} {row.name}: {reason}.")
        if revisions:  # something was removed: offer catalog replacements that fit what is left
            fill([r for r in pool if r.id not in requested],
                 label="Added {id} {name} (${price} per person) as a replacement that fits the constraints.")
    else:
        fill(pool)

    total_pp = sum(int(r.price_usd) for r in chosen)
    plan = planner.schedule()
    for day in plan:
        regions = {a["region"] for a in day["activities"]}
        if not day["activities"]:
            warnings.append(f"Day {day['day']} has no activities (free day).")
        elif len({area_of(r) for r in regions}) > 1:
            warnings.append(f"Day {day['day']} combines {' and '.join(sorted(regions))}, which are in different "
                            "parts of the country; travel time is not documented, so the team should confirm it.")
    if group_size is None:
        assumptions.append("Group size not given: totals are per person.")
    assumptions += ["Transport, meals, and entry tickets are not included unless an experience lists them.",
                    "Morning/afternoon slots are indicative; opening hours and travel times are not documented.",
                    "This is a suggested plan, not a booking; the team confirms availability by email."]

    checks = [{"constraint": f"fits in {days} day(s)", "satisfied": True,
               "detail": f"{len(chosen)} experience(s) scheduled"}]
    if budget is not None:
        checks.append({"constraint": f"budget ${budget:g} per person", "satisfied": total_pp <= budget,
                       "detail": f"estimated ${total_pp} per person, ${budget - total_pp:g} remaining"})
    if group_size:
        checks.append({"constraint": f"group of {group_size}", "satisfied": not any(hard_problems(r) for r in chosen),
                       "detail": "every experience accepts this group size"})
    if month_num:
        checks.append({"constraint": f"runs in {MONTHS[month_num - 1].title()}", "satisfied": True,
                       "detail": "every experience is in season"})

    result = {"status": "success", "feasible": bool(chosen) and all(c["satisfied"] for c in checks),
              "days": plan, "estimated_total_per_person_usd": total_pp,
              "estimated_group_total_usd": total_pp * group_size if group_size else None,
              "constraint_checks": checks, "revisions": revisions, "warnings": warnings,
              "assumptions": assumptions, "not_found": not_found}
    if not chosen:
        if not pool and not requested:
            why = "no catalog experience matches the requested interests, region, season, and group size"
        elif budget is not None:
            cheapest = min((r.price_usd for r in (pool or [get_row(i) for i in requested if i in catalog.index])),
                           default=None)
            why = (f"the cheapest matching experience costs ${int(cheapest)} per person, above the ${budget:g} budget"
                   if cheapest is not None and cheapest > budget else "the constraints cannot all be met")
        else:
            why = "the constraints cannot all be met"
        result["infeasible_reason"] = why
        result["note"] = "Explain why no plan fits and ask which constraint the visitor can relax."
    return result


# Tool definitions in MCP style: name, description, inputSchema (JSON Schema), and outputSchema.
_RESULT_STATUS = {"status": {"type": "string", "enum": ["success", "error"]},
                  "error": {"type": "string", "description": "error code when status is error"},
                  "message": {"type": "string"}}
_EXPERIENCE = {"type": "object", "description": "documented catalog facts for one experience",
               "properties": {k: {"type": "string"} for k in
                              ["id", "name", "region", "category", "duration", "group_size", "season", "included",
                               "good_for", "description", "accessibility", "food", "transport", "availability"]}
               | {"duration_hours": {"type": "integer"}, "price_usd_per_person": {"type": "integer"}}}
_INTERESTS = {"type": "array", "items": {"type": "string"},
              "description": "visitor interests, e.g. ['nature', 'food', 'history']"}
_MONTH = {"type": "string", "description": "travel month as a name or number, e.g. 'July' or '7'"}
_GROUP = {"type": "integer", "minimum": 1, "maximum": 50, "description": "number of people"}

TOOL_SPECS = [
    {
        "name": "record_customer_interest",
        "description": ("Save a visitor's booking inquiry so the team can follow up by email. Call ONLY when you "
                        "have the visitor's real name, a valid email, the number of people, and what they are "
                        "interested in. Returns an inquiry_id. An inquiry is not a confirmed reservation."),
        "inputSchema": {"type": "object", "required": ["email", "name", "message", "group_size"],
                        "additionalProperties": False,
                        "properties": {
                            "email": {"type": "string", "format": "email", "description": "the visitor's email address"},
                            "name": {"type": "string", "description": "the visitor's name as they gave it"},
                            "group_size": {"type": "integer", "minimum": 1, "maximum": 50,
                                           "description": "how many people will join, as the visitor said it"},
                            "message": {"type": "string", "description": "what they want: experience IDs, dates, notes"}}},
        "outputSchema": {"type": "object", "properties": _RESULT_STATUS | {
            "inquiry_id": {"type": "string", "description": "e.g. INQ-20261011-A1B2C3"},
            "saved_at": {"type": "string", "description": "UTC timestamp"},
            "fields": {"type": "object", "description": "which inputs failed validation and why"},
            "next_step": {"type": "string"}}},
    },
    {
        "name": "record_feedback",
        "description": ("Log a question you cannot answer from the business knowledge or tools (transport, "
                        "accessibility, dietary needs, safety, payment, services we do not offer), or a complaint, "
                        "correction, suggestion, or praise. Returns a feedback_id. Only for a question the visitor "
                        "actually asked or feedback they gave: never to save a booking request, to note missing "
                        "contact details, or to ask about availability for a requested date (the inquiry covers that)."),
        "inputSchema": {"type": "object", "required": ["question"], "additionalProperties": False,
                        "properties": {
                            "question": {"type": "string", "description": "the visitor's question or feedback, in full"},
                            "kind": {"type": "string", "enum": list(FEEDBACK_KINDS),
                                     "description": "type of entry; default unanswered_question"}}},
        "outputSchema": {"type": "object", "properties": _RESULT_STATUS | {
            "feedback_id": {"type": "string", "description": "e.g. FB-20261011-D4E5F6"},
            "saved_at": {"type": "string"}, "next_step": {"type": "string"}}},
    },
    {
        "name": "search_experiences",
        "description": ("Search the experience catalog. All filters are optional: keywords and interests rank "
                        "results; category, region, price, duration, group size, and month exclude experiences "
                        "that do not fit. Returns documented catalog facts only."),
        "inputSchema": {"type": "object", "additionalProperties": False, "properties": {
            "query": {"type": "string", "description": "free-text keywords, e.g. 'cooking village'"},
            "interests": _INTERESTS,
            "category": {"type": "string", "enum": ["Culture", "History", "Nature", "Food", "Craft", "Coast",
                                                    "Adventure", "Overnight"]},
            "region": {"type": "string", "description": "e.g. Chouf, Batroun, Bekaa, Beirut, or 'north' / 'south'"},
            "max_price_per_person": {"type": "number", "minimum": 0, "description": "USD per person"},
            "max_duration_hours": {"type": "number", "minimum": 0},
            "group_size": _GROUP, "month": _MONTH,
            "limit": {"type": "integer", "minimum": 1, "maximum": 20, "description": "max results, default 5"}}},
        "outputSchema": {"type": "object", "properties": _RESULT_STATUS | {
            "total_matches": {"type": "integer"}, "results": {"type": "array", "items": _EXPERIENCE},
            "filters_applied": {"type": "object"},
            "excluded_by_filter": {"type": "object", "description": "how many experiences each filter excluded"},
            "note": {"type": "string"}}},
    },
    {
        "name": "compare_experiences",
        "description": ("Compare 2-6 catalog experiences side by side on price, duration, region, season, "
                        "inclusions, group size, and notes. Marks what is unknown. Optionally computes group totals."),
        "inputSchema": {"type": "object", "required": ["experience_ids"], "additionalProperties": False,
                        "properties": {
                            "experience_ids": {"type": "array", "items": {"type": "string"}, "minItems": 2,
                                               "maxItems": 6, "description": "catalog IDs, e.g. ['D04', 'D08']"},
                            "group_size": _GROUP}},
        "outputSchema": {"type": "object", "properties": _RESULT_STATUS | {
            "compared": {"type": "array", "items": _EXPERIENCE},
            "summary": {"type": "object", "description": "cheapest, most_expensive, shortest, longest, price difference"},
            "unknown_for_all_options": {"type": "array", "items": {"type": "string"}},
            "not_found": {"type": "array", "items": {"type": "string"}}}},
    },
    {
        "name": "create_itinerary",
        "description": ("Build a suggested day-by-day plan from the catalog. Enforces hard constraints (days, "
                        "budget, group size, season), removes or replaces experiences that break them, and explains "
                        "when no plan fits. Give experience_ids to plan specific experiences, or interests/region to "
                        "let the tool choose. The plan is a suggestion, not a booking."),
        "inputSchema": {"type": "object", "required": ["days"], "additionalProperties": False, "properties": {
            "days": {"type": "integer", "minimum": 1, "maximum": 7, "description": "trip length in days"},
            "budget_per_person": {"type": "number", "minimum": 0, "description": "USD per person for all activities"},
            "total_budget": {"type": "number", "minimum": 0, "description": "USD for the whole group (needs group_size)"},
            "group_size": _GROUP, "interests": _INTERESTS,
            "region": {"type": "string", "description": "limit to a region, e.g. 'north' or 'Chouf'"},
            "month": _MONTH,
            "experience_ids": {"type": "array", "items": {"type": "string"},
                               "description": "specific experiences to include, e.g. ['D01', 'D03']"},
            "max_activities_per_day": {"type": "integer", "minimum": 1, "maximum": 2, "description": "default 2"}}},
        "outputSchema": {"type": "object", "properties": _RESULT_STATUS | {
            "feasible": {"type": "boolean"},
            "days": {"type": "array", "description": "[{day, activities: [{slot, id, name, region, price_usd_per_person}]}]"},
            "estimated_total_per_person_usd": {"type": "integer"},
            "estimated_group_total_usd": {"type": ["integer", "null"]},
            "constraint_checks": {"type": "array"}, "revisions": {"type": "array", "items": {"type": "string"}},
            "warnings": {"type": "array"}, "assumptions": {"type": "array"},
            "infeasible_reason": {"type": "string"}}},
    },
]

TOOL_FUNCTIONS = {"record_customer_interest": record_customer_interest, "record_feedback": record_feedback,
                  "search_experiences": search_experiences, "compare_experiences": compare_experiences,
                  "create_itinerary": create_itinerary}

SPEC_BY_NAME = {s["name"]: s for s in TOOL_SPECS}


def validate_arguments(schema: dict, args: dict) -> tuple:
    """Check arguments against a tool's inputSchema. Returns (cleaned_args, errors).
    Light coercion: '4' -> 4 for integers and numbers, 'D04' -> ['D04'] for string arrays."""
    props, errors, clean = schema.get("properties", {}), [], {}
    for name in schema.get("required", []):
        if args.get(name) in (None, "", []):
            errors.append(f"missing required argument '{name}'")
    for name, value in args.items():
        if name not in props:
            errors.append(f"unexpected argument '{name}'")
            continue
        if value is None:
            continue
        spec, kind = props[name], props[name].get("type")
        try:
            if kind == "integer":
                if isinstance(value, bool) or float(value) != int(float(value)):
                    raise ValueError
                value = int(float(value))
            elif kind == "number":
                if isinstance(value, bool):
                    raise ValueError
                value = float(value)
            elif kind == "string":
                if not isinstance(value, (str, int, float)):
                    raise ValueError
                value = str(value)
            elif kind == "array":
                value = [value] if isinstance(value, str) else value
                if not isinstance(value, list):
                    raise ValueError
                value = [str(v) for v in value]
        except (TypeError, ValueError):
            errors.append(f"'{name}' must be of type {kind}")
            continue
        if "enum" in spec and value not in spec["enum"]:
            matches = [e for e in spec["enum"] if str(e).lower() == str(value).lower()]
            if matches:
                value = matches[0]
            else:
                errors.append(f"'{name}' must be one of {spec['enum']}")
                continue
        if "minimum" in spec and value < spec["minimum"] or "maximum" in spec and value > spec["maximum"]:
            errors.append(f"'{name}' must be between {spec.get('minimum')} and {spec.get('maximum')}")
            continue
        if kind == "array" and ("minItems" in spec and len(value) < spec["minItems"]
                                or "maxItems" in spec and len(value) > spec["maxItems"]):
            errors.append(f"'{name}' must have {spec.get('minItems', 0)}-{spec.get('maxItems', 'any')} items")
            continue
        clean[name] = value
    return clean, errors




def _redact(args: dict) -> dict:
    return {k: mask_email(v) if k == "email" else v for k, v in args.items()}


def execute_tool(name: str, arguments) -> dict:
    """Validate and run one tool call. Always returns a dict with a 'status'; never raises."""
    start = time.perf_counter()
    if name not in TOOL_FUNCTIONS:
        result, args = {"status": "error", "error": "unknown_tool",
                        "message": f"No tool named {name!r}. Available: {list(TOOL_FUNCTIONS)}"}, {}
    else:
        try:
            args = json.loads(arguments or "{}") if isinstance(arguments, str) else dict(arguments or {})
        except (json.JSONDecodeError, TypeError, ValueError):
            args = None
        if not isinstance(args, dict):
            result, args = {"status": "error", "error": "invalid_json",
                            "message": "Tool arguments were not a valid JSON object."}, {}
        else:
            # Smaller models often fill every optional field with null, "" or []: treat those as "not given".
            args = {k: v for k, v in args.items() if v is not None and v != "" and v != []}
            clean, problems = validate_arguments(SPEC_BY_NAME[name]["inputSchema"], args)
            if problems:
                result = {"status": "error", "error": "invalid_arguments", "details": problems,
                          "message": "Tool NOT executed. Fix the arguments (ask the visitor if information is missing)."}
            else:
                try:
                    result = TOOL_FUNCTIONS[name](**clean)
                except Exception as exc:  # a tool bug must not crash the conversation
                    result = {"status": "error", "error": "tool_exception",
                              "message": f"{type(exc).__name__}: {exc}"[:300]}
    entry = {"timestamp": _now_iso(), "tool": name, "arguments": _redact(args), "status": result.get("status"),
             "error": result.get("error"), "latency_s": round(time.perf_counter() - start, 3)}
    if VERBOSE:
        print(f"[TOOL] {name}({json.dumps(entry['arguments'], ensure_ascii=False)[:160]}) -> {entry['status']}"
              + (f" ({entry['error']})" if entry["error"] else ""))
    return result


def _parse_args(raw) -> dict:
    try:
        parsed = json.loads(raw or "{}") if isinstance(raw, str) else dict(raw or {})
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


def lead_details_not_from_visitor(args: dict, visitor_text: str) -> list:
    """Which contact details in a record_customer_interest call the visitor never typed ('email', 'name').

    Models sometimes fill a missing email with a plausible guess (e.g. firstname.lastname@example.com); that
    must never be saved. The email has to appear in the visitor's own messages, and so does at least one part of
    the name. Malformed emails are left to the normal argument validation."""
    text = visitor_text.casefold()
    words = set(re.findall(r"\w+", text))
    email = str(args.get("email", "")).strip().casefold()
    if not EMAIL_VALID_RE.match(email):
        return []
    missing = [] if email in text else ["email"]
    name_parts = [w for w in re.findall(r"\w+", str(args.get("name", "")).casefold()) if len(w) > 1]
    if name_parts and not any(w in words for w in name_parts):
        missing.append("name")
    return missing


NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20}
_GROUP_PHRASES = [  # (pattern, people) for the ways visitors say a group size without a number
    (r"\b(just me|only me|myself|alone|solo|by myself|one person|for me)\b", 1),
    (r"\b(couple|the two of us|both of us|my (wife|husband|partner|friend|sister|brother|mother|father|mom|dad|"
     r"son|daughter|fianc[eé]e?) and (i|me)|me and my \w+|\w+ and (i|me))\b", 2),
]


def group_size_from_visitor(group_size, visitor_text: str) -> bool:
    """Did the visitor state this number of people (as a digit, a number word, or a phrase like 'my sister and
    me')? Stops the model from assuming a group size the visitor never gave."""
    try:
        n = int(group_size)
    except (TypeError, ValueError):
        return True  # a missing or malformed value is left to the argument validation
    text = visitor_text.casefold()
    if re.search(rf"(?<![\d.$]){n}(?![\d.])", text) or any(v == n and re.search(rf"\b{w}\b", text)
                                                            for w, v in NUMBER_WORDS.items()):
        return True
    return any(v == n and re.search(pattern, text) for pattern, v in _GROUP_PHRASES)


def is_loggable(message: str, route=None) -> bool:
    """May this visitor message be logged with record_feedback?

    Yes if the router saw an unanswered question or feedback in it (primary or secondary intent), or if the
    message itself mentions something only the team can answer (transport, diet, access, payment...) or reads
    as feedback (a complaint, a suggestion). General questions (weather, restaurants), typo IDs, and booking
    requests waiting for an email are not team questions, so they are not logged."""
    intents = {route.intent, *route.secondary} if route is not None else set()
    return ("FEEDBACK_UNKNOWN" in intents or bool(UNKNOWN_RE.search(message or ""))
            or bool(FEEDBACK_RE.search(message or "")))


def guarded_execute(name: str, arguments, visitor_text: str, message: str = "", route=None) -> dict:
    """execute_tool(), with two checks Python enforces whatever the model decides: an inquiry is refused when
    its contact details did not come from the visitor, and feedback is refused when the visitor's message is
    not an unanswered question or feedback."""
    if name == "record_feedback" and route is not None and not is_loggable(message, route):
        print("[TOOL] record_feedback -> refused (the message is not an unanswered question or feedback)")
        return {"status": "error", "error": "not_feedback",
                "message": ("Tool NOT executed: the visitor's message is not a question about our services that the "
                            "documents cannot answer, and not feedback, so it must not be logged. Answer it directly: "
                            "give general information, ask for missing details, or ask which experience they meant.")}
    if name == "record_customer_interest":
        args = _parse_args(arguments)
        missing = lead_details_not_from_visitor(args, visitor_text)
        if args.get("group_size") is not None and not group_size_from_visitor(args["group_size"], visitor_text):
            missing.append("number of people")
        if missing:
            print(f"[TOOL] record_customer_interest -> refused (not given by the visitor: {', '.join(missing)})")
            return {"status": "error", "error": "details_not_from_visitor", "missing": missing,
                    "message": (f"Tool NOT executed: the visitor has not given their {' and '.join(missing)} in this "
                                "conversation. Ask the visitor for it. Never invent, guess, or complete these details.")}
    return execute_tool(name, arguments)


# =============================================================================================== exact arithmetic and seasons
def budget_calculator(experience_ids: list, group_size: int = 1, budget_per_person: float = None,
                      total_budget: float = None) -> dict:
    """Exact price arithmetic for a set of experiences, so the model never does sums in its head."""
    ids = _normalize_ids(experience_ids)
    catalog = load_catalog()
    missing = [i for i in ids if i not in catalog.index]
    if missing:
        return {"status": "error", "error": "unknown_experience", "not_found": missing,
                "message": f"Not in the catalog: {', '.join(missing)}. Ask the visitor which experience they meant."}
    if not ids:
        return {"status": "error", "error": "no_experiences", "message": "Give at least one experience ID."}
    group_size = int(group_size or 1)
    lines = [{"id": i, "name": catalog.loc[i, "name"], "price_usd_per_person": int(catalog.loc[i, "price_usd"]),
              "group_total_usd": int(catalog.loc[i, "price_usd"]) * group_size} for i in ids]
    per_person = sum(line["price_usd_per_person"] for line in lines)
    result = {"status": "success", "items": lines, "group_size": group_size,
              "total_per_person_usd": per_person, "total_for_group_usd": per_person * group_size,
              "calculation": f"({' + '.join(str(line['price_usd_per_person']) for line in lines)}) x {group_size} "
                             f"= {per_person * group_size} USD"}
    if budget_per_person is not None:
        result["within_budget_per_person"] = per_person <= budget_per_person
        result["difference_per_person_usd"] = round(budget_per_person - per_person, 2)
    if total_budget is not None:
        result["within_total_budget"] = per_person * group_size <= total_budget
        result["difference_total_usd"] = round(total_budget - per_person * group_size, 2)
    return result


_DATE_FORMATS = ("%Y-%m-%d", "%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y", "%d %B", "%d %b", "%B %d", "%b %d")


def _parse_date(value: str, today: date | None = None) -> date | None:
    """Parse '2026-11-21', '21 November', 'Nov 21 2026'...; a date without a year is the next upcoming one."""
    text = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", str(value).strip().replace(",", " "))
    text = " ".join(text.split())
    today = today or date.today()
    for fmt in _DATE_FORMATS:
        has_year = "%Y" in fmt
        try:  # a year is always added before parsing, so 29 February parses in leap years
            parsed = datetime.strptime(text if has_year else f"{text} {today.year}",
                                       fmt if has_year else f"{fmt} %Y").date()
        except ValueError:
            continue
        if not has_year and parsed < today:
            try:
                parsed = parsed.replace(year=today.year + 1)
            except ValueError:  # 29 February in a non-leap next year
                parsed = parsed.replace(year=today.year + 1, day=28)
        return parsed
    return None


def season_check(experience_id: str, when: str) -> dict:
    """Does an experience run in a given month or on a given date? Uses the catalog's season only: availability
    on a specific date is always confirmed by the team."""
    ids = _normalize_ids([experience_id])
    catalog = load_catalog()
    if not ids or ids[0] not in catalog.index:
        return {"status": "error", "error": "unknown_experience",
                "message": f"{experience_id!r} is not in the catalog. Ask the visitor which experience they meant."}
    row = catalog.loc[ids[0]]
    day = _parse_date(when)
    try:
        month = day.month if day else _parse_month(when)
    except ValueError:
        month = None
    if not month:
        return {"status": "error", "error": "invalid_date",
                "message": "Give a month (e.g. 'November') or a date (e.g. '21 November' or '2026-11-21')."}
    runs = month in season_months(row["season"])
    return {"status": "success", "id": ids[0], "name": row["name"], "season": row["season"],
            "checked": day.isoformat() if day else MONTHS[month - 1].title(), "runs_that_month": runs,
            "note": ("Within the documented season; availability on the exact date is confirmed by the team."
                     if runs else "Outside the documented season, so it does not run then; suggest another "
                                  "month or a similar experience that runs then.")}


TOOL_SPECS += [
    {
        "name": "budget_calculator",
        "description": ("Exact price arithmetic: per-person and group totals for one or more catalog experiences, "
                        "and whether they fit a budget. Use it instead of doing sums yourself."),
        "inputSchema": {"type": "object", "required": ["experience_ids"], "additionalProperties": False,
                        "properties": {
                            "experience_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1,
                                               "maxItems": 10, "description": "catalog IDs, e.g. ['D04', 'D08']"},
                            "group_size": _GROUP,
                            "budget_per_person": {"type": "number", "minimum": 0, "description": "USD per person"},
                            "total_budget": {"type": "number", "minimum": 0, "description": "USD for the group"}}},
        "outputSchema": {"type": "object", "properties": _RESULT_STATUS | {
            "items": {"type": "array"}, "total_per_person_usd": {"type": "integer"},
            "total_for_group_usd": {"type": "integer"}, "calculation": {"type": "string"},
            "within_budget_per_person": {"type": "boolean"}, "within_total_budget": {"type": "boolean"}}},
    },
    {
        "name": "season_check",
        "description": ("Check whether one catalog experience runs in a given month or on a given date, from its "
                        "documented season. Availability on an exact date is still confirmed by the team."),
        "inputSchema": {"type": "object", "required": ["experience_id", "when"], "additionalProperties": False,
                        "properties": {
                            "experience_id": {"type": "string", "description": "a catalog ID, e.g. 'D20'"},
                            "when": {"type": "string", "description": "a month or date, e.g. 'November' or '21 November'"}}},
        "outputSchema": {"type": "object", "properties": _RESULT_STATUS | {
            "runs_that_month": {"type": "boolean"}, "season": {"type": "string"}, "checked": {"type": "string"},
            "note": {"type": "string"}}},
    },
]
TOOL_FUNCTIONS |= {"budget_calculator": budget_calculator, "season_check": season_check}
SPEC_BY_NAME = {s["name"]: s for s in TOOL_SPECS}


def tool_catalog_text() -> str:
    """The tools as they are described inside the ReAct system prompt: name, purpose, and arguments."""
    blocks = []
    for spec in TOOL_SPECS:
        props, required = spec["inputSchema"].get("properties", {}), set(spec["inputSchema"].get("required", []))
        args = []
        for name, prop in props.items():
            kind = prop.get("type", "any")
            if kind == "array":
                kind = f"list of {prop.get('items', {}).get('type', 'any')}"
            extra = f" one of {prop['enum']}" if "enum" in prop else ""
            desc = f" - {prop['description']}" if prop.get("description") else ""
            args.append(f"    - {name} ({kind}{', required' if name in required else ''}){extra}{desc}")
        blocks.append(f"{spec['name']}: {spec['description']}\n  Arguments:\n" + ("\n".join(args) or "    (none)"))
    return "\n\n".join(blocks)

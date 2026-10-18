"""Offline tests for the tools, the guards, and the rule-based router (no API calls).

    python -m pytest tests_tools.py -q
"""
import itertools
import json
import re

import pandas as pd
import pytest
from pathlib import Path

from douroub import router, tools

G, L, F = "GENERAL_QUESTION", "LEAD_COLLECTION", "FEEDBACK_UNKNOWN"
CATALOG = pd.read_csv(Path(__file__).resolve().parent / "me" / "experiences.csv").set_index("id")


def _placed(plan):
    return [a for d in plan["days"] for a in d["activities"] if not a.get("continued")]


@pytest.fixture(autouse=True)
def isolated_logs(tmp_path, monkeypatch):
    """Every test writes logs to a temporary folder and starts with an empty de-duplication memory."""
    monkeypatch.setattr(tools, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(tools, "VERBOSE", True)
    tools._RECENT.clear()
    return tmp_path / "logs"


def _lines(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []


@pytest.mark.parametrize("msg,last,intent,secondary", [
    ("my email is a.b@example.com", None, "LEAD_COLLECTION", []),
    ("Is pickup from the airport included?", None, "FEEDBACK_UNKNOWN", []),
    ("Karim", "Could you share your name and email?", "LEAD_COLLECTION", []),
    ("Karim", None, "GENERAL_QUESTION", []),
    ("Book D03 for 2, is it vegan?", None, "LEAD_COLLECTION", ["FEEDBACK_UNKNOWN"]),
    ("What's in D09?", None, "GENERAL_QUESTION", []),
])
def test_rules_router_key_cases(msg, last, intent, secondary):
    r = router.route_with_rules(msg, last)
    assert (r.intent, r.secondary, r.method) == (intent, secondary, "rules")


def test_parse_router_json_accepts_fenced_and_rejects_invalid():
    raw = ('```json\n{"intent": "lead_collection", "secondary": ["FEEDBACK_UNKNOWN", "LEAD_COLLECTION", "X"], '
           '"reason": "wants to book"}\n```')
    assert router._parse_router_json(raw) == ("LEAD_COLLECTION", ["FEEDBACK_UNKNOWN"], "wants to book")
    for bad in ["", "LEAD", '{"intent": "BOOKING"}']:
        with pytest.raises(ValueError):
            router._parse_router_json(bad)


def test_lead_saved_with_id_timestamp_and_push(isolated_logs, capsys):
    r = tools.record_customer_interest("Lina@Example.com ", " Lina  Haddad", "D03 for 2 people on Sunday", group_size=2)
    assert r["status"] == "success" and re.fullmatch(r"INQ-\d{8}-[0-9A-F]{6}", r["inquiry_id"])
    assert "not a confirmed reservation" in r["next_step"]
    saved = _lines(isolated_logs / "inquiries.jsonl")
    assert saved == [{"inquiry_id": r["inquiry_id"], "timestamp": r["saved_at"], "name": "Lina Haddad",
                      "email": "lina@example.com", "group_size": 2, "message": "D03 for 2 people on Sunday",
                      "status": "new"}]
    out = capsys.readouterr().out
    assert "[PUSH] New inquiry" in out and "l***@example.com" in out and "lina@example.com" not in out


@pytest.mark.parametrize("email,name,message,bad_field", [
    ("lina@", "Lina", "D03 for two people", "email"),
    ("lina.example.com", "Lina", "D03 for two people", "email"),
    ("lina@example", "Lina", "D03 for two people", "email"),
    ("", "Lina", "D03 for two people", "email"),
    ("lina@example.com", "unknown", "D03 for two people", "name"),
    ("lina@example.com", "", "D03 for two people", "name"),
    ("lina@example.com", "L", "D03 for two people", "name"),
    ("lina@example.com", "Lina", "hi", "message"),
    ("lina@example.com", "Lina", "", "message"),
])
def test_lead_validation_rejects_and_saves_nothing(isolated_logs, email, name, message, bad_field):
    r = tools.record_customer_interest(email, name, message, group_size=2)
    assert r["status"] == "error" and r["error"] == "validation_failed" and bad_field in r["fields"]
    assert "NOT saved" in r["message"] and "inquiry_id" not in r
    assert not (isolated_logs / "inquiries.jsonl").exists()


def test_duplicate_lead_is_saved_once(isolated_logs):
    a = tools.record_customer_interest("lina@example.com", "Lina", "D03 for two people", group_size=2)
    b = tools.record_customer_interest("LINA@example.com", "Lina", "D03 for two  people", group_size=2)
    assert a["inquiry_id"] == b["inquiry_id"] and b["duplicate"] is True
    assert len(_lines(isolated_logs / "inquiries.jsonl")) == 1


def test_unique_ids_for_different_leads():
    ids = {tools.record_customer_interest("a@example.com", "Ana", f"Interested in D0{i} for two", group_size=2)["inquiry_id"] for i in range(1, 8)}
    assert len(ids) == 7


def test_storage_failure_is_reported_not_hidden(monkeypatch):
    def broken(*_):
        raise PermissionError("disk is read-only")
    monkeypatch.setattr(tools, "_append_jsonl", broken)
    for r in [tools.record_customer_interest("lina@example.com", "Lina", "D03 for two people", group_size=2),
              tools.record_feedback("Is pickup included?")]:
        assert r["status"] == "error" and r["error"] == "storage_failed" and "Do not tell" in r["message"]


def test_feedback_saved_with_kind(isolated_logs, capsys):
    r = tools.record_feedback("Is the Qadisha hike wheelchair accessible?")
    c = tools.record_feedback("The guide was late.", kind="complaint")
    assert r["status"] == c["status"] == "success" and re.fullmatch(r"FB-\d{8}-[0-9A-F]{6}", r["feedback_id"])
    saved = _lines(isolated_logs / "feedback.jsonl")
    assert [(x["kind"], x["question"]) for x in saved] == [
        ("unanswered_question", "Is the Qadisha hike wheelchair accessible?"), ("complaint", "The guide was late.")]
    assert "[PUSH] New complaint" in capsys.readouterr().out


@pytest.mark.parametrize("question,kind", [("", "unanswered_question"), ("hi", "unanswered_question"),
                                           ("Is pickup included?", "rant")])
def test_feedback_validation(isolated_logs, question, kind):
    r = tools.record_feedback(question, kind)
    assert r["status"] == "error" and not (isolated_logs / "feedback.jsonl").exists()


def test_search_results_are_grounded_in_the_catalog():
    r = tools.search_experiences(limit=20)
    assert r["total_matches"] == 20
    for card in r["results"]:
        row = CATALOG.loc[card["id"]]
        assert (card["name"], card["region"], card["price_usd_per_person"], card["season"]) == \
               (row["name"], row["region"], row["price_usd"], row["season"])
        assert card["transport"].startswith("Not documented") and card["availability"].startswith("Inquiry only")


@pytest.mark.parametrize("filters", [
    {"max_price_per_person": 30}, {"region": "Chouf"}, {"category": "nature"}, {"group_size": 14},
    {"month": "July"}, {"month": 1}, {"max_duration_hours": 4}, {"region": "south"},
    {"interests": ["food"], "max_price_per_person": 35, "month": "August", "group_size": 2},
])
def test_search_respects_hard_filters(filters):
    r = tools.search_experiences(limit=20, **filters)
    assert r["status"] == "success" and r["total_matches"] >= 1
    month = tools._parse_month(filters.get("month"))
    for card in r["results"]:
        row = CATALOG.loc[card["id"]]
        if "max_price_per_person" in filters:
            assert row.price_usd <= filters["max_price_per_person"]
        if filters.get("region") == "south":
            assert row.region in {"Sidon", "Tyre", "Jezzine"}
        elif "region" in filters:
            assert row.region == filters["region"]
        if "category" in filters:
            assert row.category.lower() == filters["category"]
        if "group_size" in filters:
            assert row.min_group <= filters["group_size"] <= row.max_group
        if month:
            assert month in tools.season_months(row.season)
        if "max_duration_hours" in filters:
            assert row.duration_hours <= filters["max_duration_hours"]


def test_search_ranks_by_keywords_and_reports_no_match():
    r = tools.search_experiences(query="kayak sunset")
    assert r["results"][0]["id"] == "D20"
    none = tools.search_experiences(query="scuba diving")
    assert none["total_matches"] == 0 and "record_feedback" in none["note"] and none["excluded_by_filter"]["keywords"] == 20
    assert tools.search_experiences(month="Smarch")["error"] == "invalid_month"


def test_season_parsing():
    assert tools.season_months("Winter only (December-March, snow-dependent)") == {12, 1, 2, 3}
    assert tools.season_months("Spring to autumn (April-November)") == set(range(4, 12))
    assert tools.season_months("Year-round; harvest season is September-October") == set(range(1, 13))


def test_compare_normalizes_ids_and_summarizes():
    r = tools.compare_experiences(["d4", "D08", "D08", "D99"], group_size=4)
    assert [c["id"] for c in r["compared"]] == ["D04", "D08"] and r["not_found"] == ["D99"]
    assert r["summary"]["cheapest"] == ["D04"] and r["summary"]["price_difference_per_person_usd"] == 5
    assert [c["group_total_usd"] for c in r["compared"]] == [220, 240]
    assert "transport" in r["unknown_for_all_options"]


def test_compare_flags_group_size_and_needs_two_ids():
    r = tools.compare_experiences(["D03", "D04"], group_size=12)
    assert r["summary"]["not_suitable_for_group_size"] == ["D03"]
    assert tools.compare_experiences(["D01"])["error"] == "need_two_valid_ids"
    assert tools.compare_experiences(["D01", "X1"])["error"] == "need_two_valid_ids"


def test_itinerary_respects_budget_and_group_total():
    r = tools.create_itinerary(days=2, budget_per_person=150, interests=["nature", "food"], month="July", group_size=2)
    assert r["feasible"] and r["estimated_total_per_person_usd"] <= 150
    assert r["estimated_total_per_person_usd"] == sum(a["price_usd_per_person"] for a in _placed(r))
    assert r["estimated_group_total_usd"] == 2 * r["estimated_total_per_person_usd"]


def test_itinerary_revises_requested_plan_over_budget():
    r = tools.create_itinerary(days=2, budget_per_person=100, experience_ids=["D08", "D06", "D03"])
    assert r["feasible"] and r["estimated_total_per_person_usd"] <= 100
    assert any("Removed D06" in x and "budget" in x for x in r["revisions"])
    assert {"D08", "D03"} <= {a["id"] for a in _placed(r)}


def test_itinerary_removes_out_of_season_and_impossible_overnight():
    r = tools.create_itinerary(days=1, experience_ids=["D19", "D06", "D05"], month="July")
    ids = {a["id"] for a in _placed(r)}
    assert "D19" not in ids and "D06" not in ids and "D05" in ids
    assert any("D19" in x and "July" in x for x in r["revisions"])
    assert any("D06" in x and "at least 2 days" in x for x in r["revisions"])


def test_itinerary_overnight_spans_two_days():
    r = tools.create_itinerary(days=2, experience_ids=["D06"])
    slots = [(d["day"], a["slot"]) for d in r["days"] for a in d["activities"] if a["id"] == "D06"]
    assert slots == [(1, "Afternoon, overnight stay"), (2, "Morning, end of overnight stay")]
    assert r["estimated_total_per_person_usd"] == 85


def test_itinerary_infeasible_explains_why():
    r = tools.create_itinerary(days=1, budget_per_person=20)
    assert r["status"] == "success" and r["feasible"] is False and "$25" in r["infeasible_reason"]
    nothing = tools.create_itinerary(days=2, interests=["scuba"])
    assert nothing["feasible"] is False and "no catalog experience" in nothing["infeasible_reason"]


def test_itinerary_total_budget_needs_group_size():
    assert tools.create_itinerary(days=2, total_budget=200)["error"] == "group_size_required"
    r = tools.create_itinerary(days=2, total_budget=200, group_size=4)
    assert r["estimated_total_per_person_usd"] <= 50 and r["estimated_group_total_usd"] <= 200


def test_itinerary_invariants_over_many_constraint_combinations():
    """Property check over 180 combinations: totals add up, budgets hold, slots are never overbooked,
    nothing repeats, and every scheduled experience is in season and accepts the group."""
    combos = itertools.product([1, 2, 3], [None, 40, 90, 200], [None, ["nature"], ["food", "history"]],
                               [None, "January", "July"], [None, 2, 9], [1, 2])
    count = 0
    for days, budget, interests, month, group, per_day in itertools.islice(combos, 0, None, 2):
        r = tools.create_itinerary(days=days, budget_per_person=budget, interests=interests, month=month,
                                 group_size=group, max_activities_per_day=per_day)
        placed = _placed(r)
        ids = [a["id"] for a in placed]
        assert len(ids) == len(set(ids)), r
        assert r["estimated_total_per_person_usd"] == sum(a["price_usd_per_person"] for a in placed)
        if budget is not None:
            assert r["estimated_total_per_person_usd"] <= budget
        for day in r["days"]:
            assert len({a["id"] for a in day["activities"]}) <= per_day
        m = tools._parse_month(month)
        for i in ids:
            row = CATALOG.loc[i]
            assert not m or m in tools.season_months(row.season)
            assert not group or row.min_group <= group <= row.max_group
        assert r["feasible"] == bool(placed)
        count += 1
    assert count >= 100


def test_tool_specs_are_complete_mcp_style():
    assert [s["name"] for s in tools.TOOL_SPECS] == ["record_customer_interest", "record_feedback",
                                                   "search_experiences", "compare_experiences", "create_itinerary",
                                                   "budget_calculator", "season_check"]
    for s in tools.TOOL_SPECS:
        assert {"name", "description", "inputSchema", "outputSchema"} <= set(s)
        assert set(s["inputSchema"].get("required", [])) <= set(s["inputSchema"]["properties"])
        assert "status" in s["outputSchema"]["properties"]
    assert tools.SPEC_BY_NAME["record_customer_interest"]["inputSchema"]["required"] == ["email", "name", "message", "group_size"]
    assert tools.SPEC_BY_NAME["record_feedback"]["inputSchema"]["required"] == ["question"]


def test_dispatcher_coerces_light_type_mistakes():
    r = tools.execute_tool("create_itinerary", '{"days": "2", "group_size": "4", "budget_per_person": "120"}')
    assert r["status"] == "success" and r["estimated_group_total_usd"] == 4 * r["estimated_total_per_person_usd"]
    assert tools.execute_tool("search_experiences", {"category": "nature"})["status"] == "success"


@pytest.mark.parametrize("args,text,missing", [
    ({"email": "a@b.com", "name": "Ana"}, "ana here, a@b.com", []),
    ({"email": "a@b.com", "name": "Ana"}, "my email is A@B.com, I'm ANA", []),
    ({"email": "ana@b.com", "name": "Ana"}, "I'm Ana", ["email"]),
    ({"email": "x@y.com", "name": "Visitor"}, "book D04, x@y.com", ["name"]),
    ({"email": "nope", "name": "Ana"}, "I'm Ana", []),  # malformed emails are left to argument validation
    ({"email": "rami@ex.com", "name": "رامي"}, "انا رامي rami@ex.com", []),
])
def test_lead_details_must_come_from_the_visitor(args, text, missing):
    assert tools.lead_details_not_from_visitor(args, text) == missing


@pytest.mark.parametrize("message,route,loggable", [
    ("Any restaurant suggestions in Byblos?", router.Route(G), False),
    ("I would like D50 please.", router.Route(L), False),
    ("I want to book the Batroun sunset kayak for 2 next weekend. My name is Nour.", router.Route(L), False),
    ("What is the weather usually like in the Bekaa in January?", router.Route(G), False),
    ("Is pickup from my hotel included?", router.Route(G), True),          # router missed it; the wording did not
    ("Book D03 for 2, is it vegan?", router.Route(L), True),
    ("Book D03 for 2 on Sunday", router.Route(L, [F]), True),
    ("Our guide was 40 minutes late and never apologised.", router.Route(F), True),
    ("هل النقل من بيروت مشمول؟", router.Route(F), True),
])
def test_only_team_questions_and_feedback_can_be_logged(message, route, loggable):
    assert tools.is_loggable(message, route) is loggable

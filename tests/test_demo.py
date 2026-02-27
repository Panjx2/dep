from src.demo import evaluate_case, parse_json_maybe, perturbations, validate_schema


def test_parse_json_maybe():
    payload, err = parse_json_maybe('{"a":1}')
    assert err is None
    assert payload == {"a": 1}


def test_validate_schema_ok_and_fail():
    ok = {
        "ticket_id": "t1",
        "customer_name": None,
        "email": None,
        "product": "api",
        "priority": "low",
        "issue_summary": "x",
        "actions_requested": ["a"],
        "needs_human": False,
        "confidence": 0.2,
    }
    assert validate_schema(ok) == []
    bad = {**ok, "product": "wrong"}
    errors = validate_schema(bad)
    assert any("SCHEMA_ENUM_MISMATCH" in e for e in errors)


def test_evaluate_case_actions_f1():
    pred = {
        "product": "api",
        "priority": "high",
        "needs_human": False,
        "customer_name": "Alice",
        "email": "a@example.com",
        "issue_summary": "API failure",
        "actions_requested": ["Reset API key"],
    }
    gold = {**pred, "actions_requested": ["reset api key", "Contact support"]}
    score = evaluate_case(pred, gold)
    assert 0 < score["actions_requested"] < 1


def test_perturbations_present():
    names = set(perturbations().keys())
    required = {
        "typos",
        "unicode_noise",
        "distractor",
        "prompt_injection",
        "contradictory_instruction",
        "field_aliasing",
        "order_shuffle",
        "format_bait",
        "long_context_overload",
    }
    assert required.issubset(names)

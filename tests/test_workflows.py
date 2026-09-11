from voice_detection.workflows import choose_action, enrich_risk


def test_context_raises_risk_for_high_value_flagged_caller() -> None:
    assert enrich_risk(0.5, {"caller_reputation": "flagged", "transaction_amount": 500_000}) == 0.86


def test_prevention_actions_follow_scenario_threshold() -> None:
    assert choose_action(41, "high_value_transfer_approval").kind == "step_up_verification"
    assert choose_action(89, "support_call").kind == "escalate"

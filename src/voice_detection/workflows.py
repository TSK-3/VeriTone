"""Configurable prevention actions and contextual risk enrichment."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

THRESHOLDS = {"support_call": 70, "high_value_transfer_approval": 40, "privileged_access_request": 30}

@dataclass(frozen=True)
class PreventionAction:
    kind: Literal["pass", "warn", "step_up_verification", "escalate", "block"]
    message: str

def enrich_risk(model_score: float, context: dict) -> float:
    """Transparent demo rule layer; replace with calibrated context fusion later."""
    adjustment = 0.20 if context.get("caller_reputation") == "flagged" else 0.06 if context.get("caller_reputation") == "unknown" else 0.0
    amount = float(context.get("transaction_amount", 0) or 0)
    adjustment += 0.16 if amount >= 500_000 else 0.08 if amount >= 100_000 else 0.0
    adjustment += 0.16 if context.get("historical_fraud_indicator") else 0.0
    return round(min(1.0, model_score + adjustment), 4)

def choose_action(risk_0_to_100: int, scenario: str) -> PreventionAction:
    threshold = THRESHOLDS.get(scenario, THRESHOLDS["support_call"])
    if risk_0_to_100 >= 90: return PreventionAction("block", "Freeze the pending action and transfer this call to fraud operations.")
    if risk_0_to_100 >= threshold + 15: return PreventionAction("escalate", "Escalate to a supervisor and require verified call-back before proceeding.")
    if risk_0_to_100 >= threshold: return PreventionAction("step_up_verification", "Require OTP/MFA or a call-back to the registered number.")
    if risk_0_to_100 >= max(20, threshold - 15): return PreventionAction("warn", "Show an integrity warning and continue monitoring.")
    return PreventionAction("pass", "Pass: genuine user provisionally. Tier 2 monitoring remains active.")

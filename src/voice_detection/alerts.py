"""Trigger-word detection and SMS alerting for the live-call prevention workflow.

Two independent alert reasons exist:

* ``threshold``  — the running risk crossed a prevention action threshold
  (``step_up_verification`` / ``escalate`` / ``block``).
* ``trigger``    — fraud trigger words (``transaction``, ``send``, ``money``, ...)
  were spoken while the AI-voice risk is elevated.

SMS delivery uses the Twilio REST API when credentials are configured. Without
credentials the message is logged to the console and the live dashboard as
``logged_console`` so the demo pipeline is always visible end-to-end.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

TRIGGER_WORDS: tuple[str, ...] = (
    "transaction", "send", "money", "transfer", "payment", "refund", "otp", "pin",
    "password", "verify", "verification", "account", "bank", "card", "credit",
    "debit", "upi", "urgent", "approve", "approval", "wire", "cash", "wallet",
    "invoice", "pay",
)

THRESHOLD_ACTIONS = {"step_up_verification", "escalate", "block"}


def detect_trigger_words(text: str) -> list[str]:
    """Return trigger words present in ``text`` (case-insensitive, word boundaries)."""
    lowered = f" {text.lower()} "
    found = []
    for word in TRIGGER_WORDS:
        for tail in (" ", ",", ".", "?", "!"):
            if f" {word}{tail}" in lowered:
                found.append(word)
                break
    return found


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_sms_body(risk: int, words: list[str], action: str) -> str:
    """Compose the prevention SMS shown to the customer."""
    words_part = f" Trigger words heard: {', '.join(words[:5])}." if words else ""
    action_part = {
        "block": "The action has been frozen.",
        "escalate": "The call is being escalated to fraud operations.",
        "step_up_verification": "Secondary verification is required before any action.",
        "warn": "Stay cautious and keep monitoring the call.",
    }.get(action, "Treat the caller's instructions with suspicion.")
    return (
        f"VeriTone ALERT - risk {risk}/100.\n"
        f"Possible AI-generated voice on your live call.{words_part}\n"
        f"{action_part}\n"
        "Do NOT approve payments or share OTPs. Hang up and call back on the official number."
    )


@dataclass(frozen=True)
class AlertRecord:
    session_id: str
    timestamp: str
    kind: str          # "threshold" | "trigger" | "test"
    risk: int
    action: str
    words: list[str] = field(default_factory=list)
    sms_status: str = "logged_console"
    sms_to: str | None = None
    sms_error: str | None = None
    body: str = ""

    def as_dict(self) -> dict:
        return {
            "session_id": self.session_id, "timestamp": self.timestamp, "kind": self.kind,
            "risk": self.risk, "action": self.action, "words": self.words,
            "sms_status": self.sms_status, "sms_to": self.sms_to,
            "sms_error": self.sms_error, "body": self.body,
        }


class SmsSender:
    """Twilio SMS via REST (``httpx``); degrades to console logging without creds."""

    def __init__(self, account_sid: str | None = None, auth_token: str | None = None,
                 from_number: str | None = None, default_to: str | None = None) -> None:
        self.account_sid = account_sid or os.getenv("TWILIO_ACCOUNT_SID", "")
        self.auth_token = auth_token or os.getenv("TWILIO_AUTH_TOKEN", "")
        self.from_number = from_number or os.getenv("TWILIO_FROM_NUMBER", "")
        self.default_to = default_to or os.getenv("ALERT_TO_NUMBER", "")
        self.configured = bool(self.account_sid and self.auth_token and self.from_number)

    def send(self, to: str | None, body: str) -> tuple[str, str | None]:
        """Return ``(status, error)``; status is ``sent`` | ``logged_console`` | ``error``."""
        recipient = to or self.default_to
        if not (self.configured and recipient):
            print(f"[VeriTone SMS · console mode] to={recipient or '<ALERT_TO_NUMBER unset>'}: {body}")
            return "logged_console", None
        try:
            import httpx
        except ImportError:  # pragma: no cover - httpx ships in the dev extra
            print(f"[VeriTone SMS · console mode] install httpx to enable real SMS: {body}")
            return "logged_console", None
        try:
            response = httpx.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json",
                data={"From": self.from_number, "To": recipient, "Body": body},
                auth=(self.account_sid, self.auth_token),
                timeout=8.0,
            )
        except Exception as exc:  # a network hiccup must never break the live call
            print(f"[VeriTone SMS · error] {exc}")
            return "error", str(exc)
        if 200 <= response.status_code < 300:
            return "sent", None
        return "error", f"Twilio {response.status_code}: {response.text[:180]}"


class AlertEngine:
    """Decides *when* to alert, applies cooldowns, sends the SMS and keeps the log."""

    def __init__(self, sender: SmsSender | None = None, cooldown_s: float | None = None,
                 trigger_risk_floor: int | None = None, sink: list | None = None) -> None:
        import time as _time
        self._time = _time
        self.sender = sender or SmsSender()
        self.cooldown_s = float(cooldown_s if cooldown_s is not None else os.getenv("ALERT_COOLDOWN_S", "60"))
        self.trigger_risk_floor = int(trigger_risk_floor if trigger_risk_floor is not None
                                      else os.getenv("ALERT_TRIGGER_RISK", "40"))
        self.sink = sink if sink is not None else []
        self._last_sent: dict[tuple[str, str], float] = {}

    def _cooldown_ok(self, session_id: str, kind: str) -> bool:
        last = self._last_sent.get((session_id, kind))
        return last is None or (self._time.time() - last) >= self.cooldown_s

    def _dispatch(self, session_id: str, kind: str, risk: int, action: str,
                  words: list[str], to: str | None = None) -> AlertRecord:
        body = build_sms_body(risk, words, action)
        status, error = self.sender.send(to, body)
        record = AlertRecord(
            session_id=session_id, timestamp=now_iso(), kind=kind, risk=risk,
            action=action, words=words, sms_status=status,
            sms_to=to or self.sender.default_to or None, sms_error=error, body=body,
        )
        self.sink.append(record.as_dict())
        self._last_sent[(session_id, kind)] = self._time.time()
        print(f"[VeriTone ALERT] {kind} risk={risk} words={words} sms={status} call={session_id}")
        return record

    def check_after_record(self, session_id: str, record: dict) -> AlertRecord | None:
        """Fire a threshold alert when a scored segment crosses a prevention action."""
        action = record.get("action", "pass")
        if action not in THRESHOLD_ACTIONS or not self._cooldown_ok(session_id, "threshold"):
            return None
        return self._dispatch(session_id, "threshold", int(record.get("risk_score", 0)), action, [])

    def check_trigger_words(self, session_id: str, words: list[str], risk: int,
                            action: str = "warn") -> AlertRecord | None:
        """Fire a trigger-word alert when fraud terms are heard at elevated risk."""
        if not words or risk < self.trigger_risk_floor or not self._cooldown_ok(session_id, "trigger"):
            return None
        return self._dispatch(session_id, "trigger", risk, action, words)

    def send_test(self, to: str | None = None, body: str | None = None) -> AlertRecord:
        """Manual pipeline check used before going on stage."""
        body = body or "VeriTone demo: the SMS prevention pipeline is active. (test message)"
        status, error = self.sender.send(to, body)
        record = AlertRecord(session_id="pipeline-test", timestamp=now_iso(), kind="test",
                             risk=0, action="pass", sms_status=status,
                             sms_to=to or self.sender.default_to or None, sms_error=error, body=body)
        self.sink.append(record.as_dict())
        return record


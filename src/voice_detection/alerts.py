"""Trigger-word detection and Twilio alerting (SMS + outbound calls) for the
live-call prevention workflow.

Two independent alert reasons exist:

* ``threshold``  — the running risk crossed a prevention action threshold
  (``step_up_verification`` / ``escalate`` / ``block``).
* ``trigger``    — fraud trigger words (``transaction``, ``send``, ``money``, ...)
  were spoken while the AI-voice risk is elevated.

All Twilio REST access lives in ``TwilioClient`` (SMS, outbound calls, call
status). Without credentials SMS/calls degrade to console logging as
``logged_console`` so the pipeline is always visible end-to-end.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

import httpx

from .models import now_iso

TRIGGER_WORDS: tuple[str, ...] = (
    "transaction", "send", "money", "transfer", "payment", "refund", "otp", "pin",
    "password", "verify", "verification", "account", "bank", "card", "credit",
    "debit", "upi", "urgent", "approve", "approval", "wire", "cash", "wallet",
    "invoice", "pay",
)

_TRIGGER_RE = re.compile(r"\b(" + "|".join(TRIGGER_WORDS) + r")\b", re.IGNORECASE)

THRESHOLD_ACTIONS = {"step_up_verification", "escalate", "block"}


def detect_trigger_words(text: str) -> list[str]:
    """Return trigger words present in ``text`` (case-insensitive, word boundaries)."""
    return [word.lower() for word in _TRIGGER_RE.findall(text)]


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


class TwilioClient:
    """All Twilio REST access: SMS alerts, outbound calls, call status.

    Degrades to console logging when credentials are absent.
    """

    _API = "https://api.twilio.com/2010-04-01/Accounts/{sid}"

    def __init__(self, account_sid: str | None = None, auth_token: str | None = None,
                 from_number: str | None = None, default_to: str | None = None) -> None:
        # Explicit "" disables the channel (tests must never send real SMS);
        # only None defers to the environment.
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID", "") if account_sid is None else account_sid
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN", "") if auth_token is None else auth_token
        self.from_number = os.getenv("TWILIO_FROM_NUMBER", "") if from_number is None else from_number
        self.default_to = os.getenv("ALERT_TO_NUMBER", "") if default_to is None else default_to
        self.content_sid = os.getenv("TWILIO_CONTENT_SID", "")
        self._template_sid: str | None = None
        self.configured = bool(self.account_sid and self.auth_token and self.from_number)

    def _request(self, method: str, path: str, data: dict | None = None, timeout: float = 10.0) -> httpx.Response:
        return httpx.request(method, self._API.format(sid=self.account_sid) + path,
                             data=data, auth=(self.account_sid, self.auth_token), timeout=timeout)

    def send(self, to: str | None, body: str) -> tuple[str, str | None]:
        """Return ``(status, error)``; status is ``sent`` | ``logged_console`` | ``error``."""
        recipient = to or self.default_to
        if not (self.configured and recipient):
            print(f"[VeriTone SMS · console mode] to={recipient or '<ALERT_TO_NUMBER unset>'}: {body}")
            return "logged_console", None
        try:
            response = self._request("POST", "/Messages.json",
                                     {"From": self.from_number, "To": recipient, "Body": body}, timeout=8.0)
        except Exception as exc:  # a network hiccup must never break the live call
            print(f"[VeriTone SMS · error] {exc}")
            return "error", str(exc)
        if 200 <= response.status_code < 300:
            return "sent", None
        error_text = f"Twilio {response.status_code}: {response.text[:180]}"
        # Trial accounts (esp. India) only allow predefined templates (code 572006).
        if "572006" in error_text or "template" in error_text.lower():
            template_status, template_error = self._send_with_template(recipient)
            if template_status == "sent":
                return "sent", None
            return template_status, template_error or error_text
        return "error", error_text

    def _send_with_template(self, recipient: str) -> tuple[str, str | None]:
        """Trial-policy fallback: send via one of Twilio's predefined content templates."""
        candidates = [sid for sid in (self.content_sid, self._template_sid) if sid]
        if not candidates:
            try:
                listing = httpx.get("https://content.twilio.com/v1/Content",
                                    auth=(self.account_sid, self.auth_token), timeout=10.0)
            except Exception as exc:
                return "error", f"template list failed: {exc}"
            if listing.status_code == 200:
                candidates = [c.get("sid") for c in listing.json().get("contents", []) if c.get("sid")]
        for sid in candidates[:8]:
            for variables in ('{}', '{"1":"VeriTone"}'):
                response = self._request("POST", "/Messages.json",
                                         {"From": self.from_number, "To": recipient,
                                          "ContentSid": sid, "ContentVariables": variables}, timeout=8.0)
                if 200 <= response.status_code < 300:
                    self._template_sid = sid
                    return "sent", None
        return "error", (
            "trial restriction: this account can only send predefined SMS templates. "
            "Fix (30s): Twilio Console → Messaging → Content Tools → copy the Content SID of a "
            "predefined template (starts with HX) → set TWILIO_CONTENT_SID, restart. "
            "Upgrading the account removes this entirely."
        )

    def place_call(self, to: str, twiml_url: str) -> dict:
        """Place an outbound call bridged to ``twiml_url``; returns ``{"call_sid", "twilio_status"}``."""
        response = self._request("POST", "/Calls.json",
                                 {"To": to, "From": self.from_number, "Url": twiml_url}, timeout=15.0)
        if not 200 <= response.status_code < 300:
            raise RuntimeError(f"Twilio call failed: {response.text[:300]}")
        payload = response.json()
        return {"call_sid": payload.get("sid"), "twilio_status": payload.get("status")}

    def call_status(self, call_sid: str) -> dict:
        """Return ``{"status", "duration_s"}`` for a placed call."""
        response = self._request("GET", f"/Calls/{call_sid}.json", timeout=15.0)
        if not 200 <= response.status_code < 300:
            raise RuntimeError(f"Twilio status failed: {response.text[:200]}")
        data = response.json()
        return {"status": data.get("status"), "duration_s": data.get("duration")}


class AlertEngine:
    """Decides *when* to alert, applies cooldowns, sends the SMS and keeps the log."""

    def __init__(self, sender: TwilioClient | None = None, cooldown_s: float | None = None,
                 trigger_risk_floor: int | None = None, sink: list | None = None) -> None:
        self.sender = sender or TwilioClient()
        self.cooldown_s = float(cooldown_s if cooldown_s is not None else os.getenv("ALERT_COOLDOWN_S", "60"))
        self.trigger_risk_floor = int(trigger_risk_floor if trigger_risk_floor is not None
                                      else os.getenv("ALERT_TRIGGER_RISK", "40"))
        self.sink = sink if sink is not None else []
        self._last_sent: dict[tuple[str, str], float] = {}

    def _cooldown_ok(self, session_id: str, kind: str) -> bool:
        last = self._last_sent.get((session_id, kind))
        return last is None or (time.time() - last) >= self.cooldown_s

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
        self._last_sent[(session_id, kind)] = time.time()
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


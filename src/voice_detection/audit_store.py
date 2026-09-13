"""Durable, encrypted, feature-only audit store (SQLite + Fernet).

This is the durable tier behind the in-memory ``SessionStore``/``LiveCallRegistry``:
every scored segment's derived record is persisted so audits survive restarts.
The privacy contract is enforced at the storage boundary:

* only *derived* records are accepted (scores, latencies, feature summaries) —
  there is no field audio and no embedding in any record by construction;
* every row is encrypted at rest with Fernet (AES + HMAC) so a stolen
  ``veritone-audit.db`` leaks nothing without the key;
* the key comes from ``AUDIT_STORE_KEY`` (base64, ``Fernet.generate_key()``) or is
  auto-generated next to the database (``<path>.key``) and reused afterwards;
* right-to-erasure is a first-class operation (``delete_session_records`` /
  ``delete_speaker_reference``).

The same store keeps **consented** speaker references (see ``speaker_refs.py``):
a reference embedding is only written when the customer's explicit consent flag is
``True``, and their segments are checked against it live.
"""
from __future__ import annotations

import base64
import json
import os
import sqlite3
import threading
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'segment',
    risk INTEGER NOT NULL DEFAULT 0,
    payload BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_records (session_id);
CREATE TABLE IF NOT EXISTS speaker_refs (
    speaker_id TEXT PRIMARY KEY,
    consent INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    payload BLOB NOT NULL
);
"""


def _load_key(path: Path) -> bytes:
    """Fernet key (base64-encoded) from AUDIT_STORE_KEY, else ``<db>.key``, else generate."""
    configured = os.getenv("AUDIT_STORE_KEY")
    if configured:
        return configured.strip().encode()
    key_file = path.with_suffix(path.suffix + ".key")
    if key_file.is_file():
        return key_file.read_text().strip().encode()
    key = Fernet.generate_key()  # already base64-encoded, exactly what Fernet wants
    key_file.parent.mkdir(parents=True, exist_ok=True)
    key_file.write_text(key.decode())
    try:  # POSIX-only; best effort so the sidecar is not world-readable
        os.chmod(key_file, 0o600)
    except OSError:
        pass
    return key


class AuditStore:
    """Encrypted persistence for derived records and consented speaker references."""

    def __init__(self, path: str | None = None) -> None:
        configured = path or os.getenv("AUDIT_STORE_PATH") or "audit/veritone-audit.db"
        self.enabled = (os.getenv("AUDIT_STORE") or "on").lower() != "off"
        self.path = Path(configured)
        self._lock = threading.Lock()
        self._connection: sqlite3.Connection | None = None
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fernet = Fernet(_load_key(self.path))
            self._connection = sqlite3.connect(self.path, check_same_thread=False)
            self._connection.executescript(_SCHEMA)
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.commit()

    def _encrypt(self, record: dict) -> bytes:
        return self._fernet.encrypt(json.dumps(record, separators=(",", ":")).encode())

    def _decrypt(self, payload: bytes) -> dict:
        try:
            return json.loads(self._fernet.decrypt(payload))
        except (InvalidToken, json.JSONDecodeError) as exc:
            raise RuntimeError("audit store row could not be decrypted (wrong AUDIT_STORE_KEY?)") from exc


    # -- derived records -------------------------------------------------------

    def put_record(self, session_id: str, record: dict, kind: str = "segment") -> int | None:
        """Persist one derived record (encrypted). Returns the row id, or ``None``
        when the store is disabled. Failures never break live scoring."""
        if not self.enabled or self._connection is None:
            return None
        try:
            payload = self._encrypt(record)
            timestamp = str(record.get("timestamp") or record.get("segment", {}).get("timestamp", ""))
            risk = int(record.get("risk_score", 0) or 0)
            with self._lock, self._connection as db:
                cursor = db.execute(
                    "INSERT INTO audit_records (session_id, timestamp, kind, risk, payload) VALUES (?, ?, ?, ?, ?)",
                    (session_id, timestamp, kind, risk, payload),
                )
                return int(cursor.lastrowid or 0)
        except Exception as exc:  # durability must never take the call down
            print(f"[VeriTone audit] could not persist record: {exc}")
            return None

    def list_records(self, session_id: str | None = None, limit: int = 100) -> list[dict]:
        if not self.enabled or self._connection is None:
            return []
        query = "SELECT id, session_id, kind, payload FROM audit_records"
        params: tuple = ()
        if session_id:
            query += " WHERE session_id = ?"
            params = (session_id,)
        query += " ORDER BY id DESC LIMIT ?"
        with self._lock:
            rows = self._connection.execute(query, (*params, max(1, limit))).fetchall()
        return [{"id": row_id, "session_id": sid, "kind": kind, **self._decrypt(payload)}
                for row_id, sid, kind, payload in rows]

    def delete_session_records(self, session_id: str) -> int:
        """Right-to-erasure for one session's stored records."""
        if not self.enabled or self._connection is None:
            return 0
        with self._lock, self._connection as db:
            cursor = db.execute("DELETE FROM audit_records WHERE session_id = ?", (session_id,))
            return int(cursor.rowcount or 0)

    # -- consented speaker references -------------------------------------------

    def put_speaker_reference(self, speaker_id: str, embedding: list[float], consent: bool, created_at: str) -> bool:
        """Write a reference embedding **only** with explicit consent."""
        if not consent:
            raise ValueError("speaker reference requires explicit consent=true")
        if not self.enabled or self._connection is None:
            return False
        payload = self._encrypt({"embedding": embedding})
        with self._lock, self._connection as db:
            db.execute(
                "INSERT INTO speaker_refs (speaker_id, consent, created_at, payload) VALUES (?, ?, ?, ?) "
                "ON CONFLICT (speaker_id) DO UPDATE SET payload = excluded.payload, created_at = excluded.created_at",
                (speaker_id, int(consent), created_at, payload),
            )
        return True

    def get_speaker_reference(self, speaker_id: str) -> dict | None:
        if not self.enabled or self._connection is None:
            return None
        with self._lock:
            row = self._connection.execute(
                "SELECT consent, created_at, payload FROM speaker_refs WHERE speaker_id = ?", (speaker_id,)
            ).fetchone()
        if row is None:
            return None
        consent, created_at, payload = row
        return {"speaker_id": speaker_id, "consent": bool(consent), "created_at": created_at, **self._decrypt(payload)}

    def delete_speaker_reference(self, speaker_id: str) -> int:
        if not self.enabled or self._connection is None:
            return 0
        with self._lock, self._connection as db:
            cursor = db.execute("DELETE FROM speaker_refs WHERE speaker_id = ?", (speaker_id,))
            return int(cursor.rowcount or 0)


STORE = AuditStore()  # process-wide singleton, one per demo console

"""Job persistence. SQLite with a JSON document per job — jobs are read whole,
written whole, and there are never many of them in flight at once.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import DB_PATH, WORK_DIR

_lock = threading.RLock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id         TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status     TEXT NOT NULL,
    data       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_updated ON jobs(updated_at DESC);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


_conn = _connect()
with _lock:
    _conn.executescript(SCHEMA)
    _conn.commit()


def workdir(job_id: str) -> Path:
    path = WORK_DIR / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def create(data: dict[str, Any]) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    stamp = now()
    data = {**data, "id": job_id, "created_at": stamp, "updated_at": stamp}
    data.setdefault("status", "created")
    data.setdefault("log", [])
    data.setdefault("assets", {})
    data.setdefault("renders", {})
    with _lock:
        _conn.execute(
            "INSERT INTO jobs (id, created_at, updated_at, status, data) VALUES (?,?,?,?,?)",
            (job_id, stamp, stamp, data["status"], json.dumps(data)),
        )
        _conn.commit()
    workdir(job_id)
    return data


def get(job_id: str) -> dict[str, Any] | None:
    with _lock:
        row = _conn.execute("SELECT data FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return json.loads(row["data"]) if row else None


def save(data: dict[str, Any]) -> dict[str, Any]:
    data["updated_at"] = now()
    with _lock:
        _conn.execute(
            "UPDATE jobs SET updated_at = ?, status = ?, data = ? WHERE id = ?",
            (data["updated_at"], data.get("status", "created"), json.dumps(data), data["id"]),
        )
        _conn.commit()
    return data


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    with _lock:
        rows = _conn.execute(
            "SELECT data FROM jobs ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [json.loads(r["data"]) for r in rows]


def delete(job_id: str) -> bool:
    with _lock:
        cur = _conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        _conn.commit()
    return cur.rowcount > 0


def log(job: dict[str, Any], message: str, level: str = "info") -> None:
    job.setdefault("log", []).append({"at": now(), "level": level, "message": message})
    job["log"] = job["log"][-300:]


class JobLocks:
    """One lock per job so a resume can't run twice concurrently."""

    def __init__(self) -> None:
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def acquire(self, job_id: str) -> bool:
        with self._guard:
            lock = self._locks.setdefault(job_id, threading.Lock())
        return lock.acquire(blocking=False)

    def release(self, job_id: str) -> None:
        with self._guard:
            lock = self._locks.get(job_id)
        if lock and lock.locked():
            lock.release()

    def held(self, job_id: str) -> bool:
        lock = self._locks.get(job_id)
        return bool(lock and lock.locked())


locks = JobLocks()

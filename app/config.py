"""Settings and the credential store.

Keys can come from two places, in priority order:

  1. the runtime keystore  (data/credentials.json, written via the API or the web console)
  2. environment / .env

The keystore is chmod 600 and never returned in plaintext over the API.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SONGLOC_DATA_DIR", ROOT / "data"))
WORK_DIR = DATA_DIR / "work"
KEYSTORE = DATA_DIR / "credentials.json"
DB_PATH = DATA_DIR / "songloc.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)


def _load_dotenv() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()


class Keystore:
    """Runtime credential storage. Reads are masked unless explicitly unmasked."""

    def __init__(self, path: Path = KEYSTORE) -> None:
        self.path = path
        self._cache: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._cache = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                self._cache = {}

    def _flush(self) -> None:
        self.path.write_text(json.dumps(self._cache, indent=2))
        os.chmod(self.path, 0o600)

    def get(self, name: str) -> str | None:
        """Runtime value wins over the environment."""
        return self._cache.get(name) or os.environ.get(name) or None

    def set_many(self, values: dict[str, str]) -> list[str]:
        written = []
        for name, value in values.items():
            value = (value or "").strip()
            if not value:
                self._cache.pop(name, None)
                continue
            self._cache[name] = value
            written.append(name)
        self._flush()
        return written

    def source(self, name: str) -> str | None:
        if name in self._cache:
            return "keystore"
        if os.environ.get(name):
            return "environment"
        return None

    @staticmethod
    def mask(value: str) -> str:
        if len(value) <= 8:
            return "•" * len(value)
        return f"{value[:4]}{'•' * 8}{value[-4:]}"

    def status(self, name: str) -> dict[str, Any]:
        value = self.get(name)
        return {
            "name": name,
            "configured": bool(value),
            "source": self.source(name),
            "preview": self.mask(value) if value else None,
        }


keystore = Keystore()


class Settings:
    """Non-secret runtime settings."""

    host: str = os.environ.get("SONGLOC_HOST", "127.0.0.1")
    port: int = int(os.environ.get("SONGLOC_PORT", "8000"))
    max_adaptation_rounds: int = int(os.environ.get("SONGLOC_MAX_ADAPT_ROUNDS", "3"))
    # A line is accepted without human flagging above this fit score.
    fit_threshold: float = float(os.environ.get("SONGLOC_FIT_THRESHOLD", "0.75"))
    # Human review gate. Turning this off is how you go from careful to fast --
    # and it is the single biggest lever on output quality.
    require_review: bool = os.environ.get("SONGLOC_REQUIRE_REVIEW", "true").lower() != "false"
    http_timeout: float = float(os.environ.get("SONGLOC_HTTP_TIMEOUT", "600"))


settings = Settings()

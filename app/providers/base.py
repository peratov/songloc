"""The provider contract.

Every external service plugs in here. To add one you subclass the stage's base
class, declare which credentials it needs, and register it. The console picks it
up automatically — nothing else in the pipeline changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Type

import httpx

from ..config import keystore, settings


class Stage(str, Enum):
    SEPARATE = "separate"       # split the master into vocal + instrumental
    TRANSCRIBE = "transcribe"   # lyrics with word-level timestamps
    ADAPT = "adapt"             # singable translation
    VOICE = "voice"             # produce the new lead vocal
    MIX = "mix"                 # place it back on the bed


@dataclass
class Credential:
    name: str                   # environment variable / keystore key
    label: str
    required: bool = True
    help: str = ""


@dataclass
class Option:
    name: str
    label: str
    default: Any = None
    choices: list[str] | None = None
    help: str = ""


class Provider:
    """Base for all providers."""

    id: str = ""
    stage: Stage
    label: str = ""
    kind: str = "api"           # api | local | manual
    docs: str = ""
    notes: str = ""
    credentials: list[Credential] = []
    options: list[Option] = []

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    # -- credentials ------------------------------------------------------

    def key(self, name: str) -> str | None:
        return keystore.get(name)

    def require(self, name: str) -> str:
        value = self.key(name)
        if not value:
            raise MissingCredential(f"{self.id}: missing credential {name}")
        return value

    @classmethod
    def is_configured(cls) -> bool:
        return all(
            (not c.required) or bool(keystore.get(c.name)) for c in cls.credentials
        )

    @classmethod
    def describe(cls) -> dict[str, Any]:
        return {
            "id": cls.id,
            "stage": cls.stage.value,
            "label": cls.label or cls.id,
            "kind": cls.kind,
            "docs": cls.docs,
            "notes": cls.notes,
            "configured": cls.is_configured(),
            "credentials": [
                {**keystore.status(c.name), "label": c.label, "required": c.required, "help": c.help}
                for c in cls.credentials
            ],
            "options": [
                {"name": o.name, "label": o.label, "default": o.default,
                 "choices": o.choices, "help": o.help}
                for o in cls.options
            ],
        }

    # -- http -------------------------------------------------------------

    def client(self, **kwargs) -> httpx.Client:
        kwargs.setdefault("timeout", settings.http_timeout)
        return httpx.Client(**kwargs)

    def opt(self, name: str, default: Any = None) -> Any:
        if name in self.config:
            return self.config[name]
        for o in self.options:
            if o.name == name:
                return o.default
        return default


class MissingCredential(RuntimeError):
    pass


class ProviderNotImplemented(RuntimeError):
    """Raised by stub adapters — the shape is right, the endpoint needs your account's spec."""


# -- stage interfaces ------------------------------------------------------

class SeparationProvider(Provider):
    stage = Stage.SEPARATE

    def separate(self, source: Path, workdir: Path) -> dict[str, Path]:
        """Return {'vocal': Path, 'instrumental': Path}."""
        raise NotImplementedError


class TranscriptionProvider(Provider):
    stage = Stage.TRANSCRIBE

    def transcribe(self, vocal: Path, language: str | None, workdir: Path) -> dict[str, Any]:
        """Return {'language': str, 'words': [{'text','start','end'}, ...]}."""
        raise NotImplementedError


class AdaptationProvider(Provider):
    stage = Stage.ADAPT

    def adapt(
        self,
        templates: list[Any],
        source_lang: str,
        target_lang: str,
        instructions: str,
        previous: list[str] | None = None,
        feedback: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Return one adapted line per template, in order."""
        raise NotImplementedError


class VoiceProvider(Provider):
    stage = Stage.VOICE

    def render(
        self,
        lines: list[str],
        notes: list[Any],
        target_lang: str,
        workdir: Path,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Return {'vocal': Path | None, 'artifacts': [Path], 'manual_step': str | None}."""
        raise NotImplementedError


class MixProvider(Provider):
    stage = Stage.MIX

    def mix(self, instrumental: Path, vocal: Path, workdir: Path, options: dict) -> Path:
        raise NotImplementedError


# -- registry --------------------------------------------------------------

REGISTRY: dict[Stage, dict[str, Type[Provider]]] = {s: {} for s in Stage}


def register(cls: Type[Provider]) -> Type[Provider]:
    if not cls.id:
        raise ValueError(f"{cls.__name__} needs an id")
    REGISTRY[cls.stage][cls.id] = cls
    return cls


def get(stage: Stage, provider_id: str, config: dict | None = None) -> Provider:
    try:
        cls = REGISTRY[stage][provider_id]
    except KeyError:
        known = ", ".join(sorted(REGISTRY[stage])) or "none"
        raise KeyError(f"no {stage.value} provider '{provider_id}'. Available: {known}")
    return cls(config)


def catalogue() -> dict[str, list[dict[str, Any]]]:
    return {
        stage.value: [cls.describe() for cls in providers.values()]
        for stage, providers in REGISTRY.items()
    }

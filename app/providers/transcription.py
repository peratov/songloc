"""Lyrics with word-level timestamps.

Word timings are not optional here — they are what the prosody engine uses to
derive line breaks, syllable slots and stress positions. A provider that returns
plain text without timings is not usable for this pipeline.

For your own recordings you usually already have the correct lyrics, so the best
option is `known-lyrics`: supply the text and let a forced aligner place it. That
removes transcription error from the pipeline entirely.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..audio import duration_seconds
from ..prosody import count_syllables, words_of
from .base import (
    Credential, Option, ProviderNotImplemented, TranscriptionProvider, register,
)


@register
class WhisperXTranscription(TranscriptionProvider):
    id = "whisperx"
    label = "WhisperX (local)"
    kind = "local"
    docs = "https://github.com/m-bain/whisperX"
    notes = "Whisper large-v3 plus forced alignment for word timings. Free, needs a GPU to be quick."
    options = [
        Option("model", "Model", default="large-v3",
               choices=["medium", "large-v2", "large-v3"]),
        Option("device", "Device", default="cpu", choices=["cpu", "cuda"]),
        Option("compute_type", "Compute type", default="int8",
               choices=["int8", "float16", "float32"]),
    ]

    def transcribe(self, vocal: Path, language: str | None, workdir: Path) -> dict[str, Any]:
        if shutil.which("whisperx") is None:
            raise ProviderNotImplemented(
                "whisperx is not installed. `pip install whisperx`, or use "
                "known-lyrics / a hosted provider for this stage."
            )
        out = workdir / "whisperx"
        out.mkdir(exist_ok=True)
        cmd = [
            "whisperx", str(vocal), "--model", self.opt("model"),
            "--device", self.opt("device"), "--compute_type", self.opt("compute_type"),
            "--output_format", "json", "--output_dir", str(out),
        ]
        if language:
            cmd += ["--language", language]
        subprocess.run(cmd, check=True, capture_output=True)
        payload = json.loads(next(out.glob("*.json")).read_text())
        words = [
            {"text": w["word"].strip(), "start": float(w["start"]), "end": float(w["end"])}
            for seg in payload.get("segments", [])
            for w in seg.get("words", [])
            if w.get("start") is not None and w.get("end") is not None
        ]
        return {"language": payload.get("language", language or "en"), "words": words}


@register
class KnownLyricsAlignment(TranscriptionProvider):
    id = "known-lyrics"
    label = "My lyrics, aligned"
    kind = "local"
    notes = (
        "You paste the real lyrics; timings are estimated from the vocal's length "
        "and syllable weighting. Zero transcription error. Swap in a forced aligner "
        "(WhisperX --align_model, or MFA) for tighter timings."
    )
    options = [Option("lyrics", "Lyrics", default="", help="One line per sung line")]

    def transcribe(self, vocal: Path, language: str | None, workdir: Path) -> dict[str, Any]:
        lyrics = (self.opt("lyrics") or "").strip()
        if not lyrics:
            raise ProviderNotImplemented("Paste the lyrics into this provider's 'lyrics' option.")
        lang = language or "en"
        total = duration_seconds(vocal)
        lines = [ln.strip() for ln in lyrics.splitlines() if ln.strip()]

        # Distribute time across lines in proportion to syllable count, with a
        # short breath between lines so the line-splitter reconstructs them.
        weights = [max(1, count_syllables(ln, lang)) for ln in lines]
        breath = 0.35
        singable = max(0.1, total - breath * max(0, len(lines) - 1))
        unit = singable / sum(weights)

        words: list[dict[str, Any]] = []
        cursor = 0.0
        for line, weight in zip(lines, weights):
            span = weight * unit
            tokens = words_of(line)
            if not tokens:
                cursor += span + breath
                continue
            token_weights = [max(1, count_syllables(t, lang)) for t in tokens]
            token_unit = span / sum(token_weights)
            t = cursor
            for token, tw in zip(tokens, token_weights):
                dur = tw * token_unit
                words.append({"text": token, "start": round(t, 3), "end": round(t + dur, 3)})
                t += dur
            cursor += span + breath
        return {"language": lang, "words": words}


@register
class ElevenLabsTranscription(TranscriptionProvider):
    id = "elevenlabs-scribe"
    label = "ElevenLabs Scribe"
    docs = "https://elevenlabs.io/docs/api-reference/speech-to-text"
    notes = "Strong word timings. Tuned for speech, so isolate the vocal first."
    credentials = [Credential("ELEVENLABS_API_KEY", "ElevenLabs API key")]
    options = [Option("model_id", "Model", default="scribe_v1")]

    def transcribe(self, vocal: Path, language: str | None, workdir: Path) -> dict[str, Any]:
        key = self.require("ELEVENLABS_API_KEY")
        data = {"model_id": self.opt("model_id"), "timestamps_granularity": "word"}
        if language:
            data["language_code"] = language
        with self.client(headers={"xi-api-key": key}) as c:
            with open(vocal, "rb") as fh:
                r = c.post(
                    "https://api.elevenlabs.io/v1/speech-to-text",
                    data=data, files={"file": (vocal.name, fh, "audio/wav")},
                )
            r.raise_for_status()
            payload = r.json()
        words = [
            {"text": w["text"].strip(), "start": float(w["start"]), "end": float(w["end"])}
            for w in payload.get("words", [])
            if w.get("type", "word") == "word" and w.get("text", "").strip()
        ]
        return {"language": payload.get("language_code", language or "en"), "words": words}


@register
class MusicAiTranscription(TranscriptionProvider):
    id = "music-ai"
    label = "Music.ai lyrics"
    docs = "https://music.ai/docs"
    notes = "Their lyrics module is trained on singing rather than speech — the best fit for sung audio."
    credentials = [Credential("MUSIC_AI_API_KEY", "Music.ai API key")]
    options = [Option("workflow", "Workflow ID", default="",
                      help="A workflow whose output is aligned lyrics")]

    def transcribe(self, vocal: Path, language: str | None, workdir: Path) -> dict[str, Any]:
        raise ProviderNotImplemented(
            "Point the 'workflow' option at your Music.ai lyrics-alignment workflow "
            "and map its output to {'words': [{'text','start','end'}]}. The job "
            "upload/create/poll flow is already implemented in separation.py:MusicAiSeparation."
        )


@register
class MockTranscription(TranscriptionProvider):
    id = "mock"
    label = "Mock (fixed lyric)"
    kind = "local"
    notes = "Four evenly spaced lines. For testing the pipeline without a model."

    LINES = [
        "I was counting all the hours till the morning",
        "You were somewhere in the shadow of the light",
        "And the quiet in the room became a warning",
        "So I carried what was left of it tonight",
    ]

    def transcribe(self, vocal: Path, language: str | None, workdir: Path) -> dict[str, Any]:
        total = duration_seconds(vocal)
        per_line = total / len(self.LINES)
        words: list[dict[str, Any]] = []
        for i, line in enumerate(self.LINES):
            tokens = words_of(line)
            step = (per_line - 0.9) / max(1, len(tokens))
            t = i * per_line
            for token in tokens:
                words.append({"text": token, "start": round(t, 3), "end": round(t + step * 0.9, 3)})
                t += step
        return {"language": language or "en", "words": words}

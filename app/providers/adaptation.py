"""Singable lyric adaptation.

This is the stage that decides whether the localised song is any good. A literal
translation will not sing: the melody fixes the syllable count and the position
of every stressed beat, so the job is closer to writing new lyrics under formal
constraint than to translating.

Providers here receive line templates (syllable counts, strong beats, rhyme
groups) plus, on retry, per-line feedback from the scorer, and return one line
per template.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .base import AdaptationProvider, Credential, Option, register

SYSTEM = """You adapt song lyrics for singing in another language.

You are not translating. You are writing new lyrics in the target language that \
carry the meaning and emotional register of the original AND fit an existing \
melody exactly. The melody cannot change.

Hard rules, in priority order:
1. Syllable count per line must match the target exactly. This is not negotiable \
— a line with the wrong count cannot be sung.
2. Stressed syllables must land on the marked strong beats. A function word \
(article, preposition, auxiliary) on a strong beat is the most audible failure \
mode in localised songs. Put a naturally stressed syllable there.
3. Preserve the rhyme scheme. Lines sharing a rhyme group must rhyme with each \
other in the target language.
4. Prefer open vowels (a, o, e) on long or high notes. Closed vowels and \
consonant clusters are hard to sustain.
5. Keep the meaning and imagery. Where a literal rendering will not fit, keep the \
emotional intent and change the image rather than padding with filler.

Never pad a line with meaningless syllables to hit the count. Rewrite the line.

Return only a JSON object: {"lines": ["line 1", "line 2", ...]} with exactly one \
entry per target line, in order. No commentary, no markdown fences."""


def build_prompt(
    templates: list[Any],
    source_lang: str,
    target_lang: str,
    instructions: str,
    previous: list[str] | None,
    feedback: list[dict[str, Any]] | None,
) -> str:
    spec = "\n".join(
        f"{t.brief()}\n    original: {t.text}" for t in templates
    )
    parts = [
        f"Source language: {source_lang}",
        f"Target language: {target_lang}",
        "",
        "Lines to adapt (syllable counts are counted in the TARGET language):",
        spec,
    ]
    if instructions:
        parts += ["", f"Additional direction from the artist: {instructions}"]
    if previous and feedback:
        problems = []
        for fb in feedback:
            if not fb.get("issues"):
                continue
            i = fb["index"]
            line = previous[i] if i < len(previous) else ""
            problems.append(
                f"L{i + 1} \"{line}\" — {'; '.join(fb['issues'])}"
            )
        if problems:
            parts += [
                "",
                "Your previous attempt had these problems. Fix them. Keep the lines "
                "that are not listed as they are:",
                *problems,
            ]
    parts += ["", 'Return JSON: {"lines": [...]}']
    return "\n".join(parts)


def parse_lines(text: str, expected: int) -> list[str]:
    """Pull the line array out of a model response, tolerating stray formatting."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError(f"could not parse model output as JSON: {cleaned[:400]}")
        data = json.loads(match.group(0))
    lines = data.get("lines") if isinstance(data, dict) else data
    if not isinstance(lines, list):
        raise ValueError("model output has no 'lines' array")
    lines = [str(x).strip() for x in lines]
    if len(lines) < expected:
        lines += [""] * (expected - len(lines))
    return lines[:expected]


@register
class AnthropicAdaptation(AdaptationProvider):
    id = "anthropic"
    label = "Anthropic"
    docs = "https://docs.claude.com/en/api/messages"
    notes = ("Handles formal constraint well and holds the imagery while it counts "
             "syllables. A good default for the adaptation stage.")
    credentials = [Credential("ANTHROPIC_API_KEY", "Anthropic API key")]
    options = [
        Option("model", "Model", default="claude-sonnet-5"),
        Option("max_tokens", "Max tokens", default=4000),
        Option("temperature", "Temperature", default=1.0),
    ]

    def adapt(self, templates, source_lang, target_lang, instructions,
              previous=None, feedback=None) -> list[str]:
        key = self.require("ANTHROPIC_API_KEY")
        prompt = build_prompt(templates, source_lang, target_lang, instructions, previous, feedback)
        with self.client(headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }) as c:
            r = c.post("https://api.anthropic.com/v1/messages", json={
                "model": self.opt("model"),
                "max_tokens": int(self.opt("max_tokens")),
                "temperature": float(self.opt("temperature")),
                "system": SYSTEM,
                "messages": [{"role": "user", "content": prompt}],
            })
            r.raise_for_status()
            payload = r.json()
        text = "".join(b.get("text", "") for b in payload.get("content", []) if b.get("type") == "text")
        return parse_lines(text, len(templates))


@register
class OpenAIAdaptation(AdaptationProvider):
    id = "openai"
    label = "OpenAI"
    docs = "https://platform.openai.com/docs/api-reference/chat"
    notes = ("Set base_url to use any OpenAI-compatible endpoint, including a local "
             "model, without writing a new adapter.")
    credentials = [Credential("OPENAI_API_KEY", "OpenAI API key")]
    options = [
        Option("model", "Model", default="gpt-4o"),
        Option("base_url", "Base URL", default="https://api.openai.com/v1",
               help="Point this at any OpenAI-compatible endpoint"),
        Option("temperature", "Temperature", default=0.9),
    ]

    def adapt(self, templates, source_lang, target_lang, instructions,
              previous=None, feedback=None) -> list[str]:
        key = self.require("OPENAI_API_KEY")
        prompt = build_prompt(templates, source_lang, target_lang, instructions, previous, feedback)
        with self.client(headers={"Authorization": f"Bearer {key}"}) as c:
            r = c.post(f"{self.opt('base_url').rstrip('/')}/chat/completions", json={
                "model": self.opt("model"),
                "temperature": float(self.opt("temperature")),
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            })
            r.raise_for_status()
            payload = r.json()
        return parse_lines(payload["choices"][0]["message"]["content"], len(templates))


@register
class GeminiAdaptation(AdaptationProvider):
    id = "gemini"
    label = "Google Gemini"
    docs = "https://ai.google.dev/api/generate-content"
    notes = "Long context, useful when you adapt a whole record in one call."
    credentials = [Credential("GEMINI_API_KEY", "Google AI Studio API key")]
    options = [Option("model", "Model", default="gemini-2.5-pro")]

    def adapt(self, templates, source_lang, target_lang, instructions,
              previous=None, feedback=None) -> list[str]:
        key = self.require("GEMINI_API_KEY")
        prompt = build_prompt(templates, source_lang, target_lang, instructions, previous, feedback)
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.opt('model')}:generateContent")
        with self.client() as c:
            r = c.post(url, params={"key": key}, json={
                "system_instruction": {"parts": [{"text": SYSTEM}]},
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json"},
            })
            r.raise_for_status()
            payload = r.json()
        text = "".join(
            p.get("text", "")
            for p in payload["candidates"][0]["content"]["parts"]
        )
        return parse_lines(text, len(templates))


@register
class ManualAdaptation(AdaptationProvider):
    id = "manual"
    label = "Human writer"
    kind = "manual"
    notes = "Skips the model. Paste the adapted lines and the scorer checks the fit."
    options = [Option("lines", "Adapted lines", default="", help="One per source line")]

    def adapt(self, templates, source_lang, target_lang, instructions,
              previous=None, feedback=None) -> list[str]:
        raw = (self.opt("lines") or "").splitlines()
        lines = [ln.strip() for ln in raw if ln.strip()]
        if len(lines) < len(templates):
            lines += [""] * (len(templates) - len(lines))
        return lines[:len(templates)]


@register
class MockAdaptation(AdaptationProvider):
    id = "mock"
    label = "Mock (echo)"
    kind = "local"
    notes = "Returns the original lines. For testing the pipeline without a model."

    def adapt(self, templates, source_lang, target_lang, instructions,
              previous=None, feedback=None) -> list[str]:
        return [t.text for t in templates]

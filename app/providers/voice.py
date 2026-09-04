"""Producing the new lead vocal.

Two routes, and the choice matters more than which vendor you pick:

  Conversion (svc-*)  — a singer tracks a guide vocal in the target language,
                        then the timbre is converted to the artist's voice.
                        Best quality, keeps vocal identity, needs a singer per
                        language. Scales linearly with cost.

  Synthesis (svs-*)   — the melody plus the adapted syllables are rendered by a
                        singing engine. No singer needed, fully batchable, but
                        audibly synthetic on exposed leads. Good for backing
                        vocals, doubles, and demo passes.

The synthesis engines (Synthesizer V, ACE Studio, VOCALOID) are desktop software
with no public API, so `svs-midi` exports a MIDI file with the syllables already
placed on the melody. That file opens directly in all three.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..audio import Note, assign_syllables_to_notes, silent_wav, write_midi
from ..prosody import syllabify, words_of
from .base import (
    Credential, Option, ProviderNotImplemented, VoiceProvider, register,
)


def _syllable_slots(lines: list[str], templates: list[Any], lang: str) -> list[tuple[float, float, list[str]]]:
    """(start, end, [syllable, ...]) per line, for placing lyrics on notes."""
    out = []
    for line, template in zip(lines, templates):
        syllables: list[str] = []
        for word in words_of(line):
            syllables.extend(syllabify(word, lang))
        out.append((template.start, template.end, syllables))
    return out


@register
class MidiHandoffVoice(VoiceProvider):
    id = "svs-midi"
    label = "MIDI for Synthesizer V / ACE Studio / VOCALOID"
    kind = "manual"
    notes = (
        "Exports the melody with the adapted syllables attached as lyric events. "
        "Open it in your singing engine, pick a voice, render, then upload the "
        "result back to this job. The engines have no public API, so this is the "
        "handoff point."
    )
    options = [Option("tempo", "Tempo (BPM)", default=120.0)]

    def render(self, lines, notes, target_lang, workdir, context) -> dict[str, Any]:
        templates = context["templates"]
        slots = _syllable_slots(lines, templates, target_lang)
        placed = assign_syllables_to_notes([Note(**n) if isinstance(n, dict) else n for n in notes], slots)
        midi_path = workdir / f"vocal_{target_lang}.mid"
        write_midi(placed, midi_path, tempo_bpm=float(self.opt("tempo")))
        lyric_path = workdir / f"lyrics_{target_lang}.txt"
        lyric_path.write_text("\n".join(lines), encoding="utf-8")
        return {
            "vocal": None,
            "artifacts": [midi_path, lyric_path],
            "manual_step": (
                f"Open {midi_path.name} in Synthesizer V or ACE Studio, choose a "
                f"{target_lang} voice, render to WAV, then POST it to "
                "/jobs/{id}/assets with role=rendered_vocal."
            ),
        }


@register
class RvcVoice(VoiceProvider):
    id = "svc-rvc"
    label = "RVC (local conversion)"
    kind = "local"
    docs = "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI"
    notes = (
        "Open source, self-hosted, scriptable. Converts an uploaded guide vocal to "
        "your trained voice model. Requires a guide vocal per language."
    )
    options = [
        Option("model_path", "Voice model (.pth)", default=""),
        Option("index_path", "Feature index (.index)", default=""),
        Option("transpose", "Transpose (semitones)", default=0),
    ]

    def render(self, lines, notes, target_lang, workdir, context) -> dict[str, Any]:
        guide = context.get("guide_vocal")
        if not guide:
            return {
                "vocal": None, "artifacts": [],
                "manual_step": (
                    "Record a guide vocal of the adapted lyrics, then POST it to "
                    "/jobs/{id}/assets with role=guide_vocal and re-run this stage."
                ),
            }
        raise ProviderNotImplemented(
            "Wire this to your RVC install: call its inference CLI with "
            f"model={self.opt('model_path')} on {guide}, and return the output path "
            "as {'vocal': Path}."
        )


@register
class KitsVoice(VoiceProvider):
    id = "svc-kits"
    label = "Kits.ai"
    docs = "https://docs.kits.ai"
    notes = "Hosted singing voice conversion. Upload a guide vocal, convert to your voice model."
    credentials = [Credential("KITS_API_KEY", "Kits.ai API key")]
    options = [Option("voice_model_id", "Voice model ID", default="")]

    BASE = "https://arpeggi.io/api/kits/v1"

    def render(self, lines, notes, target_lang, workdir, context) -> dict[str, Any]:
        key = self.require("KITS_API_KEY")
        guide = context.get("guide_vocal")
        model_id = self.opt("voice_model_id")
        if not guide:
            return {
                "vocal": None, "artifacts": [],
                "manual_step": "Upload a guide vocal (role=guide_vocal) and re-run this stage.",
            }
        if not model_id:
            raise ProviderNotImplemented("Set 'voice_model_id' to your Kits.ai voice model.")
        with self.client(headers={"Authorization": f"Bearer {key}"}) as c:
            with open(guide, "rb") as fh:
                r = c.post(
                    f"{self.BASE}/voice-conversions",
                    data={"voiceModelId": str(model_id)},
                    files={"soundFile": (Path(guide).name, fh, "audio/wav")},
                )
            r.raise_for_status()
            job_id = r.json()["id"]
        return {
            "vocal": None,
            "artifacts": [],
            "manual_step": (
                f"Kits.ai conversion {job_id} submitted. Poll "
                f"{self.BASE}/voice-conversions/{job_id} and POST the finished audio "
                "to /jobs/{id}/assets with role=rendered_vocal. "
                "(Implement polling here to make this fully automatic.)"
            ),
            "external_job": job_id,
        }


@register
class ElevenLabsVoice(VoiceProvider):
    id = "svc-elevenlabs"
    label = "ElevenLabs (speech only)"
    docs = "https://elevenlabs.io/docs"
    notes = (
        "Speech synthesis, not singing. Included for spoken-word sections, intros "
        "and narration only. It will not follow a melody — do not use it for sung lines."
    )
    credentials = [Credential("ELEVENLABS_API_KEY", "ElevenLabs API key")]
    options = [Option("voice_id", "Voice ID", default=""),
               Option("model_id", "Model", default="eleven_multilingual_v2")]

    def render(self, lines, notes, target_lang, workdir, context) -> dict[str, Any]:
        key = self.require("ELEVENLABS_API_KEY")
        voice_id = self.opt("voice_id")
        if not voice_id:
            raise ProviderNotImplemented("Set 'voice_id' to an ElevenLabs voice.")
        out = workdir / f"spoken_{target_lang}.mp3"
        with self.client(headers={"xi-api-key": key}) as c:
            r = c.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                json={"text": "\n".join(lines), "model_id": self.opt("model_id")},
            )
            r.raise_for_status()
            out.write_bytes(r.content)
        return {"vocal": out, "artifacts": [out], "manual_step": None}


@register
class UploadedVocal(VoiceProvider):
    id = "uploaded"
    label = "Vocal I rendered elsewhere"
    kind = "manual"
    notes = "Uses whatever was uploaded as role=rendered_vocal. The exit hatch for any engine."

    def render(self, lines, notes, target_lang, workdir, context) -> dict[str, Any]:
        rendered = context.get("rendered_vocal")
        if not rendered:
            return {
                "vocal": None, "artifacts": [],
                "manual_step": "POST your rendered vocal to /jobs/{id}/assets with role=rendered_vocal.",
            }
        return {"vocal": Path(rendered), "artifacts": [], "manual_step": None}


@register
class MockVoice(VoiceProvider):
    id = "mock"
    label = "Mock (silence)"
    kind = "local"
    notes = "Renders silence of the right length. For testing the pipeline without an engine."

    def render(self, lines, notes, target_lang, workdir, context) -> dict[str, Any]:
        templates = context["templates"]
        end = max((t.end for t in templates), default=1.0)
        out = silent_wav(workdir / f"vocal_{target_lang}.wav", end)
        return {"vocal": out, "artifacts": [out], "manual_step": None}

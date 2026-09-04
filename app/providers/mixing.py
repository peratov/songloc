"""Placing the new lead back on the instrumental bed."""

from __future__ import annotations

from pathlib import Path

from ..audio import mix as ffmpeg_mix
from .base import MixProvider, Option, register


@register
class FfmpegMix(MixProvider):
    id = "ffmpeg"
    label = "ffmpeg sum"
    kind = "local"
    notes = (
        "A straight sum at the gains you set. Enough for approvals and demos. "
        "For release, take the stems into a DAW — the localised vocal needs its own "
        "de-essing, level ride and reverb match."
    )
    options = [
        Option("vocal_gain_db", "Vocal gain (dB)", default=0.0),
        Option("instrumental_gain_db", "Instrumental gain (dB)", default=0.0),
    ]

    def mix(self, instrumental: Path, vocal: Path, workdir: Path, options: dict) -> Path:
        out = workdir / "master.wav"
        return ffmpeg_mix(
            instrumental, vocal, out,
            vocal_gain_db=float(options.get("vocal_gain_db", self.opt("vocal_gain_db"))),
            instrumental_gain_db=float(
                options.get("instrumental_gain_db", self.opt("instrumental_gain_db"))
            ),
        )


@register
class StemsOnlyMix(MixProvider):
    id = "stems-only"
    label = "Deliver stems, no mix"
    kind = "local"
    notes = "Skips mixing and hands off the vocal and instrumental for your engineer."

    def mix(self, instrumental: Path, vocal: Path, workdir: Path, options: dict) -> Path:
        return vocal

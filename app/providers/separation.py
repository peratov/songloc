"""Stem separation. You only need this if you are localising a mixdown.

If you have the multitrack session — and for your own recordings you do — skip
this stage entirely by uploading the instrumental bounce directly. Separation is
lossy; a real instrumental stem always sounds better than a separated one.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from ..audio import FFMPEG
from .base import (
    Credential, Option, ProviderNotImplemented, SeparationProvider, register,
)


@register
class PassthroughSeparation(SeparationProvider):
    id = "stems-provided"
    label = "Use my own stems"
    kind = "local"
    notes = "No separation. Expects an instrumental (and optionally a vocal) already uploaded."

    def separate(self, source: Path, workdir: Path) -> dict[str, Path]:
        instrumental = workdir / "instrumental_provided.wav"
        vocal = workdir / "vocal_provided.wav"
        found = {}
        if instrumental.exists():
            found["instrumental"] = instrumental
        if vocal.exists():
            found["vocal"] = vocal
        if "instrumental" not in found:
            raise FileNotFoundError(
                "stems-provided needs an instrumental upload "
                "(POST /jobs/{id}/assets with role=instrumental)"
            )
        found.setdefault("vocal", source)
        return found


@register
class DemucsSeparation(SeparationProvider):
    id = "demucs"
    label = "Demucs (local)"
    kind = "local"
    docs = "https://github.com/adefossez/demucs"
    notes = "Free and good. Needs a GPU to be fast; CPU works but runs ~10x real time."
    options = [
        Option("model", "Model", default="htdemucs_ft",
               choices=["htdemucs", "htdemucs_ft", "mdx_extra"]),
        Option("device", "Device", default="cpu", choices=["cpu", "cuda"]),
    ]

    def separate(self, source: Path, workdir: Path) -> dict[str, Path]:
        if shutil.which("demucs") is None:
            raise ProviderNotImplemented(
                "demucs is not installed. `pip install demucs`, or switch this "
                "stage to stems-provided / a hosted provider."
            )
        out = workdir / "demucs"
        out.mkdir(exist_ok=True)
        subprocess.run(
            ["demucs", "--two-stems", "vocals", "-n", self.opt("model"),
             "-d", self.opt("device"), "-o", str(out), str(source)],
            check=True, capture_output=True,
        )
        stem_dir = next((out / self.opt("model")).iterdir())
        return {
            "vocal": stem_dir / "vocals.wav",
            "instrumental": stem_dir / "no_vocals.wav",
        }


@register
class MusicAiSeparation(SeparationProvider):
    id = "music-ai"
    label = "Music.ai"
    docs = "https://music.ai/docs"
    notes = "Hosted separation built for music rather than speech. Job-based API."
    credentials = [
        Credential("MUSIC_AI_API_KEY", "Music.ai API key",
                   help="Dashboard → Applications → API key"),
    ]
    options = [Option("workflow", "Workflow ID", default="",
                      help="The Music.ai workflow slug that outputs vocal + instrumental stems")]

    BASE = "https://api.music.ai/api"

    def separate(self, source: Path, workdir: Path) -> dict[str, Path]:
        key = self.require("MUSIC_AI_API_KEY")
        workflow = self.opt("workflow")
        if not workflow:
            raise ProviderNotImplemented(
                "Set the 'workflow' option to the Music.ai workflow that returns stems."
            )
        headers = {"Authorization": key}
        with self.client(headers=headers) as c:
            # 1. upload
            up = c.get(f"{self.BASE}/upload").json()
            with open(source, "rb") as fh:
                c.put(up["uploadUrl"], content=fh.read())
            # 2. create job
            job = c.post(f"{self.BASE}/job", json={
                "name": f"songloc-{source.stem}",
                "workflow": workflow,
                "params": {"inputUrl": up["downloadUrl"]},
            }).json()
            # 3. poll
            result = self._poll(c, job["id"])
        return self._download(result, workdir)

    def _poll(self, c, job_id: str, interval: float = 5.0, limit: float = 1800.0):
        deadline = time.time() + limit
        while time.time() < deadline:
            data = c.get(f"{self.BASE}/job/{job_id}").json()
            if data.get("status") == "SUCCEEDED":
                return data.get("result", {})
            if data.get("status") == "FAILED":
                raise RuntimeError(f"Music.ai job failed: {data.get('error')}")
            time.sleep(interval)
        raise TimeoutError("Music.ai job did not finish in time")

    def _download(self, result: dict, workdir: Path) -> dict[str, Path]:
        mapping = {"vocal": None, "instrumental": None}
        for key, url in result.items():
            k = key.lower()
            if "vocal" in k and "no" not in k:
                mapping["vocal"] = url
            elif any(t in k for t in ("instrumental", "accompaniment", "backing", "no_vocals")):
                mapping["instrumental"] = url
        out: dict[str, Path] = {}
        with self.client() as c:
            for role, url in mapping.items():
                if not url:
                    continue
                dest = workdir / f"{role}.wav"
                dest.write_bytes(c.get(url).content)
                out[role] = dest
        if "instrumental" not in out:
            raise RuntimeError(f"workflow returned no instrumental stem: {list(result)}")
        return out


@register
class LalalSeparation(SeparationProvider):
    id = "lalal"
    label = "LALAL.ai"
    docs = "https://www.lalal.ai/api/help/"
    notes = "Pay-per-minute. Upload, then poll /check/ for the split result."
    credentials = [Credential("LALAL_API_KEY", "LALAL.ai licence key")]

    def separate(self, source: Path, workdir: Path) -> dict[str, Path]:
        raise ProviderNotImplemented(
            "LALAL.ai adapter is a stub. Their upload/split/check endpoints are "
            "documented at https://www.lalal.ai/api/help/ — implement separate() "
            "to return {'vocal': Path, 'instrumental': Path}."
        )


@register
class MockSeparation(SeparationProvider):
    id = "mock"
    label = "Mock (no-op)"
    kind = "local"
    notes = "Copies the source to both roles. For wiring up and testing the pipeline."

    def separate(self, source: Path, workdir: Path) -> dict[str, Path]:
        vocal = workdir / "vocal.wav"
        instrumental = workdir / "instrumental.wav"
        for dest in (vocal, instrumental):
            subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(source),
                            "-ac", "1", "-ar", "44100", str(dest)], check=True)
        return {"vocal": vocal, "instrumental": instrumental}

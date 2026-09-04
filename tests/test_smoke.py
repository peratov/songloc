"""Smoke tests. Run with:  python -m pytest tests/ -q

These exercise the whole pipeline using the mock providers, so they pass with no
API keys configured. They need ffmpeg on PATH.
"""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.prosody import (  # noqa: E402
    LineTemplate, Word, build_templates, count_syllables, rhymes, score_lines,
)
from app.providers import REGISTRY, Stage  # noqa: E402


@pytest.fixture(scope="session")
def tone_file(tmp_path_factory):
    """A stepped melody so the pitch tracker has notes to find."""
    path = tmp_path_factory.mktemp("audio") / "song.wav"
    pitches = [262, 294, 330, 349, 392, 349, 330, 294]
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for f in pitches:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency={f}:duration=1.0"]
    cmd += [
        "-filter_complex",
        "".join(f"[{i}:a]" for i in range(len(pitches))) + f"concat=n={len(pitches)}:v=0:a=1[o]",
        "-map", "[o]", "-ar", "44100", "-ac", "1", str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


# -- prosody ---------------------------------------------------------------

@pytest.mark.parametrize("text,lang,low,high", [
    ("I was counting all the hours till the morning", "en", 10, 13),
    ("estaba contando las horas hasta el amanecer", "es", 14, 17),
    ("あさまでじかんをかぞえてた", "ja", 12, 14),
    ("我一直数着到天亮的时间", "zh", 10, 12),
])
def test_syllable_counts_are_in_range(text, lang, low, high):
    assert low <= count_syllables(text, lang) <= high


def test_rhyme_detection():
    assert rhymes("till the morning", "became a warning")
    assert rhymes("hasta el día", "me advertía", "es")
    assert not rhymes("the morning", "the light")


def test_templates_capture_lines_and_rhyme_scheme():
    lines = [
        ("I was counting all the hours till the morning", 0.0),
        ("You were somewhere in the shadow of the light", 4.0),
        ("And the quiet in the room became a warning", 8.0),
        ("So I carried what was left of it tonight", 12.0),
    ]
    words = []
    for text, offset in lines:
        tokens = text.split()
        step = 3.2 / len(tokens)
        for i, tok in enumerate(tokens):
            words.append(Word(tok, offset + i * step, offset + i * step + step * 0.85))

    templates = build_templates(words, lang="en")
    assert len(templates) == 4
    assert [t.rhyme_group for t in templates] == ["A", "B", "A", "B"]
    assert all(t.syllables > 0 for t in templates)
    assert all(0 in t.strong_positions for t in templates)


def test_scorer_flags_a_line_that_is_too_long():
    template = LineTemplate(
        index=0, text="original", lang="es", start=0.0, end=3.0, syllables=11,
        strong_positions=[0, 4, 8],
    )
    good = score_lines(["Yo contaba cada hora hasta el día"], [template], "es")[0]
    bad = score_lines(
        ["El silencio de la sala me advertía por completo"], [template], "es"
    )[0]
    assert good.fit > bad.fit
    assert bad.issues and "too many" in bad.issues[0]


def test_scorer_round_trips_through_to_dict():
    template = LineTemplate(index=0, text="x", lang="en", start=0, end=2, syllables=8)
    restored = LineTemplate.from_dict(template.to_dict())
    assert restored.syllables == 8


# -- audio -----------------------------------------------------------------

def test_pitch_tracking_finds_the_melody(tone_file):
    from app.audio import vocal_to_notes
    notes = vocal_to_notes(tone_file)
    assert len(notes) >= 6
    assert notes[0].pitch == 60          # middle C
    assert all(n.duration > 0 for n in notes)


def test_midi_export_carries_lyrics(tmp_path, tone_file):
    import mido
    from app.audio import Note, assign_syllables_to_notes, write_midi

    notes = [Note(i * 0.5, i * 0.5 + 0.4, 60 + i) for i in range(4)]
    placed = assign_syllables_to_notes(notes, [(0.0, 2.0, ["mor", "ning", "light", "now"])])
    path = write_midi(placed, tmp_path / "out.mid")
    events = [m.text for tr in mido.MidiFile(path).tracks for m in tr if m.type == "lyrics"]
    assert events == ["mor", "ning", "light", "now"]


# -- providers -------------------------------------------------------------

def test_every_stage_has_providers_registered():
    for stage in Stage:
        assert REGISTRY[stage], f"{stage.value} has no providers"


def test_provider_descriptions_are_serialisable():
    from app.providers import catalogue
    import json
    json.dumps(catalogue())


# -- pipeline --------------------------------------------------------------

def test_full_run_stops_at_review_then_completes(tone_file, monkeypatch, tmp_path):
    monkeypatch.setenv("SONGLOC_DATA_DIR", str(tmp_path))
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    job = client.post("/jobs", json={
        "title": "test",
        "source_lang": "en",
        "target_langs": ["es"],
        "stages": {s: {"provider": "mock"}
                   for s in ("separate", "transcribe", "adapt", "voice")},
    }).json()
    jid = job["id"]

    with open(tone_file, "rb") as fh:
        r = client.post(f"/jobs/{jid}/assets", data={"role": "source"},
                        files={"file": ("song.wav", fh, "audio/wav")})
    assert r.status_code == 200

    state = client.post(f"/jobs/{jid}/run?wait=true").json()
    assert state["status"] == "awaiting_review"
    assert state["renders"]["es"]["mean_fit"] is not None

    detail = client.get(f"/jobs/{jid}/renders/es").json()
    assert detail["lines"] and detail["lines"][0]["target_syllables"] > 0

    done = client.post(f"/jobs/{jid}/renders/es/approve", json={
        "lines": [
            "Yo contaba cada hora hasta el día",
            "Tú estabas en la sombra de la luz",
            "El silencio de la sala me advertía",
            "Y cargué lo que quedaba en esa cruz",
        ],
        "by": "test",
    }).json()
    assert done["status"] == "done"
    assert done["renders"]["es"]["status"] == "done"

    files = [f["name"] for f in client.get(f"/jobs/{jid}/files").json()["files"]]
    assert any(f.endswith("master.wav") for f in files)


def test_review_can_be_disabled(tone_file, monkeypatch, tmp_path):
    monkeypatch.setenv("SONGLOC_DATA_DIR", str(tmp_path))
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    job = client.post("/jobs", json={
        "title": "no review", "source_lang": "en", "target_langs": ["pt"],
        "require_review": False,
        "stages": {s: {"provider": "mock"}
                   for s in ("separate", "transcribe", "adapt", "voice")},
    }).json()
    with open(tone_file, "rb") as fh:
        client.post(f"/jobs/{job['id']}/assets", data={"role": "source"},
                    files={"file": ("song.wav", fh, "audio/wav")})
    state = client.post(f"/jobs/{job['id']}/run?wait=true").json()
    assert state["status"] == "done"


def test_unknown_provider_is_rejected():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    r = client.post("/jobs", json={
        "title": "bad", "target_langs": ["fr"],
        "stages": {"adapt": {"provider": "nope"}},
    })
    assert r.status_code == 400
    assert "nope" in r.json()["detail"]

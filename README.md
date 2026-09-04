# songloc

A pipeline for localising your own recordings into other languages: stems →
timed lyrics → singable adaptation → new lead vocal → mix.

It is a backend plus a small console. Every external service is a swappable
provider; you enter API keys once and pick a provider per stage per job.

The part that isn't a thin wrapper is the prosody engine. A translated lyric has
to land on a melody that cannot change — same syllable count, stresses on the
same beats, rhymes in the same places. songloc derives those constraints from
your vocal, hands them to the language model as hard requirements, scores what
comes back, and retries the lines that fail. Then it stops and asks a human.

---

## Install

```bash
git clone <your-repo> songloc && cd songloc
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # or enter keys in the console
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000/>. API docs at `/docs`.

**ffmpeg and ffprobe must be on your PATH.** Everything else is pip-installable.
No GPU is required unless you choose the local Demucs or WhisperX providers.

Run the tests — they use mock providers and need no keys:

```bash
python -m pytest tests/ -q
```

---

## Quickstart

The fastest useful configuration for your own material, because you already
have the stems and the correct lyrics:

1. **Connections** → paste an `ANTHROPIC_API_KEY` (or OpenAI, or Gemini).
2. **New job** →
   - Split the master: **Use my own stems**
   - Get timed lyrics: **My lyrics, aligned** — paste the real lyrics, one line per sung line
   - Write singable lyrics: **Anthropic**
   - Produce the vocal: **MIDI for Synthesizer V / ACE Studio / VOCALOID**
   - Localise into: `es, pt, ja`
3. Upload the master, and upload the instrumental separately as `role=instrumental`.
4. Run. The job stops at the review gate with a rhythm ruler per line.
5. Edit anything that doesn't fit, approve, and collect the MIDI from **Files**.

Or from the API:

```bash
JOB=$(curl -s localhost:8000/jobs -H 'Content-Type: application/json' -d '{
  "title": "Harbour Lights",
  "source_lang": "en",
  "target_langs": ["es", "pt"],
  "instructions": "Keep the harbour imagery. Conversational register, not literary.",
  "stages": {
    "transcribe": {"provider": "known-lyrics", "options": {"lyrics": "line one\nline two"}},
    "adapt":      {"provider": "anthropic"},
    "voice":      {"provider": "svs-midi"}
  }
}' | jq -r .id)

curl -s -F role=source -F file=@master.wav        localhost:8000/jobs/$JOB/assets
curl -s -F role=instrumental -F file=@instr.wav   localhost:8000/jobs/$JOB/assets
curl -s -X POST "localhost:8000/jobs/$JOB/run?wait=true" | jq .

curl -s localhost:8000/jobs/$JOB/renders/es | jq '.lines[] | {target_syllables, adapted, fit}'
curl -s -X POST localhost:8000/jobs/$JOB/renders/es/approve \
  -H 'Content-Type: application/json' -d '{"lines": ["..."], "by": "ana"}'
```

---

## How a job moves

```
  upload ──► separate ──► transcribe ──► analyse
                                            │
                          per target language ▼
                        adapt ⇄ score  ──►  REVIEW GATE  ──►  voice  ──►  mix
                        (retries failing         │                │
                         lines up to 3x)         │                │
                                            you approve      may pause for
                                            or edit          a file only you
                                                             can supply
```

A job runs as far as it can and stops at a gate. Resuming is just `POST
/jobs/{id}/run` again — every stage checks whether its output already exists, so
re-running is safe and cheap.

There are two gates:

- **review** — the adapted lyrics are waiting for a human. On by default.
- **asset** — the pipeline needs a file only you can supply: a guide vocal, or a
  vocal you rendered in a desktop singing engine.

### The review gate

This is the difference between a localisation you can release and one you can't,
so it is on by default and the console makes it the main screen. Each line shows:

- a **rhythm ruler** — tall teal ticks are beats that need a stressed syllable,
  short grey ticks are ordinary slots, rose ticks are syllables you've added
  beyond what the melody has room for, dashed ticks are slots you're short
- the original line, and the adapted line as an editable field that recounts
  syllables as you type
- the rhyme group, and any issues the scorer found

Edits are re-scored on approval. You can turn the gate off per job
(`require_review: false`) or globally (`SONGLOC_REQUIRE_REVIEW=false`). Do that
for demos and scratch passes, not for anything you intend to ship.

---

## Providers

Pick one per stage. Anything marked *mock* runs with no keys, so you can wire up
and test the whole pipeline before spending anything.

### Split the master

| id | what it is | needs |
|---|---|---|
| `stems-provided` | No separation — you upload the instrumental. **Use this for your own recordings.** A real stem always beats a separated one. | — |
| `demucs` | Local, open source, good. ~10× real time on CPU. | `pip install demucs` |
| `music-ai` | Hosted, built for music. Job-based API. | `MUSIC_AI_API_KEY` + a workflow id |
| `lalal` | Pay per minute. **Stub — see below.** | `LALAL_API_KEY` |
| `mock` | Copies the source to both roles. | — |

### Get timed lyrics

Word-level timestamps are mandatory here — they're what the prosody engine uses
to find line breaks, syllable slots and stress positions. A provider that returns
plain text is not usable.

| id | what it is | needs |
|---|---|---|
| `known-lyrics` | You paste the real lyrics; timings are estimated by syllable weight. Zero transcription error. **Best for your own material.** | — |
| `whisperx` | Whisper large-v3 + forced alignment. Free, wants a GPU. | `pip install whisperx` |
| `elevenlabs-scribe` | Strong word timings, tuned for speech — isolate the vocal first. | `ELEVENLABS_API_KEY` |
| `music-ai` | Their lyrics module is trained on singing rather than speech. **Stub.** | `MUSIC_AI_API_KEY` |
| `mock` | Four fixed lines. | — |

### Write singable lyrics

| id | needs |
|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` |
| `openai` | `OPENAI_API_KEY` — set `base_url` to use any OpenAI-compatible endpoint, including a local model |
| `gemini` | `GEMINI_API_KEY` |
| `manual` | Nothing. You paste the lines, the scorer checks them. |
| `mock` | Echoes the original. |

### Produce the vocal

Two routes, and this choice matters more than which vendor you pick.

| id | route | needs |
|---|---|---|
| `svs-midi` | **Synthesis.** Exports MIDI with your adapted syllables already placed on the melody. Opens in Synthesizer V, ACE Studio or VOCALOID. No singer, fully batchable. | — |
| `svc-rvc` | **Conversion.** Local RVC. Converts a guide vocal to your trained voice. **Stub.** | RVC install + a guide vocal |
| `svc-kits` | **Conversion.** Hosted. Submits the job; polling is left for you to add. | `KITS_API_KEY` + a guide vocal |
| `svc-elevenlabs` | Speech, **not singing**. For spoken intros and narration only — it will not follow a melody. | `ELEVENLABS_API_KEY` |
| `uploaded` | Uses whatever you upload as `role=rendered_vocal`. The exit hatch for any engine. | — |
| `mock` | Silence of the right length. | — |

Conversion keeps your vocal identity and sounds better, but needs a singer per
language. Synthesis needs nobody and scales, but is audibly synthetic on exposed
leads — fine for backing vocals, doubles and demo passes.

Synthesizer V, ACE Studio and VOCALOID have no public API. `svs-midi` is the
handoff: it writes a `.mid` with lyric meta events at syllable level
(`coun`/`ting`, `mor`/`ning`), you render in the engine, then upload the result
back with `role=rendered_vocal&lang=es` and re-run.

### Place it on the bed

| id | what it is |
|---|---|
| `ffmpeg` | A straight sum at the gains you set. Fine for approvals. |
| `stems-only` | Skips mixing and hands off the stems. |

---

## Stubs

Four adapters have the right shape but need your account's specifics before they
work: `separate/lalal`, `transcribe/music-ai`, `voice/svc-rvc`, and polling in
`voice/svc-kits`. They raise `ProviderNotImplemented` with a message saying
exactly what to fill in. The Music.ai upload → create → poll flow is fully
implemented in `separation.py` and is the template for the others.

## Adding a provider

```python
# app/providers/mine.py
from pathlib import Path
from .base import Credential, Option, TranscriptionProvider, register

@register
class MyTranscription(TranscriptionProvider):
    id = "my-service"
    label = "My Service"
    docs = "https://example.com/docs"
    notes = "One line the console shows next to it."
    credentials = [Credential("MY_SERVICE_KEY", "My Service API key")]
    options = [Option("model", "Model", default="v2")]

    def transcribe(self, vocal: Path, language: str | None, workdir: Path) -> dict:
        key = self.require("MY_SERVICE_KEY")
        with self.client(headers={"Authorization": f"Bearer {key}"}) as c:
            ...
        return {"language": "en", "words": [{"text": "hello", "start": 0.0, "end": 0.4}]}
```

Add `mine` to the import line in `app/providers/__init__.py`. That's it — the
credential appears in the console's Connections screen, the provider appears in
the stage dropdown, and `GET /credentials` reports it. Nothing else changes.

## Credentials

Two sources, runtime first:

1. `data/credentials.json` — written by the console or `POST /credentials`,
   chmod 600, never returned to the browser in full
2. environment / `.env`

`GET /credentials` lists every credential any registered provider wants, whether
it's set, and which providers use it.

---

## API

| | |
|---|---|
| `GET /health` | provider counts, whether review is on |
| `GET /providers` | full catalogue with credentials and options |
| `GET /credentials` · `POST /credentials` | read status, write keys |
| `POST /jobs` · `GET /jobs` · `GET /jobs/{id}` · `DELETE /jobs/{id}` | job CRUD |
| `POST /jobs/{id}/assets` | multipart upload; `role` ∈ source, instrumental, vocal, guide_vocal, rendered_vocal; optional `lang` |
| `POST /jobs/{id}/run` | advance the job; `?wait=true` to run synchronously |
| `GET /jobs/{id}/renders/{lang}` | line-by-line originals, adaptations, constraints, fit scores |
| `POST /jobs/{id}/renders/{lang}/approve` | approve, optionally replacing lines |
| `GET /jobs/{id}/files` · `GET /jobs/{id}/files/{path}` | list and download artifacts |
| `POST /prosody/syllables` · `POST /prosody/score` | prosody utilities, usable standalone |

The two prosody endpoints work without a job, so you can score lyrics a human
wrote by hand, or wire the scorer into whatever you already use.

---

## How the scoring works

`app/prosody.py` is the interesting file.

**Syllable counting** uses pyphen for ~19 Latin/Cyrillic/Greek languages, mora
counting for Japanese (kana minus small kana, `ー` and `っ` counting as one), and
character counting for Chinese and Korean. English falls back to vowel-group
counting with the usual corrections for silent terminal `e`, `-le`, and `-ed`.

**Line templates** come from the aligned words. A gap of ≥0.6s ends a line. Each
line records its syllable count, its duration, its rhyme group, and which slots
are strong beats — slots held noticeably longer than the song's median, plus the
first syllable of every multi-syllable word, plus the line opening.

**Fit** weights syllable count 0.5, stress placement 0.3, rhyme 0.2, then
penalises anything that would push the singer past ~135% of the original
syllable rate. Being off by one syllable is tolerated — elision and split notes
absorb it. Being off by more is not.

The adaptation loop scores every returned line, sends the failures back with
specific complaints (`L3 "…" — 4 syllables too many (needs 12); breaks rhyme
group A`), and retries up to `SONGLOC_MAX_ADAPT_ROUNDS` times.

---

## What this doesn't do

- **It doesn't mix for release.** `ffmpeg` sums the vocal onto the bed at fixed
  gains. A localised vocal needs its own de-essing, level ride and reverb match;
  take the stems into a DAW.
- **It doesn't know how anything sounds.** The scorer counts syllables and
  guesses at stress from word length and position. It cannot hear that a line
  is ugly, that a vowel is wrong on a high note, or that a phrase is idiomatic
  nonsense. That is what the review gate is for.
- **Stress detection is coarse.** Without a pronunciation dictionary per
  language, "first syllable of a polysyllabic word" is the proxy. It reliably
  catches a function word sitting on a downbeat, which is the failure you can
  actually hear. It will miss subtler things.
- **Synthesis is not a finished lead vocal.** For anything exposed, use the
  conversion route or a real singer.

## Cost

Per-song cost is dominated by human review time, not API calls. Get the workflow
right on one song into two or three languages before pointing it at a catalogue —
that first song tells you whether the catalogue-wide version is viable.

## Licensing

Built on the assumption that you own the recordings and the publishing. If you
run this on material you don't own, a translated lyric is a derivative work and
needs the publisher's licence.

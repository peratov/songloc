"""The orchestrator.

A job runs as far as it can and then stops at a gate. There are two kinds:

  review  — the adapted lyrics are waiting for a human. This is on by default and
            it is the difference between a localisation you can release and one
            you can't. Turn it off per job with require_review=false.

  asset   — the pipeline needs a file only you can supply: a guide vocal, or a
            vocal rendered in a desktop singing engine.

Resuming is just calling run() again. Every stage is idempotent — it checks
whether its output already exists before doing the work.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from . import store
from .audio import Note, duration_seconds, vocal_to_notes
from .config import settings
from .prosody import LineTemplate, Word, build_templates, score_lines
from .providers import Stage, get as get_provider
from .providers.base import MissingCredential, ProviderNotImplemented

DEFAULT_STAGES = {
    "separate": {"provider": "mock", "options": {}},
    "transcribe": {"provider": "mock", "options": {}},
    "adapt": {"provider": "mock", "options": {}},
    "voice": {"provider": "mock", "options": {}},
    "mix": {"provider": "ffmpeg", "options": {}},
}


class Gate(Exception):
    """Stops the run cleanly and records what the job is waiting for."""

    def __init__(self, kind: str, message: str, detail: dict | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.detail = detail or {}


def _stage_config(job: dict, stage: str) -> tuple[str, dict]:
    cfg = {**DEFAULT_STAGES, **job.get("stages", {})}.get(stage, DEFAULT_STAGES[stage])
    return cfg.get("provider", "mock"), cfg.get("options", {}) or {}


def _templates_from_job(job: dict) -> list[LineTemplate]:
    return [LineTemplate.from_dict(t) for t in job["analysis"]["templates"]]


def _notes_from_job(job: dict) -> list[Note]:
    return [Note(**n) for n in job["analysis"].get("notes", [])]


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------

def stage_separate(job: dict) -> None:
    if job.get("stems"):
        return
    workdir = store.workdir(job["id"])
    source = job["assets"].get("source")
    if not source:
        raise Gate("asset", "Upload the song first (role=source).", {"role": "source"})

    provider_id, options = _stage_config(job, "separate")
    provider = get_provider(Stage.SEPARATE, provider_id, options)
    store.log(job, f"separate: {provider_id}")
    stems = provider.separate(Path(source), workdir)
    job["stems"] = {k: str(v) for k, v in stems.items()}
    store.log(job, f"separate: got {', '.join(job['stems'])}")


def stage_transcribe(job: dict) -> None:
    if job.get("transcript"):
        return
    workdir = store.workdir(job["id"])
    vocal = job["stems"].get("vocal")
    if not vocal:
        raise Gate("asset", "No vocal stem to transcribe.", {"role": "vocal"})

    provider_id, options = _stage_config(job, "transcribe")
    provider = get_provider(Stage.TRANSCRIBE, provider_id, options)
    store.log(job, f"transcribe: {provider_id}")
    result = provider.transcribe(Path(vocal), job.get("source_lang"), workdir)
    if not result.get("words"):
        raise RuntimeError(
            f"{provider_id} returned no word timings. This pipeline needs word-level "
            "timestamps — switch to a provider that produces them."
        )
    job["transcript"] = result
    job["source_lang"] = result.get("language") or job.get("source_lang") or "en"
    store.log(job, f"transcribe: {len(result['words'])} words, language {job['source_lang']}")


def stage_analyze(job: dict) -> None:
    """Local. Builds the singability constraints and the melody note list."""
    if job.get("analysis"):
        return
    lang = job["source_lang"]
    words = [Word(**w) for w in job["transcript"]["words"]]
    templates = build_templates(words, lang=lang, gap_threshold=job.get("line_gap", 0.6))

    notes: list[Note] = []
    vocal = job["stems"].get("vocal")
    if vocal and job.get("extract_melody", True):
        try:
            notes = vocal_to_notes(Path(vocal))
            store.log(job, f"analyze: {len(notes)} notes tracked from the vocal")
        except Exception as exc:  # melody is optional — only svs-midi needs it
            store.log(job, f"analyze: melody tracking skipped ({exc})", "warn")

    job["analysis"] = {
        "templates": [t.to_dict() for t in templates],
        "notes": [n.to_dict() for n in notes],
        "line_count": len(templates),
        "total_syllables": sum(t.syllables for t in templates),
    }
    store.log(job, f"analyze: {len(templates)} lines, {job['analysis']['total_syllables']} syllables")


def stage_adapt(job: dict, lang: str) -> None:
    render = job["renders"].setdefault(lang, {"lang": lang, "status": "pending"})
    if render.get("approved"):
        return
    templates = _templates_from_job(job)
    provider_id, options = _stage_config(job, "adapt")
    provider = get_provider(Stage.ADAPT, provider_id, options)

    lines = render.get("lines")
    fits = render.get("fits")
    rounds = render.get("rounds", 0)
    max_rounds = int(job.get("max_adaptation_rounds", settings.max_adaptation_rounds))
    threshold = float(job.get("fit_threshold", settings.fit_threshold))

    while rounds < max_rounds:
        feedback = [f for f in (fits or []) if f.get("issues")] or None
        store.log(job, f"adapt[{lang}]: {provider_id} round {rounds + 1}")
        lines = provider.adapt(
            templates,
            source_lang=job["source_lang"],
            target_lang=lang,
            instructions=job.get("instructions", ""),
            previous=lines,
            feedback=feedback,
        )
        fits = [f.to_dict() for f in score_lines(lines, templates, lang)]
        rounds += 1
        weak = [f for f in fits if f["fit"] < threshold]
        store.log(
            job,
            f"adapt[{lang}]: round {rounds} — {len(fits) - len(weak)}/{len(fits)} lines fit",
        )
        if not weak or provider_id in ("manual", "mock"):
            break

    render.update({
        "lines": lines,
        "fits": fits,
        "rounds": rounds,
        "mean_fit": round(sum(f["fit"] for f in fits) / len(fits), 3) if fits else 0.0,
        "needs_attention": [f["index"] for f in fits if f["fit"] < threshold],
        "status": "adapted",
    })


def stage_review(job: dict, lang: str) -> None:
    render = job["renders"][lang]
    if render.get("approved"):
        return
    require = job.get("require_review", settings.require_review)
    if not require:
        render["approved"] = True
        render["approved_by"] = "auto (review disabled)"
        return
    render["status"] = "awaiting_review"
    raise Gate(
        "review",
        f"Adapted lyrics for {lang} are waiting for approval.",
        {
            "lang": lang,
            "needs_attention": render.get("needs_attention", []),
            "mean_fit": render.get("mean_fit"),
            "approve": f"POST /jobs/{job['id']}/renders/{lang}/approve",
        },
    )


def stage_voice(job: dict, lang: str) -> None:
    render = job["renders"][lang]
    if render.get("vocal"):
        return
    workdir = store.workdir(job["id"]) / lang
    workdir.mkdir(parents=True, exist_ok=True)

    provider_id, options = _stage_config(job, "voice")
    provider = get_provider(Stage.VOICE, provider_id, options)
    store.log(job, f"voice[{lang}]: {provider_id}")

    context = {
        "templates": _templates_from_job(job),
        "guide_vocal": job["assets"].get(f"guide_vocal:{lang}") or job["assets"].get("guide_vocal"),
        "rendered_vocal": job["assets"].get(f"rendered_vocal:{lang}")
        or job["assets"].get("rendered_vocal"),
        "instrumental": job["stems"].get("instrumental"),
        "source_lang": job["source_lang"],
    }
    result = provider.render(render["lines"], _notes_from_job(job), lang, workdir, context)

    artifacts = [str(p) for p in result.get("artifacts", [])]
    render["artifacts"] = sorted(set(render.get("artifacts", []) + artifacts))

    if result.get("vocal"):
        render["vocal"] = str(result["vocal"])
        render["status"] = "voiced"
        return

    render["status"] = "awaiting_asset"
    raise Gate("asset", result.get("manual_step") or "A vocal file is needed.", {
        "lang": lang,
        "artifacts": render["artifacts"],
        "upload": f"POST /jobs/{job['id']}/assets  (role=rendered_vocal, lang={lang})",
    })


def stage_mix(job: dict, lang: str) -> None:
    render = job["renders"][lang]
    if render.get("master"):
        return
    workdir = store.workdir(job["id"]) / lang
    workdir.mkdir(parents=True, exist_ok=True)
    instrumental = job["stems"].get("instrumental")
    if not instrumental:
        raise Gate("asset", "No instrumental to mix onto.", {"role": "instrumental"})

    provider_id, options = _stage_config(job, "mix")
    provider = get_provider(Stage.MIX, provider_id, options)
    store.log(job, f"mix[{lang}]: {provider_id}")
    master = provider.mix(Path(instrumental), Path(render["vocal"]), workdir, options)
    render["master"] = str(master)
    render["status"] = "done"
    store.log(job, f"mix[{lang}]: {Path(master).name}")


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def run(job_id: str) -> dict[str, Any]:
    """Advance a job as far as it will go. Safe to call repeatedly."""
    job = store.get(job_id)
    if job is None:
        raise KeyError(job_id)

    if not store.locks.acquire(job_id):
        return job  # already running

    job["status"] = "running"
    job["waiting_on"] = None
    job["error"] = None
    store.save(job)

    gates: list[dict] = []
    try:
        try:
            stage_separate(job)
            stage_transcribe(job)
            stage_analyze(job)
        except Gate as gate:
            job["status"] = f"awaiting_{gate.kind}"
            job["waiting_on"] = {"kind": gate.kind, "message": gate.message, **gate.detail}
            store.log(job, f"waiting: {gate.message}", "warn")
            return store.save(job)

        for lang in job.get("target_langs", []):
            try:
                stage_adapt(job, lang)
                stage_review(job, lang)
                stage_voice(job, lang)
                stage_mix(job, lang)
            except Gate as gate:
                gates.append({"kind": gate.kind, "message": gate.message, **gate.detail})
                store.log(job, f"waiting[{lang}]: {gate.message}", "warn")
            except (MissingCredential, ProviderNotImplemented) as exc:
                job["renders"][lang]["status"] = "blocked"
                job["renders"][lang]["error"] = str(exc)
                store.log(job, f"blocked[{lang}]: {exc}", "error")

        done = [r for r in job["renders"].values() if r.get("status") == "done"]
        if gates:
            job["status"] = f"awaiting_{gates[0]['kind']}"
            job["waiting_on"] = gates[0]
            job["waiting_all"] = gates
        elif done and len(done) == len(job.get("target_langs", [])):
            job["status"] = "done"
            store.log(job, "job complete")
        else:
            job["status"] = "blocked"
        return store.save(job)

    except Exception as exc:
        job["status"] = "failed"
        job["error"] = f"{type(exc).__name__}: {exc}"
        store.log(job, job["error"], "error")
        store.log(job, traceback.format_exc()[-1500:], "error")
        return store.save(job)
    finally:
        store.locks.release(job_id)


def approve(job_id: str, lang: str, lines: list[str] | None = None, by: str = "operator") -> dict:
    """Sign off a render's lyrics, optionally replacing them with edited versions."""
    job = store.get(job_id)
    if job is None:
        raise KeyError(job_id)
    render = job.get("renders", {}).get(lang)
    if render is None:
        raise KeyError(f"no render for {lang}")

    if lines:
        templates = _templates_from_job(job)
        render["lines"] = lines
        render["fits"] = [f.to_dict() for f in score_lines(lines, templates, lang)]
        render["mean_fit"] = round(
            sum(f["fit"] for f in render["fits"]) / len(render["fits"]), 3
        ) if render["fits"] else 0.0
        threshold = float(job.get("fit_threshold", settings.fit_threshold))
        render["needs_attention"] = [f["index"] for f in render["fits"] if f["fit"] < threshold]
        render["edited"] = True

    render["approved"] = True
    render["approved_by"] = by
    render["status"] = "approved"
    store.log(job, f"approved[{lang}] by {by} (mean fit {render.get('mean_fit')})")
    store.save(job)
    return run(job_id)

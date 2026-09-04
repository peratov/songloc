"""HTTP API.

Start with:  uvicorn app.main:app --reload
Console at:  http://127.0.0.1:8000/
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from . import pipeline, store
from .config import ROOT, WORK_DIR, keystore, settings
from .prosody import LineTemplate, count_syllables, rhyme_key, score_lines
from .providers import Stage, catalogue
from .providers.base import REGISTRY

app = FastAPI(
    title="songloc",
    description="Song localisation pipeline: stems → aligned lyrics → singable "
                "adaptation → new lead vocal → mix.",
    version="0.1.0",
)

WEB = ROOT / "web"


# --------------------------------------------------------------------------
# Console
# --------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def console() -> str:
    index = WEB / "index.html"
    if not index.exists():
        return "<h1>songloc</h1><p>API is up. See <a href='/docs'>/docs</a>.</p>"
    return index.read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "review_required_by_default": settings.require_review,
        "fit_threshold": settings.fit_threshold,
        "providers": {s.value: len(p) for s, p in REGISTRY.items()},
    }


# --------------------------------------------------------------------------
# Providers and credentials — "input all the APIs" lives here
# --------------------------------------------------------------------------

@app.get("/providers")
def providers() -> dict[str, Any]:
    return catalogue()


@app.get("/credentials")
def credentials() -> dict[str, Any]:
    """Every credential any registered provider wants, and whether it is set."""
    seen: dict[str, dict[str, Any]] = {}
    for stage, provs in REGISTRY.items():
        for cls in provs.values():
            for cred in cls.credentials:
                entry = seen.setdefault(cred.name, {
                    **keystore.status(cred.name),
                    "label": cred.label,
                    "help": cred.help,
                    "used_by": [],
                })
                entry["used_by"].append(f"{stage.value}/{cls.id}")
    return {"credentials": sorted(seen.values(), key=lambda c: c["name"])}


class CredentialUpdate(BaseModel):
    values: dict[str, str] = Field(
        ..., description="Credential name → value. An empty value deletes it."
    )


@app.post("/credentials")
def set_credentials(body: CredentialUpdate) -> dict[str, Any]:
    written = keystore.set_many(body.values)
    return {"saved": written, "credentials": credentials()["credentials"]}


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------

class StageConfig(BaseModel):
    provider: str
    options: dict[str, Any] = Field(default_factory=dict)


class JobCreate(BaseModel):
    title: str = "untitled"
    source_lang: str | None = None
    target_langs: list[str] = Field(default_factory=list)
    instructions: str = Field(
        "", description="Direction for the adaptation, e.g. 'keep the sister imagery, "
                        "register is conversational not formal'"
    )
    stages: dict[str, StageConfig] = Field(default_factory=dict)
    require_review: bool | None = None
    fit_threshold: float | None = None
    max_adaptation_rounds: int | None = None
    line_gap: float = Field(0.6, description="Seconds of silence that ends a line")
    extract_melody: bool = True


def _validate_stages(stages: dict[str, StageConfig]) -> None:
    for name, cfg in stages.items():
        try:
            stage = Stage(name)
        except ValueError:
            raise HTTPException(400, f"unknown stage '{name}'")
        if cfg.provider not in REGISTRY[stage]:
            known = ", ".join(sorted(REGISTRY[stage]))
            raise HTTPException(400, f"no {name} provider '{cfg.provider}'. Available: {known}")


@app.post("/jobs", status_code=201)
def create_job(body: JobCreate) -> dict[str, Any]:
    _validate_stages(body.stages)
    data = body.model_dump()
    data["stages"] = {k: v.model_dump() for k, v in body.stages.items()}
    for key in ("require_review", "fit_threshold", "max_adaptation_rounds"):
        if data.get(key) is None:
            data.pop(key)
    return store.create(data)


@app.get("/jobs")
def list_jobs(limit: int = 50) -> dict[str, Any]:
    jobs = store.list_jobs(limit)
    return {"jobs": [_summary(j) for j in jobs]}


def _summary(job: dict) -> dict[str, Any]:
    return {
        "id": job["id"],
        "title": job.get("title"),
        "status": job.get("status"),
        "source_lang": job.get("source_lang"),
        "target_langs": job.get("target_langs", []),
        "updated_at": job.get("updated_at"),
        "lines": job.get("analysis", {}).get("line_count"),
        "waiting_on": job.get("waiting_on"),
        "renders": {
            lang: {"status": r.get("status"), "mean_fit": r.get("mean_fit"),
                   "needs_attention": r.get("needs_attention", [])}
            for lang, r in job.get("renders", {}).items()
        },
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return job


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str) -> dict[str, Any]:
    if not store.delete(job_id):
        raise HTTPException(404, "no such job")
    shutil.rmtree(WORK_DIR / job_id, ignore_errors=True)
    return {"deleted": job_id}


ASSET_ROLES = {"source", "instrumental", "vocal", "guide_vocal", "rendered_vocal"}


@app.post("/jobs/{job_id}/assets")
async def upload_asset(
    job_id: str,
    role: str = Form(..., description=f"One of: {', '.join(sorted(ASSET_ROLES))}"),
    lang: str | None = Form(None, description="Target language, for per-language vocals"),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    if role not in ASSET_ROLES:
        raise HTTPException(400, f"role must be one of {sorted(ASSET_ROLES)}")

    workdir = store.workdir(job_id)
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    name = f"{role}_{lang}{suffix}" if lang else f"{role}{suffix}"
    dest = workdir / name
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)

    key = f"{role}:{lang}" if lang else role
    job["assets"][key] = str(dest)

    # Uploading a fresh vocal for a language means that render should re-run.
    if role == "rendered_vocal" and lang and lang in job.get("renders", {}):
        job["renders"][lang].pop("vocal", None)
        job["renders"][lang].pop("master", None)
    # Providing stems directly should be visible to the passthrough separator.
    if role in {"instrumental", "vocal"}:
        alias = workdir / f"{role}_provided.wav"
        if dest.suffix.lower() == ".wav":
            shutil.copyfile(dest, alias)

    store.log(job, f"asset: {key} → {dest.name}")
    store.save(job)
    return {"role": key, "path": str(dest), "job": _summary(job)}


@app.post("/jobs/{job_id}/run")
def run_job(job_id: str, background: BackgroundTasks, wait: bool = False) -> dict[str, Any]:
    if store.get(job_id) is None:
        raise HTTPException(404, "no such job")
    if wait:
        return _summary(pipeline.run(job_id))
    background.add_task(pipeline.run, job_id)
    return {"started": job_id, "poll": f"/jobs/{job_id}"}


class Approval(BaseModel):
    lines: list[str] | None = Field(
        None, description="Replacement lines. Omit to approve as-is."
    )
    by: str = "operator"


@app.post("/jobs/{job_id}/renders/{lang}/approve")
def approve_render(job_id: str, lang: str, body: Approval) -> dict[str, Any]:
    try:
        job = pipeline.approve(job_id, lang, body.lines, body.by)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    return _summary(job)


@app.get("/jobs/{job_id}/renders/{lang}")
def get_render(job_id: str, lang: str) -> dict[str, Any]:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    render = job.get("renders", {}).get(lang)
    if render is None:
        raise HTTPException(404, f"no render for {lang}")
    templates = job.get("analysis", {}).get("templates", [])
    return {
        "render": render,
        "lines": [
            {
                "index": i,
                "original": templates[i]["text"] if i < len(templates) else "",
                "adapted": render.get("lines", [""] * len(templates))[i]
                if i < len(render.get("lines", [])) else "",
                "target_syllables": templates[i]["syllables"] if i < len(templates) else None,
                "strong_positions": templates[i]["strong_positions"] if i < len(templates) else [],
                "rhyme_group": templates[i].get("rhyme_group") if i < len(templates) else None,
                "fit": render.get("fits", [])[i] if i < len(render.get("fits", [])) else None,
            }
            for i in range(len(templates))
        ],
    }


@app.get("/jobs/{job_id}/files")
def list_files(job_id: str) -> dict[str, Any]:
    base = WORK_DIR / job_id
    if not base.exists():
        raise HTTPException(404, "no such job")
    files = [
        {"name": str(p.relative_to(base)), "bytes": p.stat().st_size,
         "url": f"/jobs/{job_id}/files/{p.relative_to(base)}"}
        for p in sorted(base.rglob("*")) if p.is_file()
    ]
    return {"files": files}


@app.get("/jobs/{job_id}/files/{path:path}")
def download_file(job_id: str, path: str):
    base = (WORK_DIR / job_id).resolve()
    target = (base / path).resolve()
    if not str(target).startswith(str(base)) or not target.is_file():
        raise HTTPException(404, "no such file")
    return FileResponse(target, filename=target.name)


# --------------------------------------------------------------------------
# Prosody utilities — usable on their own, no job required
# --------------------------------------------------------------------------

class SyllableRequest(BaseModel):
    text: str
    lang: str = "en"


@app.post("/prosody/syllables")
def syllables(body: SyllableRequest) -> dict[str, Any]:
    return {
        "text": body.text,
        "lang": body.lang,
        "syllables": count_syllables(body.text, body.lang),
        "rhyme_key": rhyme_key(body.text, body.lang),
    }


class ScoreRequest(BaseModel):
    lang: str
    candidates: list[str]
    templates: list[dict[str, Any]] = Field(
        ..., description="Line templates, as returned in a job's analysis.templates"
    )


@app.post("/prosody/score")
def score(body: ScoreRequest) -> dict[str, Any]:
    templates = [LineTemplate.from_dict(t) for t in body.templates]
    fits = score_lines(body.candidates, templates, body.lang)
    return {
        "fits": [f.to_dict() for f in fits],
        "mean_fit": round(sum(f.fit for f in fits) / len(fits), 3) if fits else 0.0,
    }

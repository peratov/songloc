"""Audio helpers. Everything here runs locally on ffmpeg + numpy — no keys needed.

The important piece is `vocal_to_notes`: it turns an isolated vocal into a note
list so the adapted lyrics can be exported as a MIDI file that Synthesizer V,
ACE Studio or VOCALOID will open with the syllables already placed on the melody.
That MIDI export is the interoperability bridge — none of those tools take an
API call, but all of them import MIDI with lyric events.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


class AudioError(RuntimeError):
    pass


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise AudioError(proc.stderr.decode("utf-8", "replace")[-2000:])
    return proc


def duration_seconds(path: str | Path) -> float:
    proc = _run([
        FFPROBE, "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(path),
    ])
    return float(json.loads(proc.stdout)["format"]["duration"])


def decode_mono(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    """Decode any audio file to a mono float32 array."""
    proc = _run([
        FFMPEG, "-v", "error", "-i", str(path),
        "-f", "f32le", "-ac", "1", "-ar", str(sample_rate), "-",
    ])
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


# --------------------------------------------------------------------------
# Pitch tracking (autocorrelation, monophonic — fine for an isolated vocal)
# --------------------------------------------------------------------------

def track_f0(
    samples: np.ndarray,
    sample_rate: int = 16000,
    hop: int = 160,          # 10 ms
    frame: int = 1024,       # 64 ms
    fmin: float = 65.0,
    fmax: float = 1200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (times, f0_hz) with 0.0 where the frame is unvoiced."""
    if samples.size < frame:
        return np.zeros(0), np.zeros(0)

    min_lag = max(2, int(sample_rate / fmax))
    max_lag = min(frame - 1, int(sample_rate / fmin))
    n_frames = 1 + (samples.size - frame) // hop

    times = np.arange(n_frames) * hop / sample_rate
    f0 = np.zeros(n_frames, dtype=np.float32)

    window = np.hanning(frame).astype(np.float32)
    for i in range(n_frames):
        seg = samples[i * hop: i * hop + frame] * window
        energy = float(np.sqrt(np.mean(seg ** 2)))
        if energy < 1e-3:
            continue
        seg = seg - seg.mean()
        corr = np.correlate(seg, seg, mode="full")[frame - 1:]
        if corr[0] <= 0:
            continue
        corr = corr / corr[0]
        window_slice = corr[min_lag:max_lag]
        if window_slice.size == 0:
            continue
        lag = int(np.argmax(window_slice)) + min_lag
        if corr[lag] < 0.35:      # too weak to trust as a pitch
            continue
        # parabolic interpolation around the peak for sub-sample accuracy
        if 0 < lag < corr.size - 1:
            a, b, c = corr[lag - 1], corr[lag], corr[lag + 1]
            denom = a - 2 * b + c
            if denom != 0:
                lag = lag + 0.5 * (a - c) / denom
        f0[i] = sample_rate / lag
    return times, f0


def _median_filter(x: np.ndarray, k: int = 5) -> np.ndarray:
    if x.size == 0:
        return x
    pad = k // 2
    padded = np.pad(x, pad, mode="edge")
    return np.array([np.median(padded[i:i + k]) for i in range(x.size)])


def hz_to_midi(hz: float) -> float:
    return 69.0 + 12.0 * math.log2(hz / 440.0) if hz > 0 else 0.0


@dataclass
class Note:
    start: float
    end: float
    pitch: int          # MIDI note number
    lyric: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return asdict(self)


def segment_notes(
    times: np.ndarray, f0: np.ndarray, min_duration: float = 0.06
) -> list[Note]:
    """Quantise a pitch track to semitones and merge stable runs into notes."""
    if times.size == 0:
        return []
    midi = np.array([round(hz_to_midi(h)) if h > 0 else 0 for h in _median_filter(f0, 7)])
    notes: list[Note] = []
    start_i = None
    for i, p in enumerate(midi):
        if p == 0:
            if start_i is not None:
                notes.append(Note(float(times[start_i]), float(times[i]), int(midi[start_i])))
                start_i = None
            continue
        if start_i is None:
            start_i = i
        elif midi[i] != midi[start_i]:
            notes.append(Note(float(times[start_i]), float(times[i]), int(midi[start_i])))
            start_i = i
    if start_i is not None:
        notes.append(Note(float(times[start_i]), float(times[-1]), int(midi[start_i])))
    return [n for n in notes if n.duration >= min_duration]


def vocal_to_notes(path: str | Path) -> list[Note]:
    """Isolated vocal file -> note list."""
    samples = decode_mono(path)
    times, f0 = track_f0(samples)
    return segment_notes(times, f0)


# --------------------------------------------------------------------------
# MIDI export — the handoff to Synthesizer V / ACE Studio / VOCALOID
# --------------------------------------------------------------------------

def assign_syllables_to_notes(
    notes: list[Note], line_syllables: list[tuple[float, float, list[str]]]
) -> list[Note]:
    """Place each line's syllables onto the notes that fall inside that line's span.

    If a line has more syllables than notes, the extra syllables split the longest
    notes. If it has fewer, trailing notes become melisma (empty lyric, which the
    singing engines read as a continuation of the previous syllable).
    """
    out: list[Note] = []
    for start, end, syllables in line_syllables:
        span = [n for n in notes if n.start < end and n.end > start]
        if not span:
            continue
        span = sorted(span, key=lambda n: n.start)

        while len(span) < len(syllables):
            longest = max(range(len(span)), key=lambda i: span[i].duration)
            n = span[longest]
            mid = (n.start + n.end) / 2
            span[longest:longest + 1] = [
                Note(n.start, mid, n.pitch),
                Note(mid, n.end, n.pitch),
            ]

        for i, n in enumerate(span):
            n.lyric = syllables[i] if i < len(syllables) else ""
            out.append(n)
    return sorted(out, key=lambda n: n.start)


def write_midi(
    notes: list[Note], path: str | Path, tempo_bpm: float = 120.0, track_name: str = "vocal"
) -> Path:
    """Write a MIDI file with lyric meta events on each note."""
    import mido

    ticks_per_beat = 480
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name=track_name, time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))

    def to_ticks(seconds: float) -> int:
        return int(round(seconds * tempo_bpm / 60.0 * ticks_per_beat))

    events: list[tuple[int, int, str, int]] = []  # (tick, order, kind, pitch)
    lyrics: dict[int, str] = {}
    for n in notes:
        on, off = to_ticks(n.start), max(to_ticks(n.end), to_ticks(n.start) + 1)
        events.append((on, 0, "on", n.pitch))
        events.append((off, 1, "off", n.pitch))
        if n.lyric:
            lyrics.setdefault(on, n.lyric)
    events.sort(key=lambda e: (e[0], e[1]))

    cursor = 0
    for tick, _, kind, pitch in events:
        delta = tick - cursor
        cursor = tick
        if kind == "on" and tick in lyrics:
            track.append(mido.MetaMessage("lyrics", text=lyrics.pop(tick), time=delta))
            delta = 0
        track.append(
            mido.Message("note_on" if kind == "on" else "note_off",
                         note=int(pitch), velocity=80 if kind == "on" else 0, time=delta)
        )

    path = Path(path)
    mid.save(str(path))
    return path


# --------------------------------------------------------------------------
# Mixing
# --------------------------------------------------------------------------

def mix(
    instrumental: str | Path,
    vocal: str | Path,
    output: str | Path,
    vocal_gain_db: float = 0.0,
    instrumental_gain_db: float = 0.0,
) -> Path:
    """Sum a new lead vocal onto the instrumental bed."""
    output = Path(output)
    _run([
        FFMPEG, "-v", "error", "-y",
        "-i", str(instrumental), "-i", str(vocal),
        "-filter_complex",
        f"[0:a]volume={instrumental_gain_db}dB[bed];"
        f"[1:a]volume={vocal_gain_db}dB[lead];"
        f"[bed][lead]amix=inputs=2:duration=longest:normalize=0[out]",
        "-map", "[out]", "-c:a", "pcm_s24le", str(output),
    ])
    return output


def silent_wav(path: str | Path, seconds: float, sample_rate: int = 44100) -> Path:
    """Used by the mock voice provider so the pipeline runs end to end without keys."""
    path = Path(path)
    _run([
        FFMPEG, "-v", "error", "-y", "-f", "lavfi",
        "-i", f"anullsrc=r={sample_rate}:cl=mono",
        "-t", f"{max(0.1, seconds):.3f}", "-c:a", "pcm_s16le", str(path),
    ])
    return path

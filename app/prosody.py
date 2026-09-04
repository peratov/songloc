"""Prosody: the part that decides whether a translated line can actually be sung.

The melody is fixed. A localised line has to land on it: same number of syllable
slots, stresses on the same slots, and ideally the same rhyme shape. Everything
here exists to (a) describe those constraints to the language model and (b) score
what comes back so bad lines get retried instead of shipped.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

try:
    import pyphen  # multi-language hyphenation, used for syllable counts
except ImportError:  # pragma: no cover - pyphen is in requirements
    pyphen = None

# Languages where pyphen's dictionaries are reliable enough to count syllables.
_PYPHEN_LANGS = {
    "en": "en_US", "es": "es", "pt": "pt_BR", "fr": "fr", "de": "de_DE",
    "it": "it_IT", "nl": "nl_NL", "pl": "pl_PL", "ru": "ru_RU", "uk": "uk_UA",
    "sv": "sv", "da": "da_DK", "nb": "nb_NO", "cs": "cs_CZ", "hu": "hu_HU",
    "tr": "tr_TR", "id": "id_ID", "ro": "ro_RO", "el": "el_GR",
}

# Mora-timed / syllable-per-character languages need their own counters.
_MORA_LANGS = {"ja"}
_CHAR_LANGS = {"zh", "zh-cn", "zh-tw", "yue", "ko", "th"}

_SMALL_KANA = set("ぁぃぅぇぉゃゅょゎァィゥェォャュョヮ")
_VOWELS = "aeiouyáéíóúàèìòùâêîôûäëïöüãõåæøœ"

_hyphenators: dict[str, Any] = {}


def _hyphenator(lang: str):
    code = _PYPHEN_LANGS.get(lang.lower().split("-")[0])
    if not code or pyphen is None:
        return None
    if code not in _hyphenators:
        try:
            _hyphenators[code] = pyphen.Pyphen(lang=code)
        except Exception:
            _hyphenators[code] = None
    return _hyphenators[code]


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def words_of(text: str) -> list[str]:
    return [w for w in re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)?", text or "", re.UNICODE) if w]


def _count_en_heuristic(word: str) -> int:
    """Vowel-group counting with the usual English corrections."""
    w = word.lower()
    w = re.sub(r"[^a-z]", "", w)
    if not w:
        return 0
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    # silent terminal 'e' ("make"), but not "the" or "-le" ("table")
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and n > 1:
        n -= 1
    if w.endswith("le") and len(w) > 2 and w[-3] not in _VOWELS:
        n += 1
    # -ed is usually silent unless preceded by t/d ("wanted", "landed")
    if w.endswith("ed") and len(w) > 3 and w[-3] not in "td" and n > 1:
        n -= 1
    return max(1, n)


def _count_mora(text: str) -> int:
    """Japanese mora count: kana minus small kana; 'ー' and 'っ' each count as one."""
    n = 0
    for ch in text:
        if ch in _SMALL_KANA:
            continue
        if unicodedata.category(ch).startswith("L") or ch in "ーッっ":
            n += 1
    return n


def count_syllables(text: str, lang: str = "en") -> int:
    """Syllable (or mora) count for a line or word."""
    text = normalise(text)
    if not text:
        return 0
    base = lang.lower().split("-")[0]

    if base in _MORA_LANGS:
        return _count_mora(re.sub(r"[^\w]", "", text))
    if base in _CHAR_LANGS or lang.lower() in _CHAR_LANGS:
        return len([c for c in text if unicodedata.category(c).startswith("L")])

    hyph = _hyphenator(lang)
    total = 0
    for word in words_of(text):
        if hyph is not None:
            n = len(hyph.inserted(word.lower()).split("-"))
            total += max(1, n)
        elif base == "en":
            total += _count_en_heuristic(word)
        else:
            # Generic Latin-script fallback: count vowel clusters.
            total += max(1, len(re.findall(f"[{_VOWELS}]+", word.lower())))
    return total


def syllabify(word: str, lang: str = "en") -> list[str]:
    """Split a word into syllable-ish chunks. Used for display, not phonology."""
    hyph = _hyphenator(lang)
    if hyph is not None:
        parts = hyph.inserted(word).split("-")
        return [p for p in parts if p] or [word]
    return [word]


def rhyme_key(text: str, lang: str = "en") -> str:
    """Crude terminal-rhyme fingerprint: last vowel cluster onward, lowercased."""
    words = words_of(text)
    if not words:
        return ""
    last = words[-1].lower()
    base = lang.lower().split("-")[0]
    if base in _MORA_LANGS or base in _CHAR_LANGS:
        return last[-1:]
    matches = list(re.finditer(f"[{_VOWELS}]+", last))
    if not matches:
        return last[-2:]
    return last[matches[-1].start():]


def rhymes(a: str, b: str, lang: str = "en") -> bool:
    ka, kb = rhyme_key(a, lang), rhyme_key(b, lang)
    return bool(ka) and ka == kb


# --------------------------------------------------------------------------
# Line templates: what the melody demands of each line
# --------------------------------------------------------------------------

@dataclass
class Word:
    text: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class LineTemplate:
    """A single melodic line, described as constraints a translation must satisfy."""

    index: int
    text: str
    lang: str
    start: float
    end: float
    syllables: int
    strong_positions: list[int] = field(default_factory=list)
    rhyme_group: str | None = None
    section: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def syllables_per_second(self) -> float:
        return self.syllables / self.duration if self.duration else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["duration"] = round(self.duration, 3)
        d["syllables_per_second"] = round(self.syllables_per_second, 2)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LineTemplate":
        """Rebuild from to_dict(), dropping the computed fields."""
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in fields})

    def brief(self) -> str:
        """The one-line constraint string handed to the model."""
        stress = ", ".join(str(p + 1) for p in self.strong_positions) or "none marked"
        rhyme = f"rhyme group {self.rhyme_group}" if self.rhyme_group else "no rhyme constraint"
        return (
            f"L{self.index + 1} | {self.syllables} syllables | "
            f"stress on syllable {stress} | {rhyme} | {self.duration:.2f}s"
        )


def group_words_into_lines(
    words: Iterable[Word], gap_threshold: float = 0.6, max_syllables: int = 18, lang: str = "en"
) -> list[list[Word]]:
    """Split an aligned word stream into lines wherever the singer breathes."""
    lines: list[list[Word]] = []
    current: list[Word] = []
    prev_end: float | None = None
    for w in words:
        gap = (w.start - prev_end) if prev_end is not None else 0.0
        too_long = sum(count_syllables(x.text, lang) for x in current) >= max_syllables
        if current and (gap >= gap_threshold or too_long):
            lines.append(current)
            current = []
        current.append(w)
        prev_end = w.end
    if current:
        lines.append(current)
    return lines


def build_templates(
    words: list[Word], lang: str = "en", gap_threshold: float = 0.6
) -> list[LineTemplate]:
    """Turn aligned words into per-line singability constraints."""
    grouped = group_words_into_lines(words, gap_threshold=gap_threshold, lang=lang)
    templates: list[LineTemplate] = []

    # A syllable held noticeably longer than the song's median is a stressed beat.
    durations: list[float] = []
    for line in grouped:
        for w in line:
            syl = max(1, count_syllables(w.text, lang))
            durations.append(w.duration / syl)
    median = sorted(durations)[len(durations) // 2] if durations else 0.0

    for i, line in enumerate(grouped):
        text = " ".join(w.text for w in line)
        strong: list[int] = []
        slot = 0
        for w in line:
            syl = max(1, count_syllables(w.text, lang))
            per = w.duration / syl if syl else 0.0
            held = bool(median) and per > median * 1.25
            # A word of two or more syllables carries a lexical stress the melody
            # has to accommodate; a held note is a beat the singer leans on.
            if held or syl >= 2:
                strong.append(slot)
            slot += syl
        if 0 not in strong:
            strong.insert(0, 0)  # line openings carry weight in nearly all pop phrasing
        templates.append(
            LineTemplate(
                index=i,
                text=text,
                lang=lang,
                start=line[0].start,
                end=line[-1].end,
                syllables=sum(count_syllables(w.text, lang) for w in line),
                strong_positions=sorted(set(strong)),
            )
        )

    assign_rhyme_groups(templates, lang)
    return templates


def assign_rhyme_groups(templates: list[LineTemplate], lang: str = "en") -> None:
    """Label lines that rhyme with each other (A, B, C ...) so translations can preserve the scheme."""
    keys: dict[str, str] = {}
    next_label = ord("A")
    counts: dict[str, int] = {}
    for t in templates:
        k = rhyme_key(t.text, lang)
        if k:
            counts[k] = counts.get(k, 0) + 1
    for t in templates:
        k = rhyme_key(t.text, lang)
        if not k or counts.get(k, 0) < 2:
            continue  # a rhyme needs a partner
        if k not in keys:
            keys[k] = chr(next_label)
            next_label += 1
        t.rhyme_group = keys[k]


# --------------------------------------------------------------------------
# Scoring: does a candidate line actually fit?
# --------------------------------------------------------------------------

@dataclass
class LineFit:
    index: int
    candidate: str
    target_syllables: int
    actual_syllables: int
    syllable_delta: int
    stress_score: float
    rhyme_ok: bool | None
    density_ratio: float
    fit: float
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fit"] = round(self.fit, 3)
        d["stress_score"] = round(self.stress_score, 3)
        d["density_ratio"] = round(self.density_ratio, 2)
        return d


def _strong_slots_of(candidate: str, lang: str) -> set[int]:
    """Approximate which slots carry stress in the candidate.

    Without a pronunciation dictionary per language, the workable proxy is: the
    first syllable of every word of two or more syllables, plus the first
    syllable of the line. This is deliberately coarse -- it catches lines that
    put a function word on a downbeat, which is the failure that is audible.
    """
    slots: set[int] = set()
    slot = 0
    for w in words_of(candidate):
        syl = max(1, count_syllables(w, lang))
        if syl >= 2 or slot == 0:
            slots.add(slot)
        slot += syl
    return slots


def score_line(
    candidate: str,
    template: LineTemplate,
    lang: str,
    rhyme_partner: str | None = None,
    syllable_tolerance: int = 1,
) -> LineFit:
    actual = count_syllables(candidate, lang)
    delta = actual - template.syllables
    issues: list[str] = []

    # Syllable count: the hard constraint. Off by one is usually singable
    # (elision, a held note split), off by more is not.
    over = max(0, abs(delta) - syllable_tolerance)
    syllable_score = max(0.0, 1.0 - over * 0.34)
    if abs(delta) > syllable_tolerance:
        issues.append(
            f"{abs(delta)} syllable{'s' if abs(delta) != 1 else ''} "
            f"{'too many' if delta > 0 else 'too few'} (needs {template.syllables})"
        )

    # Stress: how many of the melody's strong beats land on a stressed syllable.
    strong = _strong_slots_of(candidate, lang)
    targets = [p for p in template.strong_positions if p < max(actual, 1)]
    if targets:
        hits = sum(1 for p in targets if p in strong or (p - 1) in strong)
        stress_score = hits / len(targets)
    else:
        stress_score = 1.0
    if stress_score < 0.6:
        missed = [p + 1 for p in targets if p not in strong and (p - 1) not in strong]
        issues.append(f"weak syllable on strong beat{'s' if len(missed) != 1 else ''} {missed}")

    # Rhyme, only when the source line had a partner to rhyme with.
    rhyme_ok: bool | None = None
    if template.rhyme_group and rhyme_partner:
        rhyme_ok = rhymes(candidate, rhyme_partner, lang)
        if not rhyme_ok:
            issues.append(f"breaks rhyme group {template.rhyme_group}")

    # Density: syllables per second the singer has to deliver.
    density = template.syllables_per_second
    actual_density = actual / template.duration if template.duration else 0.0
    ratio = actual_density / density if density else 1.0
    if ratio > 1.35:
        issues.append(f"{ratio:.0%} of the original syllable rate — likely unsingable at tempo")

    weights = {"syllables": 0.5, "stress": 0.3, "rhyme": 0.2}
    fit = syllable_score * weights["syllables"] + stress_score * weights["stress"]
    if rhyme_ok is None:
        fit += weights["rhyme"]  # no constraint to violate
    else:
        fit += weights["rhyme"] * (1.0 if rhyme_ok else 0.0)
    if ratio > 1.35:
        fit *= 0.8

    return LineFit(
        index=template.index,
        candidate=candidate,
        target_syllables=template.syllables,
        actual_syllables=actual,
        syllable_delta=delta,
        stress_score=stress_score,
        rhyme_ok=rhyme_ok,
        density_ratio=ratio,
        fit=max(0.0, min(1.0, fit)),
        issues=issues,
    )


def score_lines(
    candidates: list[str], templates: list[LineTemplate], lang: str
) -> list[LineFit]:
    """Score a whole adapted lyric, resolving rhyme partners within the candidate set."""
    by_group: dict[str, int] = {}
    fits: list[LineFit] = []
    for i, template in enumerate(templates):
        candidate = candidates[i] if i < len(candidates) else ""
        partner = None
        if template.rhyme_group:
            first = by_group.get(template.rhyme_group)
            if first is not None:
                partner = candidates[first] if first < len(candidates) else None
            else:
                by_group[template.rhyme_group] = i
        fits.append(score_line(candidate, template, lang, rhyme_partner=partner))
    return fits

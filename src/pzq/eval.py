"""WER / CER without external dependencies.

Two normalizations, deliberately different:

* :func:`normalize` strips *everything* including spaces -- correct for CER on
  scripts that do not delimit words (Chinese, Japanese, Thai).
* :func:`normalize_words` keeps spaces as token boundaries -- correct for WER on
  space-delimited languages.

Mixing them up produces impressively low error rates that mean nothing, so they
are separate functions rather than a flag.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Sequence

_PUNCT = re.compile(r"[\s，。、？！；：,\.\?!;:\"'“”‘’()（）\[\]【】\-—…]+")
_PUNCT_KEEP_SPACE = re.compile(r"[^\w\s]+", re.UNICODE)

# Locales the checkpoint reports as CER rather than WER.
CER_LOCALES = ("zh", "ja", "ko", "th")


def normalize(text: str, keep_case: bool = False) -> str:
    """Strip punctuation and *all* whitespace; keep characters intact.  For CER."""
    text = unicodedata.normalize("NFKC", text)
    if not keep_case:
        text = text.lower()
    return _PUNCT.sub("", text)


def normalize_words(text: str, keep_case: bool = False) -> list[str]:
    """Lowercase, drop punctuation, keep whitespace as token boundaries.  For WER."""
    text = unicodedata.normalize("NFKC", text)
    if not keep_case:
        text = text.lower()
    return _PUNCT_KEEP_SPACE.sub(" ", text).split()


def edit_distance(ref: Sequence, hyp: Sequence) -> int:
    if len(ref) < len(hyp):
        ref, hyp = hyp, ref
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1]


def cer(refs: Iterable[str], hyps: Iterable[str]) -> float:
    """Corpus-level character error rate."""
    total_err = 0
    total_len = 0
    for ref, hyp in zip(refs, hyps):
        r, h = normalize(ref), normalize(hyp)
        total_err += edit_distance(r, h)
        total_len += len(r)
    return total_err / max(total_len, 1)


def wer(refs: Iterable[str], hyps: Iterable[str]) -> float:
    """Corpus-level word error rate, whitespace-delimited."""
    total_err = 0
    total_len = 0
    for ref, hyp in zip(refs, hyps):
        r, h = normalize_words(ref), normalize_words(hyp)
        total_err += edit_distance(r, h)
        total_len += len(r)
    return total_err / max(total_len, 1)


def should_use_cer(locale: str) -> bool:
    """True for locales where whitespace does not delimit words."""
    return locale.lower().split("-")[0] in CER_LOCALES


def score(refs: Iterable[str], hyps: Iterable[str], locale: str = "en") -> dict[str, float]:
    """Return the right metric for ``locale`` (plus both, for inspection)."""
    refs, hyps = list(refs), list(hyps)
    return {
        "cer": cer(refs, hyps),
        "wer": wer(refs, hyps),
        "primary": cer(refs, hyps) if should_use_cer(locale) else wer(refs, hyps),
        "n": float(len(refs)),
    }

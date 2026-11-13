"""Minimal public subset of ``pzq`` — scoring utilities only.

The full package (quantization, bit-plan search, sensitivity analysis, export)
is not part of this release. This subset is what is needed to recompute the
WER/CER numbers published in ``README.md``: text normalisation and edit distance,
nothing model-specific.

Kept dependency-free on purpose: standard library only, so a reviewer can read
all 90 lines in one sitting and verify that the metric is not tuned.
"""

from .eval import cer, edit_distance, normalize, normalize_words, score, should_use_cer, wer

__version__ = "0.1.0-public"

__all__ = [
    "cer",
    "edit_distance",
    "normalize",
    "normalize_words",
    "score",
    "should_use_cer",
    "wer",
]

"""Control arm A: original CombinedDTA architecture, unchanged.

Run this with train_test.py to isolate the effect of the training-side
changes (AdamW weight decay, cosine schedule, label noise, one-sided floor
loss, and the pairwise margin loss).  The network itself is the original
architecture; only the class is renamed so train_test.py can select it.
"""

try:
    from .combined_dta import CombinedDTA
except ImportError:  # pragma: no cover - models dir directly on sys.path
    from combined_dta import CombinedDTA


class CombinedDTAControl(CombinedDTA):
    """Original CombinedDTA architecture (training-side changes only)."""

    pass

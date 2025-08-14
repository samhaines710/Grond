"""Movement-type normalization utilities.

This module provides helper functions for normalizing movement_type
values emitted by the ML classifier into canonical string labels.

Our models may output integers (e.g., 0, 1, 2) or strings ("0", "1",
"2"). Downstream strategy logic expects human‑readable labels
("CALL", "PUT", "NEUTRAL"). Normalizing in one place ensures
consistency across the system and prevents default fallbacks from
being triggered due to mismatched types.
"""

from __future__ import annotations

from typing import Union

__all__ = ["normalize_movement_type"]

# Central map from classifier outputs to strategy labels.  Accepts
# both numeric and string forms.  Unknown inputs are mapped to
# "NEUTRAL".
_MOVEMENT_MAP: dict = {
    0: "CALL",
    1: "PUT",
    2: "NEUTRAL",
    "0": "CALL",
    "1": "PUT",
    "2": "NEUTRAL",
    "CALL": "CALL",
    "PUT": "PUT",
    "NEUTRAL": "NEUTRAL",
}

def normalize_movement_type(x: Union[int, str, None]) -> str:
    """Return a canonical movement_type label.

    Parameters
    ----------
    x : int | str | None
        The raw movement_type produced by the classifier or consumed
        by strategy logic. May be an integer class ID, a string ID,
        or already one of the canonical labels.

    Returns
    -------
    str
        One of "CALL", "PUT", or "NEUTRAL".  Unknown or ``None``
        inputs default to "NEUTRAL".
    """
    if x is None:
        return "NEUTRAL"
    # direct lookup for known keys
    if x in _MOVEMENT_MAP:
        return _MOVEMENT_MAP[x]
    # fallback: normalize to upper‑cased string and lookup again
    s = str(x).strip().upper()
    return _MOVEMENT_MAP.get(s, "NEUTRAL")
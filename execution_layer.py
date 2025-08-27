"""
Execution layer.

Fixes:
- MANUAL EXECUTION logs now include full context (action, movement, probabilities, strategy, source).
- Backward compatible signature: supports both positional (ticker, size, side) and richer kwargs.
- Adds robust probability extraction (accepts list/tuple or dict).
"""

from __future__ import annotations

from typing import Callable, Optional, List, Any, Tuple, Union

from utils.logging_utils import get_logger, write_status

logger = get_logger(__name__)


def _extract_probs(
    probs: Optional[Union[List[float], Tuple[float, float, float], dict]]
) -> Optional[Tuple[float, float, float]]:
    """
    Normalize various probability formats to (p_call, p_put, p_neu).
    Accepts:
      - [p_call, p_put, p_neu]
      - {"call": x, "put": y, "neutral": z} or {"p_call":..., "p_put":..., "p_neutral":...}
    """
    if probs is None:
        return None
    try:
        if isinstance(probs, (list, tuple)) and len(probs) == 3:
            pc, pp, pn = float(probs[0]), float(probs[1]), float(probs[2])
            return pc, pp, pn
        if isinstance(probs, dict):
            # common key variants
            keys = {k.lower(): v for k, v in probs.items()}
            def get_any(*names: str) -> Optional[float]:
                for n in names:
                    if n in keys:
                        return float(keys[n])
                return None
            pc = get_any("call", "p_call", "prob_call")
            pp = get_any("put", "p_put", "prob_put")
            pn = get_any("neutral", "p_neutral", "prob_neutral", "neu")
            if pc is not None and pp is not None and pn is not None:
                return pc, pp, pn
    except Exception:
        return None
    return None


class ManualExecutor:
    def __init__(self, notify_fn: Optional[Callable[[str], None]] = None) -> None:
        self.notify_fn = notify_fn

    def place_order(
        self,
        ticker: str,
        size: float,
        side: str,
        action: Optional[str] = None,
        movement: Optional[str] = None,
        probs: Optional[Any] = None,
        strategy: Optional[str] = None,
        source: Optional[str] = None,
    ) -> None:
        """
        Log and (in a real integration) submit an order.

        Parameters
        ----------
        ticker : str
            Underlying symbol.
        size : float
            Order size (contracts or units).
        side : str
            'buy' or 'sell'.
        action : Optional[str]
            High-level action (e.g., 'BUY_CALL', 'SELL_PUT'). If not provided,
            headline uses side only.
        movement : Optional[str]
            Normalized movement_type ('CALL','PUT','NEUTRAL').
        probs : Optional[Any]
            Either [p_call, p_put, p_neutral] or a dict with those keys.
        strategy : Optional[str]
            Strategy branch chosen ('trend-follow' or 'mean-revert').
        source : Optional[str]
            Greeks source ('polygon' or 'fallback').

        The log message will always start with "MANUAL EXECUTION → ...".
        """
        act = (action or side.upper()).strip()
        if act and act != side.upper():
            headline = f"MANUAL EXECUTION → {act} {size} {ticker}"
        else:
            headline = f"MANUAL EXECUTION → {side.upper()} {size} {ticker}"

        parts: List[str] = []
        if movement:
            parts.append(f"mv={movement}")
        p = _extract_probs(probs)
        if p:
            pc, pp, pn = p
            parts.append(f"p=[C:{pc:.3f},P:{pp:.3f},N:{pn:.3f}]")
        if strategy:
            parts.append(f"strategy={strategy}")
        if source:
            parts.append(f"src={source}")

        msg = headline if not parts else f"{headline} | " + " ".join(parts)

        # Persist to status log and emit to logger for Render
        write_status(msg)
        try:
            logger.info(msg)
        except Exception:
            pass

        if self.notify_fn:
            try:
                self.notify_fn(msg)
            except Exception:
                logger.exception("Error in notify_fn")

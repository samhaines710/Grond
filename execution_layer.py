"""
Execution layer.

Fixes:
- MANUAL EXECUTION logs now include full context (action, movement, probabilities, strategy, source).
- Keeps backward compatibility with existing calls (side-only) but prefers the richer signature.

Public API:
    ManualExecutor(notify_fn: Optional[Callable[[str], None]] = None)
    place_order(ticker: str, size: float, side: str, *,
                action: Optional[str] = None,
                movement: Optional[str] = None,
                probs: Optional[list] = None,
                strategy: Optional[str] = None,
                source: Optional[str] = None) -> None
"""

from __future__ import annotations

from typing import Callable, Optional, List

from utils.logging_utils import get_logger, write_status

logger = get_logger(__name__)

class ManualExecutor:
    def __init__(self, notify_fn: Optional[Callable[[str], None]] = None) -> None:
        self.notify_fn = notify_fn

    def place_order(
        self,
        *,
        ticker: str,
        size: float,
        side: str,
        action: Optional[str] = None,
        movement: Optional[str] = None,
        probs: Optional[List[float]] = None,
        strategy: Optional[str] = None,
        source: Optional[str] = None,
    ) -> None:
        """
        Log and (in a real integration) submit an order.

        Parameters
        ----------
        ticker: str
            The underlying symbol.
        size: float
            Size of the order (notional number of contracts).
        side: str
            'buy' or 'sell'.
        action: Optional[str]
            The high-level action string (e.g., 'BUY_CALL', 'SELL_PUT').
        movement: Optional[str]
            The normalized movement_type ('CALL','PUT','NEUTRAL').
        probs: Optional[List[float]]
            List [p_call,p_put,p_neu] giving classifier probabilities.
        strategy: Optional[str]
            The strategy branch used ('trend-follow','mean-revert', etc.).
        source: Optional[str]
            Data source for Greeks ('polygon' or 'fallback').

        The log message will always start with "MANUAL EXECUTION → ...".
        """
        # Human-readable headline
        act = (action or side.upper()).strip()
        if act and act != side.upper():
            headline = f"MANUAL EXECUTION → {act} {size} {ticker}"
        else:
            headline = f"MANUAL EXECUTION → {side.upper()} {size} {ticker}"

        # Detailed context
        parts = []
        if movement:
            parts.append(f"mv={movement}")
        if probs and len(probs) == 3:
            try:
                pc, pp, pn = float(probs[0]), float(probs[1]), float(probs[2])
                parts.append(f"p=[C:{pc:.3f},P:{pp:.3f},N:{pn:.3f}]")
            except Exception:
                pass
        if strategy:
            parts.append(f"strategy={strategy}")
        if source:
            parts.append(f"src={source}")

        msg = headline if not parts else f"{headline} | " + " ".join(parts)

        # Log to status and optional notifier
        write_status(msg)
        if self.notify_fn:
            try:
                self.notify_fn(msg)
            except Exception:
                logger.exception("Error in notify_fn")

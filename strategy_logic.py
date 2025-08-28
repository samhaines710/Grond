"""
Strategy logic with updated thresholds.

Key changes:
- Separate CALL_THRESHOLD and PUT_THRESHOLD (defaults: 0.60 and 0.55).
- Symmetric edge gating and directionally correct actions.
- PUT signals trigger SELL_PUT when p_put ≥ PUT_THRESHOLD and edge ≥ MIN_MARGIN.

Environment variables (optional):
  CALL_THRESHOLD    — minimum p_call to trigger a call (default: 0.60).
  PUT_THRESHOLD     — minimum p_put to trigger a put (default: 0.55).
  NEUTRAL_MAX       — maximum p_neu before forcing REVIEW (default: 0.58).
  MIN_MARGIN        — minimum margin between top and second-best class (default: 0.08).
  MAX_SPREAD_RATIO  — liquidity filter (unchanged here).
  DELTA_MIN/DELTA_MAX, ENTRY_COST_RATIO, MIN_EV — unchanged in this file.
"""

from __future__ import annotations
import os
from typing import Any, Dict, Tuple

from config import EXIT_PROFIT_TARGET, EXIT_STOP_LOSS

def _envf(key: str, default: str) -> float:
    try:
        return float(os.getenv(key, default))
    except Exception:
        return float(default)

CALL_THRESHOLD   = _envf("CALL_THRESHOLD",   "0.60")
PUT_THRESHOLD    = _envf("PUT_THRESHOLD",    "0.55")
NEUTRAL_MAX      = _envf("NEUTRAL_MAX",      "0.58")
MIN_MARGIN       = _envf("MIN_MARGIN",       "0.08")
MAX_SPREAD_RATIO = _envf("MAX_SPREAD_RATIO", "0.015")
DELTA_MIN        = _envf("DELTA_MIN",        "0.25")
DELTA_MAX        = _envf("DELTA_MAX",        "0.45")
ENTRY_COST_RATIO = _envf("ENTRY_COST_RATIO", "0.025")
MIN_EV           = _envf("MIN_EV",           "0.0")

class StrategyLogic:
    def __init__(self) -> None:
        self.logic_branches = {
            "trend-follow": lambda *_: None,
            "mean-revert":  lambda *_: None,
        }

    def _edge_and_nextbest(self, p_call: float, p_put: float, p_neu: float, mv: str) -> Tuple[float, float]:
        if mv == "CALL":
            nextbest = max(p_put, p_neu)
            return p_call - nextbest, nextbest
        if mv == "PUT":
            nextbest = max(p_call, p_neu)
            return p_put - nextbest, nextbest
        return 0.0, max(p_call, p_put)

    def _choose_branch(self, ctx: Dict[str, Any], mv: str) -> str:
        recent = float(ctx.get("recent_move_pct", 0.0))
        if mv == "CALL":
            return "trend-follow" if recent >= 0.0 else "mean-revert"
        if mv == "PUT":
            return "trend-follow" if recent <= 0.0 else "mean-revert"
        return "mean-revert"

    def _liq_ok(self, ctx: Dict[str, Any]) -> bool:
        sr = float(ctx.get("spread_ratio", 0.0) or ctx.get("avg_spread_ratio", 0.0) or 0.0)
        delta = float(ctx.get("delta", 0.0))
        if sr <= 0.0:
            return False
        if sr > MAX_SPREAD_RATIO:
            return False
        if not (DELTA_MIN <= abs(delta) <= DELTA_MAX):
            return False
        return True

    def _ev_ok(self, mv: str, p_call: float, p_put: float) -> bool:
        target = float(EXIT_PROFIT_TARGET)
        stop   = float(EXIT_STOP_LOSS)
        cost   = float(ENTRY_COST_RATIO)
        if mv == "CALL":
            p = p_call
        elif mv == "PUT":
            p = p_put
        else:
            return False
        ev = p * target + (1.0 - p) * stop - cost
        return ev >= MIN_EV

    def execute_strategy(self, movement_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        probs = context.get("probs", [0.0, 0.0, 0.0])
        try:
            p_call, p_put, p_neu = [float(x) for x in probs]
        except Exception:
            p_call = p_put = p_neu = 0.0

        mv = (movement_type or "NEUTRAL").upper()
        if p_neu > NEUTRAL_MAX or mv == "NEUTRAL":
            return {"action": "REVIEW", "strategy": "neutral", "edge": 0.0}

        edge, _ = self._edge_and_nextbest(p_call, p_put, p_neu, mv)
        branch = self._choose_branch(context, mv)

        # Liquidity check
        if not self._liq_ok(context):
            return {"action": "REVIEW", "strategy": "illiquid", "edge": edge}

        # EV gate
        if not self._ev_ok(mv, p_call, p_put):
            return {"action": "REVIEW", "strategy": "low_ev", "edge": edge}

        if mv == "CALL":
            if p_call >= CALL_THRESHOLD and edge >= MIN_MARGIN and p_call >= p_put:
                return {"action": "BUY_CALL", "strategy": branch, "edge": edge}
            return {"action": "REVIEW", "strategy": "call_blocked", "edge": edge}

        if mv == "PUT":
            if p_put >= PUT_THRESHOLD and edge >= MIN_MARGIN and p_put >= p_call:
                return {"action": "SELL_PUT", "strategy": branch, "edge": edge}
            return {"action": "REVIEW", "strategy": "put_blocked", "edge": edge}

        return {"action": "REVIEW", "strategy": "neutral", "edge": 0.0}

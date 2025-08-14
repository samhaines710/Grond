"""
Central orchestrator for the Grond trading system.

This version:
- Normalizes movement_type so numeric/NumPy IDs become "CALL"/"PUT"/"NEUTRAL".
- GATES ε-greedy exploration to avoid trading when the base signal is NEUTRAL,
  unless explicitly allowed via ALLOW_NEUTRAL_BANDIT.
- Logs normalization and exploration decisions for auditability.
"""

from __future__ import annotations

import os
import logging
import time
from datetime import datetime
from typing import Dict

import numpy as np
from prometheus_client import Counter

from config import (
    TICKERS,
    BANDIT_EPSILON,   # default epsilon (can be overridden by env)
    ORDER_SIZE,
    SERVICE_NAME,
    tz,
)
from monitoring_ops import start_monitoring_server

from data_ingestion import (
    HistoricalDataLoader,
    RealTimeDataStreamer,
    REALTIME_CANDLES,
    REALTIME_LOCK,
)
from pricing_engines import DerivativesPricer
from ml_classifier import MLClassifier
from strategy_logic import StrategyLogic
from signal_generation import BanditAllocator
from execution_layer import ManualExecutor

from utils import (
    write_status,
    reformat_candles,
    calculate_breakout_prob,
    calculate_recent_move_pct,
    calculate_time_of_day,
    calculate_volume_ratio,
    compute_rsi,
    compute_corr_deviation,
    compute_skew_ratio,
    detect_yield_spike,
    fetch_option_greeks,
    append_signal_log,
)
from utils.movement import normalize_movement_type
from utils.messaging import send_telegram

# ─── Prometheus metrics ────────────────────────────────────────────────────────
SIGNALS_PROCESSED = Counter(
    f"{SERVICE_NAME}_signals_total",
    "Number of signals generated",
    ["ticker", "movement_type"],
)
EXECUTIONS = Counter(
    f"{SERVICE_NAME}_executions_total",
    "Number of executions placed",
    ["ticker", "action"],
)

# ─── Exploration controls (env-overridable) ───────────────────────────────────
# Effective epsilon may be overridden at runtime without code changes.
_EPSILON = float(os.getenv("BANDIT_EPSILON", str(BANDIT_EPSILON)))
# Do we allow exploration to escalate NEUTRAL into CALL/PUT?
_ALLOW_NEUTRAL_BANDIT = os.getenv("ALLOW_NEUTRAL_BANDIT", "0").lower() in {"1", "true", "yes"}

class GrondOrchestrator:
    """Central orchestrator for data ingestion, signal generation, and execution."""

    def __init__(self) -> None:
        logging.getLogger().setLevel(logging.INFO)
        write_status("Starting monitoring server…")
        start_monitoring_server()

        # Data feeders
        self.hist_loader = HistoricalDataLoader()
        self.rt_stream = RealTimeDataStreamer()
        self.rt_stream.start()

        # Core engines
        self.pricer = DerivativesPricer()
        self.classifier = MLClassifier()
        self.logic = StrategyLogic()
        self.bandit = BanditAllocator(
            list(self.logic.logic_branches.keys()),
            epsilon=_EPSILON,
        )

        # Manual executor uses Telegram for notifications
        self.executor = ManualExecutor(notify_fn=send_telegram)

        write_status(
            f"GrondOrchestrator initialized (epsilon={_EPSILON}, "
            f"allow_neutral_bandit={_ALLOW_NEUTRAL_BANDIT})"
        )

    def _decide_movement(self, base_mv: str) -> tuple[str, bool]:
        """Return (final_mv, explored_flag) with exploration gating.

        - Only explore away from CALL/PUT by default.
        - NEUTRAL stays NEUTRAL unless ALLOW_NEUTRAL_BANDIT is enabled.
        """
        explored = False
        mv = base_mv

        roll = np.random.rand()
        if base_mv in ("CALL", "PUT"):
            if roll < _EPSILON:
                cand = self.bandit.select_arm()
                explored = (cand != base_mv)
                mv = cand
        elif base_mv == "NEUTRAL":
            if _ALLOW_NEUTRAL_BANDIT and roll < _EPSILON:
                cand = self.bandit.select_arm()
                # Only escalate to an actionable arm
                mv = "CALL" if cand == "CALL" else ("PUT" if cand == "PUT" else "NEUTRAL")
                explored = (mv != base_mv)
            else:
                mv = "NEUTRAL"
                explored = False
        else:
            # Unknown labels should never occur after normalization,
            # but clamp to NEUTRAL if they do.
            mv = "NEUTRAL"
            explored = False

        return mv, explored

    def run(self) -> None:
        """Main loop that processes new bars and acts on generated signals."""
        write_status("Entering main loop.")
        while True:
            now = datetime.now(tz)
            for ticker in TICKERS:
                # Pull current 5-minute bars
                with REALTIME_LOCK:
                    raw = list(REALTIME_CANDLES.get(ticker, []))
                if len(raw) < 5:
                    continue

                bars = reformat_candles(raw)

                # Feature calculation
                breakout   = calculate_breakout_prob(bars)
                recent_pct = calculate_recent_move_pct(bars)
                vol_ratio  = calculate_volume_ratio(bars)
                rsi_val    = compute_rsi(bars)
                corr_dev   = compute_corr_deviation(ticker)
                skew       = compute_skew_ratio(ticker)
                ys2        = detect_yield_spike("2year")
                ys10       = detect_yield_spike("10year")
                ys30       = detect_yield_spike("30year")
                tod        = calculate_time_of_day(now)
                greeks     = fetch_option_greeks(ticker)

                theta_raw = float(greeks.get("theta", 0.0))
                theta_day = theta_raw
                theta_5m  = theta_day / 78.0  # ~78 five-minute bars in regular session

                features: Dict[str, float | str] = {
                    "breakout_prob":      breakout,
                    "recent_move_pct":    recent_pct,
                    "volume_ratio":       vol_ratio,
                    "rsi":                rsi_val,
                    "corr_dev":           corr_dev,
                    "skew_ratio":         skew,
                    "yield_spike_2year":  ys2,
                    "yield_spike_10year": ys10,
                    "yield_spike_30year": ys30,
                    "time_of_day":        tod,
                    "delta":  float(greeks.get("delta", 0.0)),
                    "gamma":  float(greeks.get("gamma", 0.0)),
                    "theta":  theta_raw,
                    "vega":   float(greeks.get("vega", 0.0)),
                    "rho":    float(greeks.get("rho", 0.0)),
                    "vanna":  float(greeks.get("vanna", 0.0)),
                    "vomma":  float(greeks.get("vomma", 0.0)),
                    "charm":  float(greeks.get("charm", 0.0)),
                    "veta":   float(greeks.get("veta", 0.0)),
                    "speed":  float(greeks.get("speed", 0.0)),
                    "zomma":  float(greeks.get("zomma", 0.0)),
                    "color":  float(greeks.get("color", 0.0)),
                    "implied_volatility": float(greeks.get("implied_volatility", 0.0)),
                    "theta_day": theta_day,
                    "theta_5m":  theta_5m,
                }

                # Classification
                cls_out = self.classifier.classify(features)
                raw_mv = cls_out.get("movement_type")
                base_mv = normalize_movement_type(raw_mv)
                if raw_mv != base_mv:
                    write_status(f"Normalized movement_type {raw_mv!r} → {base_mv}")

                # Exploration with gating
                mv, explored = self._decide_movement(base_mv)
                if explored:
                    write_status(f"Exploration override: base={base_mv} → chosen={mv}")
                # Propagate normalized/final label for downstream consumers
                cls_out["movement_type"] = mv

                context = {**features, **cls_out}
                strat = self.logic.execute_strategy(mv, context)
                SIGNALS_PROCESSED.labels(ticker=ticker, movement_type=mv).inc()

                action = strat["action"]
                if action not in ("AVOID", "REVIEW"):
                    side = "buy" if action.endswith("_CALL") else "sell"
                    self.executor.place_order(
                        ticker=ticker,
                        size=ORDER_SIZE,
                        side=side,
                    )
                    EXECUTIONS.labels(ticker=ticker, action=action).inc()

                    append_signal_log({
                        "time": now.isoformat(),
                        "ticker": ticker,
                        "movement_type": mv,
                        "action": action,
                        **greeks,
                    })

            # Sleep until next 5-minute bar
            time.sleep(300)

if __name__ == "__main__":
    write_status("Launching Grond Orchestrator…")
    orchestrator = GrondOrchestrator()
    orchestrator.run()

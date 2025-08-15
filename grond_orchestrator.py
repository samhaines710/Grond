"""
Central orchestrator for the Grond trading system.

This version:
- Enforces a SINGLETON lock at startup so only one instance can run.
- Normalizes movement_type so numeric/NumPy IDs become "CALL"/"PUT"/"NEUTRAL".
- GATES ε-greedy exploration so NEUTRAL does not escalate to a trade
  unless ALLOW_NEUTRAL_BANDIT is explicitly enabled.
- Adds a SAFETY CLAMP so no execution can occur on NEUTRAL when the
  gate is off, even if some other path misbehaves.
- Logs normalization and exploration decisions for auditability.
"""

from __future__ import annotations

import os
import fcntl
import logging
import time
from datetime import datetime
from typing import Dict, Tuple

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
_EPSILON = float(os.getenv("BANDIT_EPSILON", str(BANDIT_EPSILON)))
_ALLOW_NEUTRAL_BANDIT = os.getenv("ALLOW_NEUTRAL_BANDIT", "0").lower() in {"1", "true", "yes"}

# ─── Singleton lock config ────────────────────────────────────────────────────
_LOCK_FILE = os.getenv("GROND_LOCK_FILE", "/tmp/grond_orchestrator.lock")
_INSTANCE_ID = os.getenv("INSTANCE_ID", f"pid-{os.getpid()}")

class GrondOrchestrator:
    """Central orchestrator for data ingestion, signal generation, and execution."""

    def __init__(self) -> None:
        logging.getLogger().setLevel(logging.INFO)

        # Acquire singleton lock (non-blocking). If locked, exit immediately.
        try:
            self._lock_fh = open(_LOCK_FILE, "w")
            fcntl.flock(self._lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._lock_fh.truncate(0)
            self._lock_fh.write(_INSTANCE_ID)
            self._lock_fh.flush()
            write_status(f"[{_INSTANCE_ID}] Acquired singleton lock: {_LOCK_FILE}")
        except BlockingIOError:
            write_status(f"[{_INSTANCE_ID}] Another instance holds {_LOCK_FILE}. Exiting.")
            raise SystemExit(0)

        write_status(f"[{_INSTANCE_ID}] Starting monitoring server…")
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
            f"[{_INSTANCE_ID}] GrondOrchestrator initialized "
            f"(epsilon={_EPSILON}, allow_neutral_bandit={_ALLOW_NEUTRAL_BANDIT})"
        )

    def _decide_movement(self, base_mv: str) -> Tuple[str, bool]:
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
                mv = "CALL" if cand == "CALL" else ("PUT" if cand == "PUT" else "NEUTRAL")
                explored = (mv != base_mv)
            else:
                mv = "NEUTRAL"
                explored = False
        else:
            mv = "NEUTRAL"
            explored = False

        return mv, explored

    def run(self) -> None:
        """Main loop that processes new bars and acts on generated signals."""
        write_status(f"[{_INSTANCE_ID}] Entering main loop.")
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
                    write_status(f"[{_INSTANCE_ID}] Normalized movement_type {raw_mv!r} → {base_mv}")

                # Exploration with gating
                mv, explored = self._decide_movement(base_mv)
                if explored:
                    write_status(f"[{_INSTANCE_ID}] Exploration override: base={base_mv} → chosen={mv}")

                # Propagate final label
                cls_out["movement_type"] = mv
                context = {**features, **cls_out}

                # Strategy decision
                strat = self.logic.execute_strategy(mv, context)
                SIGNALS_PROCESSED.labels(ticker=ticker, movement_type=mv).inc()

                # SAFETY CLAMP: if NEUTRAL and neutral-bandit is disabled, never execute
                if mv == "NEUTRAL" and not _ALLOW_NEUTRAL_BANDIT:
                    if strat.get("action") not in ("AVOID", "REVIEW"):
                        write_status(f"[{_INSTANCE_ID}] Safety clamp: preventing execution on NEUTRAL (ticker={ticker})")
                    action = "REVIEW"
                else:
                    action = strat.get("action", "REVIEW")

                if action not in ("AVOID", "REVIEW"):
                    side = "buy" if action.endswith("_CALL") else "sell"
                    self.executor.place_order(ticker=ticker, size=ORDER_SIZE, side=side)
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
    write_status(f"[{_INSTANCE_ID}] Launching Grond Orchestrator…")
    orchestrator = GrondOrchestrator()
    orchestrator.run()

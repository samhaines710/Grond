"""Central orchestrator for the Grond trading system.

This module coordinates real-time data streaming, feature engineering,
classification, bandit exploration, strategy execution, and order placement.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Dict

import numpy as np
from prometheus_client import Counter

from config import (
    TICKERS,
    BANDIT_EPSILON,
    ORDER_SIZE,
    SERVICE_NAME,
    tz,
)
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
from monitoring_ops import start_monitoring_server
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

# ─── Prometheus metrics ──────────────────────────────────────────────────────
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
            epsilon=BANDIT_EPSILON,
        )
        self.executor = ManualExecutor(notify_fn=write_status)

        write_status("GrondOrchestrator initialized.")

    def run(self) -> None:
        """Main loop that processes new bars and acts on generated signals."""
        write_status("Entering main loop.")
        while True:
            now = datetime.now(tz)
            for ticker in TICKERS:
                # Pull current 5‑minute bars
                with REALTIME_LOCK:
                    raw = list(REALTIME_CANDLES.get(ticker, []))
                if len(raw) < 5:
                    continue

                bars = reformat_candles(raw)

                # Feature calculation
                breakout = calculate_breakout_prob(bars)
                recent_pct = calculate_recent_move_pct(ticker, bars)
                vol_ratio = calculate_volume_ratio(ticker, bars)
                rsi_val = compute_rsi(bars)
                corr_dev = compute_corr_deviation(ticker)
                skew = compute_skew_ratio(ticker)
                ys2 = detect_yield_spike("2year")
                ys10 = detect_yield_spike("10year")
                ys30 = detect_yield_spike("30year")
                tod = calculate_time_of_day(now)
                greeks = fetch_option_greeks(ticker)

                features: Dict[str, float] = {
                    **greeks,
                    "breakout_prob": breakout,
                    "recent_move_pct": recent_pct,
                    "volume_ratio": vol_ratio,
                    "rsi": rsi_val,
                    "corr_dev": corr_dev,
                    "skew_ratio": skew,
                    "yield_spike_2year": ys2,
                    "yield_spike_10year": ys10,
                    "yield_spike_30year": ys30,
                    "time_of_day": tod,
                }

                # Classification + ε–greedy exploration
                cls_out = self.classifier.classify(features)
                base_mv = cls_out["movement_type"]
                exploration = np.random.rand() < BANDIT_EPSILON
                mv = self.bandit.select_arm() if exploration else base_mv

                context = {**features, **cls_out}
                strat = self.logic.execute_strategy(mv, context)
                SIGNALS_PROCESSED.labels(
                    ticker=ticker, movement_type=mv
                ).inc()

                action = strat["action"]
                if action not in ("AVOID", "REVIEW"):
                    side = "buy" if action.endswith("_CALL") else "sell"
                    self.executor.place_order(
                        ticker=ticker,
                        size=ORDER_SIZE,
                        side=side,
                    )
                    EXECUTIONS.labels(
                        ticker=ticker,
                        action=action,
                    ).inc()

                    # Log to CSV
                    append_signal_log(
                        {
                            "time": now.isoformat(),
                            "ticker": ticker,
                            "movement_type": mv,
                            "action": action,
                            **greeks,
                        }
                    )

            # Sleep until next 5‑minute bar
            time.sleep(300)


if __name__ == "__main__":
    write_status("Launching Grond Orchestrator…")
    orchestrator = GrondOrchestrator()
    orchestrator.run()

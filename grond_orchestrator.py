import time
import logging
from datetime import datetime
import numpy as np
import pytz

from config import (
    TICKERS,
    BANDIT_EPSILON,
    ORDER_SIZE,
    SERVICE_NAME,
    tz
)
from monitoring_ops import start_monitoring_server
from prometheus_client import Counter

from data_ingestion import (
    HistoricalDataLoader,
    RealTimeDataStreamer,
    REALTIME_CANDLES,
    REALTIME_LOCK
)
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
    fetch_option_greeks
)
from pricing_engines import DerivativesPricer
from movement_classifier import MovementClassifier
from strategy_logic import StrategyLogic
from signal_generation import BanditAllocator
from execution_layer import ManualExecutor

# ── Metrics ────────────────────────────────────────────────────────────────────
SIGNALS_PROCESSED = Counter(
    f"{SERVICE_NAME}_signals_total",
    "Number of signals generated", ["ticker", "movement_type"]
)
EXECUTIONS = Counter(
    f"{SERVICE_NAME}_executions_total",
    "Number of executions placed", ["ticker", "action"]
)

# ── Orchestrator ──────────────────────────────────────────────────────────────
class GrondOrchestrator:
    def __init__(self):
        # Start Prometheus metrics server
        start_monitoring_server()

        # Historical loader (for future use, e.g. static backfills)
        self.hist_loader = HistoricalDataLoader()

        # RT data streamer
        self.rt_stream = RealTimeDataStreamer()
        self.rt_stream.start()

        # Analytics
        self.pricer     = DerivativesPricer()  # uses QuantLib by default
        self.classifier = MovementClassifier()
        self.logic      = StrategyLogic()
        self.bandit     = BanditAllocator(
            list(self.logic.logic_branches.keys()),
            epsilon=BANDIT_EPSILON
        )

        # Manual executor with Telegram notifications via write_status
        self.executor = ManualExecutor(notify_fn=lambda msg: write_status(msg))

    def run(self):
        """
        Main 5-minute loop: generate signals and send manual execution prompts.
        """
        while True:
            now = datetime.now(tz)

            for ticker in TICKERS:
                # Grab latest in-memory 5m bars
                with REALTIME_LOCK:
                    raw = list(REALTIME_CANDLES.get(ticker, []))
                if len(raw) < 5:
                    continue

                # Feature engineering
                bars        = reformat_candles(raw)
                breakout    = calculate_breakout_prob(bars)
                move_pct    = calculate_recent_move_pct(ticker, bars)
                vol_ratio   = calculate_volume_ratio(ticker, bars)
                rsi         = compute_rsi(bars)
                corr_dev    = compute_corr_deviation(ticker)
                skew_ratio  = compute_skew_ratio(ticker)
                ys2         = detect_yield_spike("2year")
                ys10        = detect_yield_spike("10year")
                ys30        = detect_yield_spike("30year")
                time_of_day = calculate_time_of_day(now)

                greeks = fetch_option_greeks(ticker)

                # Fair price via QuantLib Black-Scholes
                model_price = self.pricer.price_black_scholes(
                    spot     = greeks.get("underlying_price", 0.0),
                    strike   = greeks.get("strike", 0.0),
                    vol      = greeks.get("implied_volatility", 0.0),
                    maturity = greeks.get("time_to_expiry", 0.0),
                    rate     = greeks.get("risk_free_rate", 0.0),
                    dividend = greeks.get("dividend_yield", 0.0),
                    option_type = "call"
                )

                features = {
                    **greeks,
                    "breakout_prob":        breakout,
                    "recent_move_pct":      move_pct,
                    "volume_ratio":         vol_ratio,
                    "rsi":                  rsi,
                    "corr_dev":             corr_dev,
                    "skew_ratio":           skew_ratio,
                    "yield_spike_2year":    ys2,
                    "yield_spike_10year":   ys10,
                    "yield_spike_30year":   ys30,
                    "time_of_day":          time_of_day,
                    "model_price":          model_price,
                    "price_diff":           greeks.get("underlying_price", 0.0) - model_price
                }

                # Movement classification
                base_mv = self.classifier.classify(features)["movement_type"]

                # Epsilon-greedy bandit
                if np.random.rand() < BANDIT_EPSILON:
                    mv = self.bandit.select_arm()
                else:
                    mv = base_mv

                # Strategy logic
                strat = self.logic.execute_strategy(mv, features)

                # Metric: signal generated
                SIGNALS_PROCESSED.labels(ticker=ticker, movement_type=mv).inc()

                action = strat["action"]
                if action not in ("AVOID", "REVIEW"):
                    # Place manual execution order
                    side = "buy" if action.endswith("_CALL") else "sell"
                    report = self.executor.place_order(
                        ticker=ticker,
                        size=ORDER_SIZE,
                        side=side
                    )
                    # Metric: execution placed
                    EXECUTIONS.labels(ticker=ticker, action=action).inc()

            # Sleep until next 5-minute bar
            time.sleep(300)


if __name__ == "__main__":
    logging.getLogger().info("Starting Grond Orchestrator…")
    GrondOrchestrator().run()

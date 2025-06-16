# grond_orchestrator.py

import time
import logging
from datetime import datetime
import numpy as np

from config import TICKERS, BANDIT_EPSILON, ORDER_SIZE, SERVICE_NAME, tz
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
from ml_classifier import MLClassifier
from strategy_logic import StrategyLogic
from signal_generation import BanditAllocator
from execution_layer import ManualExecutor

# Prometheus metrics
SIGNALS_PROCESSED = Counter(
    f"{SERVICE_NAME}_signals_total",
    "Number of signals generated",
    ["ticker", "movement_type"]
)
EXECUTIONS = Counter(
    f"{SERVICE_NAME}_executions_total",
    "Number of executions placed",
    ["ticker", "action"]
)

class GrondOrchestrator:
    def __init__(self):
        logging.getLogger().setLevel(logging.INFO)
        start_monitoring_server()
        self.hist_loader = HistoricalDataLoader()
        self.rt_stream   = RealTimeDataStreamer()
        self.rt_stream.start()

        self.pricer     = DerivativesPricer()
        self.classifier = MLClassifier()
        self.logic      = StrategyLogic()
        self.bandit     = BanditAllocator(
            list(self.logic.logic_branches.keys()),
            epsilon=BANDIT_EPSILON
        )
        self.executor   = ManualExecutor(notify_fn=lambda msg: write_status(msg))

    def run(self):
        while True:
            now = datetime.now(tz)

            for ticker in TICKERS:
                with REALTIME_LOCK:
                    raw = list(REALTIME_CANDLES.get(ticker, []))
                if len(raw) < 5:
                    continue

                bars     = reformat_candles(raw)
                greeks   = fetch_option_greeks(ticker)
                features = {
                    **greeks,
                    "breakout_prob":      calculate_breakout_prob(bars),
                    "recent_move_pct":    calculate_recent_move_pct(ticker, bars),
                    "volume_ratio":       calculate_volume_ratio(ticker, bars),
                    "rsi":                compute_rsi(bars),
                    "corr_dev":           compute_corr_deviation(ticker),
                    "skew_ratio":         compute_skew_ratio(ticker),
                    "yield_spike_2year":  detect_yield_spike("2year"),
                    "yield_spike_10year": detect_yield_spike("10year"),
                    "yield_spike_30year": detect_yield_spike("30year"),
                    "time_of_day":        calculate_time_of_day(now),
                }

                cls_out = self.classifier.classify(features)
                base_mv = cls_out["movement_type"]

                if np.random.rand() < BANDIT_EPSILON:
                    mv = self.bandit.select_arm()
                else:
                    mv = base_mv

                context = {**features, **cls_out}
                strat   = self.logic.execute_strategy(mv, context)
                SIGNALS_PROCESSED.labels(ticker=ticker, movement_type=mv).inc()

                action = strat["action"]
                if action not in ("AVOID", "REVIEW"):
                    side = "buy" if action.endswith("_CALL") else "sell"
                    self.executor.place_order(
                        ticker=ticker,
                        size=ORDER_SIZE,
                        side=side
                    )
                    EXECUTIONS.labels(ticker=ticker, action=action).inc()

            time.sleep(300)


if __name__ == "__main__":
    logging.getLogger().info("Starting Grond Orchestrator…")
    GrondOrchestrator().run()

# grond_orchestrator.py

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
from pricing_engines import DerivativesPricer
from ml_classifier      import MLClassifier
from strategy_logic     import StrategyLogic
from signal_generation  import BanditAllocator
from execution_layer    import ManualExecutor

from utils.logging_utils   import write_status
from utils.calendar_utils  import calculate_time_of_day
from utils.file_io         import append_signal_log
from utils.http_client     import rate_limited
from utils.greeks_helpers  import calculate_breakout_prob, calculate_recent_move_pct
from utils.http_client     import safe_fetch_polygon_data   # if needed for on-the-fly calls
from utils.statistics      import (  # you can pull these into a tiny stats module
    calculate_volume_ratio,
    compute_rsi,
    compute_corr_deviation,
    compute_skew_ratio,
    detect_yield_spike
)
from utils.pricing_helpers import fetch_option_greeks  # or keep it in greeks_helpers

# ─── Prometheus metrics ─────────────────────────────────────────────────────────
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
        write_status("Starting monitoring server…")
        start_monitoring_server()

        # Data feeders
        self.hist_loader = HistoricalDataLoader()
        self.rt_stream   = RealTimeDataStreamer()
        self.rt_stream.start()

        # Core engines
        self.pricer     = DerivativesPricer()
        self.classifier = MLClassifier()
        self.logic      = StrategyLogic()
        self.bandit     = BanditAllocator(
            list(self.logic.logic_branches.keys()),
            epsilon=BANDIT_EPSILON
        )
        self.executor   = ManualExecutor(notify_fn=write_status)

        write_status("GrondOrchestrator initialized.")

    def run(self):
        write_status("Entering main loop.")
        while True:
            now = datetime.now(tz)

            for ticker in TICKERS:
                # Pull current 5min bars from the shared deque
                with REALTIME_LOCK:
                    bars = list(REALTIME_CANDLES.get(ticker, []))
                if len(bars) < 5:
                    continue

                # Feature calculation
                rets = [b for b in bars]  # already OHLCV dicts
                breakout   = calculate_breakout_prob(rets)
                recent_pct = calculate_recent_move_pct(ticker, rets)
                vol_ratio  = calculate_volume_ratio(ticker, rets)
                rsi        = compute_rsi(rets)
                corr_dev   = compute_corr_deviation(ticker)
                skew       = compute_skew_ratio(ticker)
                ys2        = detect_yield_spike("2year")
                ys10       = detect_yield_spike("10year")
                ys30       = detect_yield_spike("30year")
                tod        = calculate_time_of_day(now)
                greeks     = fetch_option_greeks(ticker)

                features = {
                    **greeks,
                    "breakout_prob":      breakout,
                    "recent_move_pct":    recent_pct,
                    "volume_ratio":       vol_ratio,
                    "rsi":                rsi,
                    "corr_dev":           corr_dev,
                    "skew_ratio":         skew,
                    "yield_spike_2year":  ys2,
                    "yield_spike_10year": ys10,
                    "yield_spike_30year": ys30,
                    "time_of_day":        tod,
                }

                # Classification + exploration
                cls_out  = self.classifier.classify(features)
                base_mv  = cls_out["movement_type"]
                mv       = (
                    self.bandit.select_arm()
                    if np.random.rand() < BANDIT_EPSILON
                    else base_mv
                )

                # Strategy & execution
                context = {**features, **cls_out}
                strat   = self.logic.execute_strategy(mv, context)
                SIGNALS_PROCESSED.labels(
                    ticker=ticker, movement_type=mv
                ).inc()

                action = strat["action"]
                if action not in ("AVOID", "REVIEW"):
                    side = "buy" if action.endswith("_CALL") else "sell"
                    self.executor.place_order(
                        ticker=ticker,
                        size=ORDER_SIZE,
                        side=side
                    )
                    EXECUTIONS.labels(
                        ticker=ticker,
                        action=action
                    ).inc()

                    # Optionally log to CSV
                    append_signal_log({
                        "time": now.isoformat(),
                        "ticker": ticker,
                        "movement_type": mv,
                        "action": action,
                        **greeks
                    })

            # Sleep to align with 5m bars
            time.sleep(300)


if __name__ == "__main__":
    write_status("Launching Grond Orchestrator…")
    GrondOrchestrator().run()

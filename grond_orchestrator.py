"""
Central orchestrator (original Grond) with log message fix.

Changes:
- Pass action, movement_type, probabilities, strategy, and source into ManualExecutor.place_order.
- No other functional changes to exploration, caps, or strategy logic.
"""

from __future__ import annotations

import os
import fcntl
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
from prometheus_client import Counter

from config import (
    TICKERS,
    BANDIT_EPSILON,
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

from utils.logging_utils import configure_logging, write_status
from utils import (
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

configure_logging()

# Prometheus counters
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

# Exploration settings
_EPSILON = float(os.getenv("BANDIT_EPSILON", str(BANDIT_EPSILON)))
_ALLOW_NEUTRAL_BANDIT = os.getenv("ALLOW_NEUTRAL_BANDIT", "0").lower() in {"1", "true", "yes"}

# Budget & dedup
_MAX_TRADES = int(os.getenv("MAX_TRADES_PER_CYCLE", "3"))
_MAX_BUYS   = int(os.getenv("MAX_BUYS_PER_CYCLE", "2"))
_MAX_SELLS  = int(os.getenv("MAX_SELLS_PER_CYCLE", "2"))
_MIN_EXEC_INTERVAL = float(os.getenv("MIN_EXEC_INTERVAL", "0.5"))

# Singleton lock
_LOCK_FILE = os.getenv("GROND_LOCK_FILE", "/tmp/grond_orchestrator.lock")
_INSTANCE_ID = os.getenv("INSTANCE_ID", f"pid-{os.getpid()}")


def _map_order_side(action: str) -> str:
    """Infer side (buy/sell) from action string."""
    A = (action or "").upper()
    if A.startswith("BUY_"):
        return "buy"
    if A.startswith("SELL_"):
        return "sell"
    if A.endswith("_CALL"):
        return "buy"
    if A.endswith("_PUT"):
        return "sell"
    return "buy"


class GrondOrchestrator:
    def __init__(self) -> None:
        logging.getLogger().setLevel(logging.INFO)

        # Acquire singleton
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

        # Data & engines
        self.hist_loader = HistoricalDataLoader()
        self.rt_stream = RealTimeDataStreamer()
        self.rt_stream.start()

        self.pricer = DerivativesPricer()
        self.classifier = MLClassifier()
        self.logic = StrategyLogic()
        self.bandit = BanditAllocator(
            list(self.logic.logic_branches.keys()),
            epsilon=_EPSILON,
        )
        self.executor = ManualExecutor(notify_fn=send_telegram)

        # Dedup cache
        self._last_exec: Dict[str, Tuple[str, float]] = {}

        write_status(
            f"[{_INSTANCE_ID}] Orchestrator ready (epsilon={_EPSILON}, "
            f"allow_neutral_bandit={_ALLOW_NEUTRAL_BANDIT})"
        )

    def _decide_movement(self, base_mv: str) -> Tuple[str, bool]:
        """Apply epsilon exploration to base movement_type (policy unchanged)."""
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
        else:
            mv = "NEUTRAL"
        return mv, explored

    def _exec_dedup_ok(self, ticker: str, action: str) -> bool:
        """Prevent log/order spam when the same action is repeated too quickly."""
        now = time.monotonic()
        last = self._last_exec.get(ticker)
        if last:
            last_action, last_ts = last
            if action == last_action and (now - last_ts) < _MIN_EXEC_INTERVAL:
                return False
        self._last_exec[ticker] = (action, now)
        return True

    def run(self) -> None:
        write_status(f"[{_INSTANCE_ID}] Entering main loop.")
        while True:
            now = datetime.now(tz)
            candidates: List[Dict[str, Any]] = []

            for ticker in TICKERS:
                # Pull recent realtime candles
                with REALTIME_LOCK:
                    raw = list(REALTIME_CANDLES.get(ticker, []))
                if len(raw) < 5:
                    continue
                bars = reformat_candles(raw)

                # --- Feature build (unchanged) ---
                breakout = calculate_breakout_prob(bars)
                recent_pct = calculate_recent_move_pct(bars)
                vol_ratio = calculate_volume_ratio(bars)
                rsi_val = compute_rsi(bars)
                corr_dev = compute_corr_deviation(ticker)
                skew = compute_skew_ratio(ticker)
                ys2 = detect_yield_spike("2year")
                ys10 = detect_yield_spike("10year")
                ys30 = detect_yield_spike("30year")
                tod = calculate_time_of_day(now)
                greeks = fetch_option_greeks(ticker)

                theta_raw = float(greeks.get("theta", 0.0))
                theta_day = theta_raw
                theta_5m = theta_day / 78.0
                source_val = str(greeks.get("source", "fallback"))

                features = {
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
                    "delta": float(greeks.get("delta", 0.0)),
                    "gamma": float(greeks.get("gamma", 0.0)),
                    "theta": theta_raw,
                    "vega": float(greeks.get("vega", 0.0)),
                    "rho": float(greeks.get("rho", 0.0)),
                    "vanna": float(greeks.get("vanna", 0.0)),
                    "vomma": float(greeks.get("vomma", 0.0)),
                    "charm": float(greeks.get("charm", 0.0)),
                    "veta": float(greeks.get("veta", 0.0)),
                    "speed": float(greeks.get("speed", 0.0)),
                    "zomma": float(greeks.get("zomma", 0.0)),
                    "color": float(greeks.get("color", 0.0)),
                    "implied_volatility": float(greeks.get("implied_volatility", 0.0)),
                    "theta_day": theta_day,
                    "theta_5m": theta_5m,
                    "source": source_val,
                }

                # Classify and normalize movement type
                cls_out = self.classifier.classify(features)
                base_mv = normalize_movement_type(cls_out.get("movement_type"))
                probs = cls_out.get("probs", [0.0, 0.0, 0.0])

                # Epsilon exploration
                mv, explored = self._decide_movement(base_mv)
                if explored:
                    write_status(f"[{_INSTANCE_ID}] Exploration override: base={base_mv} → chosen={mv}")

                # Respect neutral gating policy
                if mv == "NEUTRAL" and not _ALLOW_NEUTRAL_BANDIT:
                    SIGNALS_PROCESSED.labels(ticker=ticker, movement_type=mv).inc()
                    continue

                # Strategy decision
                ctx = {**features, **cls_out, "movement_type": mv}
                strat = self.logic.execute_strategy(mv, ctx)
                SIGNALS_PROCESSED.labels(ticker=ticker, movement_type=mv).inc()

                action = strat.get("action", "REVIEW")
                edge   = float(strat.get("edge", 0.0))
                strategy_name = strat.get("strategy")

                if action not in ("AVOID", "REVIEW"):
                    candidates.append({
                        "ticker": ticker,
                        "action": action,
                        "mv": mv,
                        "probs": probs,
                        "edge": edge,
                        "strategy": strategy_name,
                        "source": source_val,
                        "greeks": greeks,
                    })

            # Rank & simple budget limits
            if candidates:
                candidates.sort(key=lambda x: x["edge"], reverse=True)
                selected: List[Dict[str, Any]] = []
                buys = sells = 0
                for c in candidates:
                    if len(selected) >= _MAX_TRADES:
                        break
                    side = _map_order_side(c["action"])
                    if side == "buy" and buys >= _MAX_BUYS:
                        continue
                    if side == "sell" and sells >= _MAX_SELLS:
                        continue
                    selected.append(c)
                    if side == "buy":
                        buys += 1
                    else:
                        sells += 1

                now_iso = now.isoformat()
                for c in selected:
                    ticker = c["ticker"]
                    action = c["action"]
                    mv     = c["mv"]
                    probs  = c["probs"]
                    side   = _map_order_side(action)

                    if not self._exec_dedup_ok(ticker, action):
                        continue

                    # Pass full context so logs are complete
                    self.executor.place_order(
                        ticker=ticker,
                        size=ORDER_SIZE,
                        side=side,
                        action=action,
                        movement=mv,
                        probs=probs,
                        strategy=c.get("strategy"),
                        source=c.get("source"),
                    )
                    EXECUTIONS.labels(ticker=ticker, action=action).inc()

                    # Persist signal
                    append_signal_log({
                        "time": now_iso,
                        "ticker": ticker,
                        "movement_type": mv,
                        "action": action,
                        **(c.get("greeks") or {}),
                    })

            time.sleep(300)


if __name__ == "__main__":
    write_status(f"[{_INSTANCE_ID}] Launching Grond Orchestrator…")
    orchestrator = GrondOrchestrator()
    orchestrator.run()

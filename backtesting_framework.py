import backtrader as bt
import pandas as pd
import numpy as np

from ml_classifier import MLClassifier
from strategy_logic import StrategyLogic
from utils import (
    reformat_candles,
    calculate_breakout_prob,
    calculate_recent_move_pct,
    calculate_time_of_day,
    fetch_option_greeks
)

class GrondBacktraderStrategy(bt.Strategy):
    params = dict(
        classifier=None,
        strat_logic=None,
        lookback=12
    )

    def __init__(self):
        self.classifier = self.p.classifier
        self.logic      = self.p.strat_logic
        self.data_open  = self.datas[0].open
        self.data_high  = self.datas[0].high
        self.data_low   = self.datas[0].low
        self.data_close = self.datas[0].close
        self.data_vol   = self.datas[0].volume

    def next(self):
        if len(self) < self.p.lookback:
            return

        bars = []
        for i in range(-self.p.lookback, 0):
            dt = self.datas[0].datetime.datetime(i)
            bars.append({
                "timestamp": int(dt.timestamp() * 1000),
                "open":      self.data_open[i],
                "high":      self.data_high[i],
                "low":       self.data_low[i],
                "close":     self.data_close[i],
                "volume":    self.data_vol[i]
            })

        candles = reformat_candles(bars)
        now     = self.datas[0].datetime.datetime(0)

        features = {
            **fetch_option_greeks(self.datas[0]._name),
            "breakout_prob":   calculate_breakout_prob(candles),
            "recent_move_pct": calculate_recent_move_pct(self.datas[0]._name, candles),
            "time_of_day":     calculate_time_of_day(now)
        }

        cls_out = self.classifier.classify(features)
        mv      = cls_out["movement_type"]
        strat   = self.logic.execute_strategy(mv, {**features, **cls_out})

        action = strat["action"]
        size   = self.broker.getcash() / self.data_close[0]

        if action.endswith("_CALL") and not self.position:
            self.buy(size=size)
        elif action.endswith("_PUT") and not self.position:
            self.sell(size=size)


def run_backtest_backtrader(
    df: pd.DataFrame,
    ticker: str,
    start_cash: float = 100_000,
    commission: float = 0.001,
    risk_free_rate: float = 0.0
) -> dict:
    cerebro = bt.Cerebro(stdstats=False)

    data_feed = bt.feeds.PandasData(
        dataname=df,
        name=ticker,
        fromdate=df.index[0],
        todate=df.index[-1]
    )
    cerebro.adddata(data_feed)

    classifier = MLClassifier()
    logic      = StrategyLogic()
    cerebro.addstrategy(
        GrondBacktraderStrategy,
        classifier=classifier,
        strat_logic=logic,
        lookback=12
    )

    cerebro.broker.setcash(start_cash)
    cerebro.broker.setcommission(commission=commission)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=risk_free_rate)
    cerebro.addanalyzer(bt.analyzers.DrawDown,      _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TimeReturn,    _name="timereturn")

    results = cerebro.run()
    strat   = results[0]

    sharpe       = strat.analyzers.sharpe.get_analysis().get("sharperatio", np.nan)
    max_drawdown = strat.analyzers.drawdown.get_analysis().max.drawdown
    timereturn   = pd.Series(strat.analyzers.timereturn.get_analysis())

    total_return = timereturn.add(1).prod() - 1
    days         = (df.index[-1] - df.index[0]).days / 365.0
    cagr         = (1 + total_return) ** (1 / days) - 1 if days > 0 else np.nan

    return {
        "Total Return":    total_return,
        "CAGR":            cagr,
        "Sharpe Ratio":    sharpe,
        "Max Drawdown %":  max_drawdown
    }

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
from movement_classifier import MovementClassifier
from strategy_logic import StrategyLogic
from utils import (
    fetch_option_greeks,
    calculate_time_of_day,
    reformat_candles,
    calculate_breakout_prob,
    calculate_recent_move_pct
)


class GrondBacktraderStrategy(bt.Strategy):
    params = dict(
        classifier=None,
        strat_logic=None,
        lookback=12  # number of 5m bars to look back for features
    )

    def __init__(self):
        # pull in classifier and strategy logic
        self.classifier = self.p.classifier
        self.logic = self.p.strat_logic
        # we assume data feed has OHLCV fields
        self.data_open = self.datas[0].open
        self.data_high = self.datas[0].high
        self.data_low  = self.datas[0].low
        self.data_close= self.datas[0].close
        self.data_vol  = self.datas[0].volume

    def next(self):
        # only proceed once we have enough history
        if len(self) < self.p.lookback:
            return

        # build a list of the last `lookback` bars
        bars = []
        for i in range(-self.p.lookback, 0):
            dt = self.datas[0].datetime.datetime(i)
            bars.append({
                "timestamp": int(dt.timestamp() * 1000),
                "open": self.data_open[i],
                "high": self.data_high[i],
                "low": self.data_low[i],
                "close": self.data_close[i],
                "volume": self.data_vol[i],
            })
        # reformat for our util functions
        candles = reformat_candles(bars)
        now = self.datas[0].datetime.datetime(0)

        # compute features
        breakout = calculate_breakout_prob(candles)
        move_pct  = calculate_recent_move_pct(self.datas[0]._name, candles)
        tod       = calculate_time_of_day(now)
        greeks    = fetch_option_greeks(self.datas[0]._name)

        features = {
            **greeks,
            "breakout_prob": breakout,
            "recent_move_pct": move_pct,
            "time_of_day": tod
        }

        # classify movement
        cls_out = self.classifier.classify(features)
        mv = cls_out["movement_type"]

        # execute strategy logic
        strat = self.logic.execute_strategy(mv, features)
        action = strat["action"]

        # simple execution: buy full cash if CALL, sell/short if PUT
        size = self.broker.getcash() / self.data_close[0]
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
    """
    Run a Backtrader backtest on a single-asset 5m DataFrame.

    :param df: DataFrame indexed by datetime, with columns ['open','high','low','close','volume'].
    :param ticker: symbol name (used for labeling).
    :param start_cash: starting capital.
    :param commission: per-trade commission fraction.
    :param risk_free_rate: used for Sharpe ratio calculation.
    :return: dict of performance metrics.
    """
    cerebro = bt.Cerebro(stdstats=False)

    data = bt.feeds.PandasData(
        dataname=df,
        name=ticker,
        fromdate=df.index[0],
        todate=df.index[-1]
    )
    cerebro.adddata(data)

    classifier = MovementClassifier()
    logic = StrategyLogic()
    cerebro.addstrategy(
        GrondBacktraderStrategy,
        classifier=classifier,
        strat_logic=logic,
        lookback=12
    )

    cerebro.broker.setcash(start_cash)
    cerebro.broker.setcommission(commission=commission)

    # analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=risk_free_rate)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")

    results = cerebro.run()
    strat = results[0]

    sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio", np.nan)
    dd = strat.analyzers.drawdown.get_analysis().max.drawdown
    tr = pd.Series(strat.analyzers.timereturn.get_analysis())

    total_ret = tr.add(1).prod() - 1
    days = (df.index[-1] - df.index[0]).days / 365.0
    cagr = (1 + total_ret) ** (1 / days) - 1 if days > 0 else np.nan

    return {
        "Total Return": total_ret,
        "CAGR": cagr,
        "Sharpe Ratio": sharpe,
        "Max Drawdown %": dd
    }

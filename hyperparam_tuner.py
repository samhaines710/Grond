import os
import json
from datetime import datetime, timedelta

import pandas as pd
from data_ingestion import HistoricalDataLoader
from backtesting_framework import run_backtest_backtrader
from signal_generation import AdaptiveHyperparamOptimizer

CONFIG_PATH     = "movement_logic_config.json"
TMP_CONFIG_PATH = "movement_logic_config.tmp.json"

def load_config(path=CONFIG_PATH) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_config(cfg: dict, path: str):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)

def backtest_with_config(cfg: dict, symbol: str="TSLA") -> float:
    """
    Fetch 30 days of historical 5m bars for `symbol`, run
    Backtrader backtest, return Sharpe ratio.
    """
    # 1) Fetch data
    end   = datetime.now()
    start = end - timedelta(days=30)
    loader = HistoricalDataLoader()
    raw = loader.fetch_bars(symbol, start, end)
    # 2) To DataFrame
    df = pd.DataFrame([{
        "open": b["o"], "high": b["h"], "low": b["l"],
        "close": b["c"], "volume": b["v"],
    } for b in raw])
    df.index = pd.to_datetime([b["t"] for b in raw], unit="ms")
    # 3) Write temp config so classifier picks up thresholds
    save_config(cfg, TMP_CONFIG_PATH)
    # 4) Run backtest
    perf = run_backtest_backtrader(df, symbol)
    # 5) Cleanup tmp file
    os.remove(TMP_CONFIG_PATH)
    return perf.get("Sharpe Ratio", 0.0)

def make_trial_config(base: dict, params: dict) -> dict:
    """
    Overlay `params` onto `base` config structure.
    Each key in params maps to one threshold in the JSON.
    """
    cfg = base.copy()

    # --- safe_greek_bands (symmetric) ---
    for greek in ("vanna","vomma","charm","veta","speed","zomma","color","rho"):
        key = f"safe_{greek}"
        val = params[key]
        cfg.setdefault("safe_greek_bands", {})[greek] = [-val, val]

    # --- tier1_greek_bands ---
    for greek in ("vanna","vomma","charm","veta"):
        key = f"tier1_{greek}"
        val = params[key]
        cfg.setdefault("tier1_greek_bands", {})[greek] = [-val, val]

    # --- vol/hedge bands ---
    for greek in ("vanna","veta"):
        vkey = f"vol_{greek}"
        hkey = f"hedge_{greek}"
        cfg.setdefault("volatility_greek_bands", {})[greek] = [-params[vkey], params[vkey]]
        cfg.setdefault("hedging_greek_bands",    {})[greek] = [-params[hkey], params[hkey]]

    # --- RSI & other singles ---
    cfg["rsi_overbought"]      = params["rsi_overbought"]
    cfg["rsi_oversold"]        = params["rsi_oversold"]
    cfg["corr_dev_threshold"]  = params["corr_dev"]
    cfg["skew_extreme"]        = params["skew_extreme"]
    cfg["yield_spike_threshold"] = params["yield_spike"]

    # --- micro thresholds ---
    cfg.setdefault("micro_threshold", {})
    cfg["micro_threshold"]["default"]  = params["micro_default"]
    cfg["micro_threshold"]["high_vol"] = params["micro_high_vol"]
    cfg["micro_threshold"]["low_vol"]  = params["micro_low_vol"]

    # --- session thresholds ---
    for sess in ("MORNING","MIDDAY","AFTERNOON"):
        base_s = cfg.setdefault("sessions", {}).get(sess, {})
        base_s["breakout_low"]  = params[f"{sess}_b_low"]
        base_s["breakout_high"] = params[f"{sess}_b_high"]
        base_s["vol_thr"]       = params[f"{sess}_vol_thr"]
        base_s["delta_thr"]     = params[f"{sess}_delta_thr"]
        cfg["sessions"][sess] = base_s

    return cfg

# --- Define hyperparameter search space ---
param_space = {
    # safe greek bands
    **{f"safe_{g}": {"low": 0.05, "high": 1.0, "step": 0.05}
       for g in ("vanna","vomma","charm","veta","speed","zomma","color","rho")},
    # tier1 greek bands
    **{f"tier1_{g}": {"low": 0.01, "high": 0.2, "step": 0.01}
       for g in ("vanna","vomma","charm","veta")},
    # vol/hedge bands
    "vol_vanna":  {"low": 0.1, "high": 2.0, "step": 0.1},
    "hedge_vanna":{"low": 0.1, "high": 1.0, "step": 0.1},
    "vol_veta":   {"low": 0.1, "high": 2.0, "step": 0.1},
    "hedge_veta": {"low": 0.1, "high": 1.0, "step": 0.1},
    # RSI & others
    "rsi_overbought":     {"low": 50,  "high": 90, "step": 5},
    "rsi_oversold":       {"low": 10,  "high": 50, "step": 5},
    "corr_dev":           {"low": 0.5, "high": 3.0, "step": 0.1},
    "skew_extreme":       {"low": 0.1, "high": 1.0, "step": 0.1},
    "yield_spike":        {"low": 0.05,"high": 0.5, "step": 0.05},
    # micro thresholds
    "micro_default":      {"low": 0.1, "high": 0.5, "step": 0.05},
    "micro_high_vol":     {"low": 0.2, "high": 1.0, "step": 0.1},
    "micro_low_vol":      {"low": 0.05,"high": 0.3, "step": 0.05},
    # session thresholds
    **{
        f"{sess}_{fld}": {
            "low": (5 if "b_low" in fld else 10),
            "high": (30 if "b_low" in fld else 60),
            "step": 5
        }
        for sess in ("MORNING","MIDDAY","AFTERNOON")
        for fld in ("b_low","b_high")
    },
    **{
        f"{sess}_vol_thr":   {"low": 0.5, "high": 3.0, "step": 0.1}
        for sess in ("MORNING","MIDDAY","AFTERNOON")
    },
    **{
        f"{sess}_delta_thr": {"low": 0.1, "high": 1.0, "step": 0.1}
        for sess in ("MORNING","MIDDAY","AFTERNOON")
    },
}

def tune(n_trials: int = 50):
    base_cfg = load_config()
    def trial_fn(trial_params):
        # build a trial-specific config
        cfg = make_trial_config(base_cfg, trial_params)
        # return Sharpe ratio from backtest
        return backtest_with_config(cfg)

    tuner = AdaptiveHyperparamOptimizer(
        backtest_func=trial_fn,
        param_space=param_space,
        n_trials=n_trials,
        direction="maximize"
    )
    study = tuner.optimize()
    best = study.best_params

    # merge best back into your real config
    final_cfg = make_trial_config(base_cfg, best)
    save_config(final_cfg, CONFIG_PATH)
    print("✅ Tuning complete. Best params:", best)
    print(f"Updated thresholds written to {CONFIG_PATH}")

if __name__ == "__main__":
    tune(n_trials=100)

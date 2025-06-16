import numpy as np
import pandas as pd
import cvxpy as cp
from pypfopt import EfficientFrontier, risk_models, expected_returns


class CVXMeanVarianceOptimizer:
    """
    Mean–variance optimizer using CVXPY.
    """

    def __init__(self, returns: pd.DataFrame, freq: int = 252):
        """
        :param returns: DataFrame of historical returns (rows = dates, cols = assets).
        :param freq: Annualization frequency (e.g. 252 trading days).
        """
        # estimate expected returns and covariance
        self.mu = expected_returns.mean_historical_return(returns, frequency=freq)
        self.S  = risk_models.sample_cov(returns, frequency=freq)

    def optimize(self,
                 target_return: float | None = None,
                 risk_aversion: float = 1.0
                 ) -> pd.Series:
        """
        Solve for portfolio weights.

        :param target_return: if provided, minimize variance subject to achieving at least this return.
                              if None, maximize (return − risk_aversion × variance).
        :param risk_aversion: trade‐off parameter when target_return is None.
        :return: Series of weights summing to 1.
        """
        n = len(self.mu)
        w = cp.Variable(n)
        mu_vals = self.mu.values
        Sigma   = self.S.values

        ret  = mu_vals @ w
        risk = cp.quad_form(w, Sigma)

        constraints = [cp.sum(w) == 1, w >= 0]
        if target_return is not None:
            constraints.append(ret >= target_return)
            prob = cp.Problem(cp.Minimize(risk), constraints)
        else:
            prob = cp.Problem(cp.Maximize(ret - risk_aversion * risk), constraints)

        prob.solve(solver=cp.OSQP)
        return pd.Series(w.value, index=self.mu.index)


class PyPortfolioOptOptimizer:
    """
    Portfolio optimization via PyPortfolioOpt library.
    """

    def __init__(self, returns: pd.DataFrame, freq: int = 252):
        """
        :param returns: DataFrame of historical returns.
        :param freq: Annualization frequency.
        """
        self.mu = expected_returns.mean_historical_return(returns, frequency=freq)
        # use Ledoit–Wolf shrinkage for covariance
        self.S  = risk_models.CovarianceShrinkage(returns, frequency=freq).ledoit_wolf()

    def max_sharpe(self, weight_bounds: tuple = (0, 1)) -> pd.Series:
        """
        Maximize Sharpe ratio subject to weight bounds.
        :param weight_bounds: (min_weight, max_weight) for each asset.
        :return: Series of optimized weights.
        """
        ef = EfficientFrontier(self.mu, self.S, weight_bounds=weight_bounds)
        ef.max_sharpe()
        return pd.Series(ef.clean_weights())

    def min_volatility(self, weight_bounds: tuple = (0, 1)) -> pd.Series:
        """
        Minimize portfolio volatility subject to weight bounds.
        :param weight_bounds: (min, max) per asset.
        :return: Series of optimized weights.
        """
        ef = EfficientFrontier(self.mu, self.S, weight_bounds=weight_bounds)
        ef.min_volatility()
        return pd.Series(ef.clean_weights())


class RiskParityOptimizer:
    """
    Risk‐parity portfolio optimizer via convex optimization.
    """

    def __init__(self, returns: pd.DataFrame, freq: int = 252):
        """
        :param returns: DataFrame of historical returns.
        :param freq: Annualization frequency.
        """
        self.S = risk_models.sample_cov(returns, frequency=freq)

    def optimize(self) -> pd.Series:
        """
        Solve for weights such that each asset contributes equally to portfolio risk.
        """
        Sigma = self.S.values
        n = Sigma.shape[0]
        w = cp.Variable(n)

        # Risk contributions: w_i * (Sigma @ w)_i
        rc = cp.multiply(w, Sigma @ w)
        total_rc = cp.sum(rc)
        # objective: minimize sum squared deviations from equal risk contribution
        target = total_rc / n
        objective = cp.sum_squares(rc - target)

        constraints = [cp.sum(w) == 1, w >= 0]
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver=cp.OSQP)

        return pd.Series(w.value, index=self.S.index)

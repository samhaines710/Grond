# pricing_engines.py

import math
from enum import Enum

# QuantLib
try:
    import QuantLib as ql
    _HAS_QUANTLIB = True
except ImportError:
    _HAS_QUANTLIB = False

# JAX
try:
    import jax.numpy as jnp
    from jax import jit
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


class EngineType(str, Enum):
    QUANTLIB = "QuantLib"
    JAX      = "JAX"
    FALLBACK = "Fallback"


class QuantLibEngine:
    def __init__(self):
        if not _HAS_QUANTLIB:
            raise ImportError("QuantLib is not installed")
        self.calendar  = ql.NullCalendar()
        self.day_count = ql.Actual365Fixed()
        today = ql.Date().todaysDate()
        ql.Settings.instance().evaluationDate = today
        self.today = today

    def price(self, spot: float, strike: float, vol: float,
              maturity: float, rate: float, dividend: float,
              option_type: str) -> float:
        settlement = self.today
        rf      = ql.YieldTermStructureHandle(ql.FlatForward(settlement, rate, self.day_count))
        dq      = ql.YieldTermStructureHandle(ql.FlatForward(settlement, dividend, self.day_count))
        vol_ts  = ql.BlackVolTermStructureHandle(
                     ql.BlackConstantVol(settlement, self.calendar, vol, self.day_count)
                  )
        process = ql.BlackScholesMertonProcess(
                      ql.QuoteHandle(ql.SimpleQuote(spot)),
                      dq, rf, vol_ts
                  )
        payoff = ql.PlainVanillaPayoff(
                     ql.Option.Call if option_type.lower()=="call" else ql.Option.Put,
                     strike
                 )
        days_to_expiry = max(int(maturity * 365), 1)
        expiry = settlement + days_to_expiry
        option = ql.VanillaOption(payoff, ql.EuropeanExercise(expiry))
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        return option.NPV()


if _HAS_JAX:
    @jit
    def _jax_bs_price(spot, strike, vol, maturity, rate, dividend, is_call):
        sqrtT = jnp.sqrt(maturity)
        d1 = (jnp.log(spot/strike) + (rate - dividend + 0.5*vol**2)*maturity) / (vol * sqrtT)
        d2 = d1 - vol*sqrtT
        cdf = lambda x: 0.5*(1.0 + jnp.erf(x / jnp.sqrt(2)))
        c1, c2 = cdf(d1), cdf(d2)
        df_r = jnp.exp(-rate * maturity)
        df_d = jnp.exp(-dividend * maturity)
        call = spot*df_d*c1 - strike*df_r*c2
        put  = strike*df_r*(1 - c2) - spot*df_d*(1 - c1)
        return jnp.where(is_call, call, put)


class JAXEngine:
    def __init__(self):
        if not _HAS_JAX:
            raise ImportError("JAX is not installed")
        self._fn = _jax_bs_price

    def price(self, spot: float, strike: float, vol: float,
              maturity: float, rate: float, dividend: float,
              option_type: str) -> float:
        is_call = 1 if option_type.lower()=="call" else 0
        return float(self._fn(spot, strike, vol, maturity, rate, dividend, is_call))


class FallbackEngine:
    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def price(self, spot: float, strike: float, vol: float,
              maturity: float, rate: float, dividend: float,
              option_type: str) -> float:
        # Intrinsic value if no time or no vol
        if maturity <= 0 or vol <= 0:
            return max(spot - strike, 0) if option_type.lower()=="call" else max(strike - spot, 0)

        sqrtT = math.sqrt(maturity)
        d1 = (math.log(spot/strike) + (rate - dividend + 0.5*vol**2)*maturity) / (vol * sqrtT)
        d2 = d1 - vol*sqrtT
        c1 = self._norm_cdf(d1)
        c2 = self._norm_cdf(d2)
        df_r = math.exp(-rate * maturity)
        df_d = math.exp(-dividend * maturity)

        if option_type.lower() == "call":
            return spot*df_d*c1 - strike*df_r*c2
        else:
            return strike*df_r*(1 - c2) - spot*df_d*(1 - c1)


class DerivativesPricer:
    def __init__(self, engine: str = EngineType.QUANTLIB.value):
        et = EngineType(engine)
        if et == EngineType.QUANTLIB:
            self.engine = QuantLibEngine()
        elif et == EngineType.JAX:
            self.engine = JAXEngine()
        else:
            self.engine = FallbackEngine()

    def price_black_scholes(self,
                            spot: float,
                            strike: float,
                            vol: float,
                            maturity: float,
                            rate: float,
                            dividend: float,
                            option_type: str) -> float:
        return self.engine.price(spot, strike, vol, maturity, rate, dividend, option_type)

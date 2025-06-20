# utils/greeks_helpers.py

import math
from logging_utils import write_status
from config import RISK_FREE_RATE, DIVIDEND_YIELDS, DEFAULT_VOLATILITIES

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5*x*x)/math.sqrt(2*math.pi)

def _norm_cdf(x: float) -> float:
    return 0.5*(1 + math.erf(x/math.sqrt(2)))

def _bs_core(S: float, K: float, T: float, r: float, q: float, sigma: float, typ: str) -> dict:
    sqrtT = math.sqrt(max(T, 1e-6))
    d1 = (math.log(max(S/K,1e-6)) + (r - q + 0.5*sigma*sigma)*T)/(sigma*sqrtT)
    d2 = d1 - sigma*sqrtT
    pdf = _norm_pdf(d1); cdf1 = _norm_cdf(d1); cdf2 = _norm_cdf(d2)
    delta = math.exp(-q*T)*(cdf1 if typ=="call" else cdf1-1)
    gamma = math.exp(-q*T)*pdf/(S*sigma*sqrtT)
    vega  = S*math.exp(-q*T)*pdf*sqrtT
    term1 = -(S*sigma*math.exp(-q*T)*pdf)/(2*sqrtT)
    if typ=="call":
        term2 = q*S*math.exp(-q*T)*cdf1 - r*K*math.exp(-r*T)*cdf2
    else:
        term2 = -q*S*math.exp(-q*T)*_norm_cdf(-d1) + r*K*math.exp(-r*T)*_norm_cdf(-d2)
    return {"delta":delta, "gamma":gamma, "theta": term1+term2, "vega":vega}

def calculate_all_greeks(
    S: float, K: float, T: float, ticker: str, typ: str="call", sigma_override: float|None=None
) -> dict:
    """
    Returns the full set of Greeks (delta, gamma, theta_day, theta_5m, vega, rho, vanna, vomma, charm, veta, speed, zomma, color).
    """
    r = RISK_FREE_RATE
    q = DIVIDEND_YIELDS.get(ticker, 0.0)
    sigma = sigma_override if sigma_override is not None else DEFAULT_VOLATILITIES.get(ticker, 0.2)

    core = _bs_core(S, K, T, r, q, sigma, typ)
    d, g, th, v = core["delta"], core["gamma"], core["theta"], core["vega"]

    # rho
    sqrtT = math.sqrt(max(T,1e-6))
    d2 = (math.log(max(S/K,1e-6)) + (r - q + 0.5*sigma*sigma)*T)/(sigma*sqrtT) - sigma*sqrtT
    c2 = _norm_cdf(d2)
    rho = (K*T*math.exp(-r*T)*c2) if typ=="call" else (-K*T*math.exp(-r*T)*(1-c2))

    # higher‐order greeks
    ev = max(sigma*0.01, 1e-3)
    eT = max(T*0.01, 1e-4)
    eS = max(S*0.01, 0.1)
    vanna = (_bs_core(S,K,T,r,q,sigma+ev,typ)["delta"] - _bs_core(S,K,T,r,q,sigma-ev,typ)["delta"])/(2*ev)
    vomma = (_bs_core(S,K,T,r,q,sigma+ev,typ)["vega"]  - _bs_core(S,K,T,r,q,sigma-ev,typ)["vega"] )/(2*ev)
    charm = (_bs_core(S,K,T+eT,r,q,sigma,typ)["delta"] - _bs_core(S,K,T-eT,r,q,sigma,typ)["delta"])/(2*eT)
    veta  = (_bs_core(S,K,T+eT,r,q,sigma,typ)["vega"]  - _bs_core(S,K,T-eT,r,q,sigma,typ)["vega"]  )/(2*eT)
    speed = (_bs_core(S+eS,K,T,r,q,sigma,typ)["gamma"] - _bs_core(S-eS,K,T,r,q,sigma,typ)["gamma"])/(2*eS)
    zomma = (_bs_core(S,K,T,r,q,sigma+ev,typ)["gamma"] - _bs_core(S,K,T,r,q,sigma-ev,typ)["gamma"])/(2*ev)
    color = (_bs_core(S+eS,K,T,r,q,sigma,typ)["theta"] - _bs_core(S-eS,K,T,r,q,sigma,typ)["theta"])/(2*eS)

    return {
        "delta":          d,
        "gamma":          g,
        "theta_day":      th/365.0,
        "theta_5m":       th/365.0*(5/60/24),
        "vega":           v,
        "rho":            rho,
        "vanna":          vanna,
        "vomma":          vomma,
        "charm":          charm/365.0,
        "veta":           veta/365.0,
        "speed":          speed,
        "zomma":          zomma,
        "color":          color,
    }

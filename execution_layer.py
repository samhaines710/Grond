import math
import threading
import logging
from datetime import datetime
from typing import Callable, Dict

from data_ingestion import REALTIME_CANDLES, REALTIME_LOCK
from utils import write_status

logger = logging.getLogger("execution_layer")
logger.setLevel(logging.INFO)


class SlippageModel:
    """
    Collection of static methods to estimate slippage components.
    """

    @staticmethod
    def fixed(spread: float) -> float:
        """
        Half the bid–ask spread.
        """
        return spread / 2.0

    @staticmethod
    def volume_impact(
        order_size: float,
        adv: float,
        impact_coefficient: float = 0.1,
        exponent: float = 0.6
    ) -> float:
        """
        Market impact based on participation rate.
        """
        participation = order_size / max(adv, 1e-6)
        return impact_coefficient * (participation ** exponent)

    @staticmethod
    def volatility_impact(
        volatility: float,
        vol_coeff: float = 0.5
    ) -> float:
        """
        Impact proportional to volatility.
        """
        return vol_coeff * volatility

    @classmethod
    def total(
        cls,
        spread: float,
        order_size: float,
        adv: float,
        volatility: float,
        vol_coeff: float = 0.5,
        impact_coeff: float = 0.1,
        exponent: float = 0.6
    ) -> float:
        """
        Sum of fixed, volume, and volatility impacts.
        """
        return (
            cls.fixed(spread)
            + cls.volume_impact(order_size, adv, impact_coefficient=impact_coeff, exponent=exponent)
            + cls.volatility_impact(volatility, vol_coeff=vol_coeff)
        )


class ManualExecutor:
    """
    Executor that emits a manual signal (e.g. via Telegram) and logs it.
    """

    def __init__(self, notify_fn: Callable[[str], None]):
        """
        :param notify_fn: function to call with a markdown-formatted message.
        """
        self.notify = notify_fn

    def place_order(self, ticker: str, size: float, side: str = "buy") -> Dict[str, Any]:
        """
        “Execute” a manual signal: log it, notify, and return a report dict.

        :param ticker: symbol to trade
        :param size:   number of contracts/shares
        :param side:   "buy" or "sell"
        :return:       order report
        """
        report = {
            "ticker":   ticker,
            "side":     side.lower(),
            "size":     size,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "note":     "Manual execution required"
        }

        # Log to console and status file
        msg = f"MANUAL EXECUTION → {side.upper()} {size} {ticker}"
        logger.info(msg)
        write_status(msg)

        # Send Telegram notification if configured
        try:
            text = f"📋 *MANUAL SIGNAL* — {side.upper()} {size} {ticker} @ {datetime.utcnow().strftime('%H:%M')} UTC"
            self.notify(text)
        except Exception as e:
            logger.warning(f"Failed to send notification: {e}")

        return report

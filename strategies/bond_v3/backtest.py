"""
V3 backtest engine — wires config_v3, signals_v3, portfolio_v3 into the shared run_bond() core.

V3 difference: trailing stops apply to HEDGE_ETFS + MANAGED_FUTURES_ETFS.
"""
import logging
import configs.bond_v3 as config
import strategies.bond_v3.signals  as signals
import strategies.bond_v3.portfolio as portfolio

from strategies.bond_shared import run_bond
from strategies.backtest_core import effective_weights_core

logger = logging.getLogger(__name__)


def effective_weights(signal_weights, recent_prices):
    return effective_weights_core(
        signal_weights, recent_prices,
        stop_etfs=config.TRAILING_STOP_ETFS,
        stop_pct=config.TRAILING_STOP_PCT,
        stop_window=config.TRAILING_STOP_WINDOW,
    )


def run(macro, prices):
    # V3: stops cover both commodities and managed futures
    stop_etfs = config.HEDGE_ETFS + config.MANAGED_FUTURES_ETFS
    return run_bond(macro, prices, config, signals, portfolio, stop_etfs=stop_etfs)

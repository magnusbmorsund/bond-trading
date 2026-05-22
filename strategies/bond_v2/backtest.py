"""
V2 backtest engine — wires config_v2, signals_v2, portfolio_v2 into the shared run_bond() core.
"""
import logging
import configs.bond_v2 as config
import strategies.bond_v2.signals  as signals
import strategies.bond_v2.portfolio as portfolio

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
    return run_bond(macro, prices, config, signals, portfolio)

"""
V1 backtest engine — wires config_v1, signals_v1, portfolio_v1 into the shared run_bond() core.
"""
import logging
import configs.bond_v1 as config
import strategies.bond_v1.signals  as signals
import strategies.bond_v1.portfolio as portfolio

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

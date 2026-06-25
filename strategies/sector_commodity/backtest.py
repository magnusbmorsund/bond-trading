"""Commodity Supercycle backtest — V2e multi-timescale momentum on a commodity-
only universe + FRED macro exposure overlay. Thin wrapper over sector_shared.backtest.
"""
import configs.sector_commodity as config
from strategies.sector_commodity import signals, portfolio, macro
from strategies.sector_shared.backtest import (
    effective_weights as _ew,
    compute_stop_pcts as _csp,
    run               as _run,
)


def effective_weights(signal_weights, recent_prices):
    return _ew(signal_weights, recent_prices, config)


def compute_stop_pcts(signal_weights, recent_prices):
    return _csp(signal_weights, recent_prices, config)


def run(prices):
    macro_exposure = None
    if getattr(config, "MACRO_ENABLED", True):
        macro_exposure = macro.compute_exposure(config, prices.index)
    return _run(prices, config, signals, portfolio, macro_exposure=macro_exposure)

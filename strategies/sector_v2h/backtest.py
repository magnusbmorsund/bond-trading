"""V2h backtest — thin wrapper. Tight trailing stop, fill-at-stop execution."""
import configs.sector_v2h as config
from strategies.sector_v2h import signals, portfolio
from strategies.sector_shared.backtest import (
    effective_weights as _ew, compute_stop_pcts as _csp, run as _run,
)
def effective_weights(signal_weights, recent_prices):
    return _ew(signal_weights, recent_prices, config)
def compute_stop_pcts(signal_weights, recent_prices):
    return _csp(signal_weights, recent_prices, config)
def run(prices):
    return _run(prices, config, signals, portfolio)

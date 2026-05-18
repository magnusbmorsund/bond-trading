"""V2c portfolio — thin wrapper around strategies.sector_shared.portfolio."""
import configs.sector_v2c as config
from strategies.sector_shared.portfolio import (
    build_weights       as _bw,
    build_weight_series as _bws,
)


def build_weights(score_row, vol_row, regime, corr_matrix=None):
    return _bw(score_row, vol_row, regime, config, corr_matrix=corr_matrix)


def build_weight_series(score_periodic, vol_periodic, regime_periodic, corr_periodic=None):
    return _bws(score_periodic, vol_periodic, regime_periodic, config, corr_periodic)

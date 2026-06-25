"""Data pipeline for the commodity-supercycle variant (commodity-only universe)."""
import configs.sector_commodity as config
from data.pipelines.sector_base import load_all as _base_load_all


def load_all(force: bool = False):
    return _base_load_all(config, "sector_commodity_prices.csv", "commodity supercycle", force=force)

"""Data pipeline for the V2h variant (same UCITS universe as V2f)."""
import configs.sector_v2h as config
from data.pipelines.sector_base import load_all as _base_load_all

def load_all(force: bool = False):
    return _base_load_all(config, "sector_v2h_prices.csv", "sector V2h", force=force)

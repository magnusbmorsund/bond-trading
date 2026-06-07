"""Data pipeline for the V2g honest sector rotation variant (same universe as V2f)."""
import configs.sector_v2g as config
from data.pipelines.sector_base import load_all as _base_load_all

def load_all(force: bool = False):
    return _base_load_all(config, "sector_v2g_prices.csv", "sector V2g", force=force)

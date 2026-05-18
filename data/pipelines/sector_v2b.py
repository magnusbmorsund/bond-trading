"""Data pipeline for the V2b sector rotation strategy."""
import configs.sector_v2b as config
from data.pipelines.sector_base import load_all as _base_load_all

def load_all(force: bool = False):
    return _base_load_all(config, "sector_v2b_prices.csv", "sector V2b", force=force)

"""Data pipeline for the V2d sector rotation strategy."""
import configs.sector_v2d as config
from data.pipelines.sector_base import load_all as _base_load_all

def load_all(force: bool = False):
    return _base_load_all(config, "sector_v2d_prices.csv", "sector V2d", force=force)

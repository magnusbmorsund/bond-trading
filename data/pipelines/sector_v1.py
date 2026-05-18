"""Data pipeline for the V1 sector rotation strategy."""
import configs.sector_v1 as config
from data.pipelines.sector_base import load_all as _base_load_all

def load_all(force: bool = False):
    return _base_load_all(config, "sector_prices.csv", "sector V1", force=force)

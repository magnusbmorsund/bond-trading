"""Data pipeline for the V2c sector rotation strategy."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config_sector_v2c as config
from data.pipeline_sector_base import load_all as _base_load_all

_LABEL = "V2c sector"
_CACHE = "sector_v2c_prices.csv"


def load_all(force: bool = False):
    return _base_load_all(config, _CACHE, _LABEL, force=force)


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s  %(message)s")
    prices = load_all(force=True)
    print(f"\nPrice data: {prices.shape}  ({prices.index[0].date()} → {prices.index[-1].date()})")
    print("Columns:", list(prices.columns))
    print("\nLatest prices:\n", prices.tail(2).T)

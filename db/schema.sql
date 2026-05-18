-- Run this once in the Supabase SQL editor for the magnus-trading project.
-- Creates the two tables that store all strategy price and macro data.

CREATE TABLE IF NOT EXISTS etf_prices (
    date        DATE              NOT NULL,
    ticker      VARCHAR(12)       NOT NULL,
    close       DOUBLE PRECISION  NOT NULL,
    updated_at  TIMESTAMPTZ       DEFAULT now(),
    PRIMARY KEY (date, ticker)
);

CREATE TABLE IF NOT EXISTS fred_series (
    date        DATE              NOT NULL,
    series_id   VARCHAR(40)       NOT NULL,
    value       DOUBLE PRECISION  NOT NULL,
    updated_at  TIMESTAMPTZ       DEFAULT now(),
    PRIMARY KEY (date, series_id)
);

-- Indexes for fast per-ticker and per-series range scans
CREATE INDEX IF NOT EXISTS idx_etf_prices_ticker_date  ON etf_prices  (ticker,    date DESC);
CREATE INDEX IF NOT EXISTS idx_fred_series_id_date     ON fred_series (series_id, date DESC);

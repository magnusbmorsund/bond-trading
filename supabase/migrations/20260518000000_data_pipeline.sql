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

CREATE INDEX IF NOT EXISTS idx_etf_prices_ticker_date  ON etf_prices  (ticker,    date DESC);
CREATE INDEX IF NOT EXISTS idx_fred_series_id_date     ON fred_series (series_id, date DESC);

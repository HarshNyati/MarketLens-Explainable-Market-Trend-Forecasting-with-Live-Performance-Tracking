-- ============================================================
-- Market Trend Predictions – Database Schema
-- Run once to initialise a fresh database.
-- ============================================================

-- ─────────────────────────────────────────────────────────
-- 1. Raw market data (crypto + stocks)
--    Populated by: candle_ingester (1h candles)
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS market_data (
    id            SERIAL PRIMARY KEY,
    symbol        VARCHAR(20)    NOT NULL,
    price         NUMERIC(18, 6) NOT NULL,
    volume        NUMERIC(24, 4) NOT NULL DEFAULT 0,
    market_type   VARCHAR(10)    NOT NULL,   -- 'crypto' | 'stock'
    recorded_at   TIMESTAMPTZ    NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_market_data_symbol
    ON market_data (symbol);

CREATE INDEX IF NOT EXISTS idx_market_data_recorded_at
    ON market_data (recorded_at DESC);

-- Unique index prevents duplicate candles for the same symbol & timestamp
CREATE UNIQUE INDEX IF NOT EXISTS idx_market_data_symbol_recorded_at
    ON market_data (symbol, recorded_at);

-- ─────────────────────────────────────────────────────────
-- 2. Trend insights
--    Populated by: save_trends.py / trend_engine.py
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trend_insights (
    id           SERIAL PRIMARY KEY,
    symbol       VARCHAR(20)  NOT NULL,
    trend        VARCHAR(20)  NOT NULL,   -- 'UP' | 'DOWN' | 'STABLE'
    confidence   VARCHAR(20)  NOT NULL,   -- 'LOW' | 'MEDIUM' | 'HIGH'
    time_window  VARCHAR(20)  NOT NULL,   -- e.g. 'short'
    generated_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trend_insights_symbol
    ON trend_insights (symbol);

CREATE INDEX IF NOT EXISTS idx_trend_insights_generated_at
    ON trend_insights (generated_at DESC);

-- ─────────────────────────────────────────────────────────
-- 3. ML prediction results
--    Populated by: ml_predictor.py / prediction_store.py
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS prediction_results (
    id               SERIAL PRIMARY KEY,
    symbol           VARCHAR(20)     NOT NULL,
    direction        VARCHAR(20)     NOT NULL,   -- 'UP' | 'DOWN' | 'NEUTRAL' | 'NO_SIGNAL'
    confidence       NUMERIC(6, 2)   NOT NULL,   -- percentage 0-100
    predicted_at     TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    candle_timestamp TIMESTAMPTZ,
    model_version    VARCHAR(50)     DEFAULT 'v1.0',
    actual_return    NUMERIC(10, 6),
    actual_label     VARCHAR(10),
    was_correct      BOOLEAN
);


CREATE INDEX IF NOT EXISTS idx_prediction_results_symbol
    ON prediction_results (symbol);

CREATE INDEX IF NOT EXISTS idx_prediction_results_predicted_at
    ON prediction_results (predicted_at DESC);

-- Ensure only one prediction per asset per candle
CREATE UNIQUE INDEX IF NOT EXISTS idx_prediction_results_symbol_candle
    ON prediction_results (symbol, candle_timestamp);


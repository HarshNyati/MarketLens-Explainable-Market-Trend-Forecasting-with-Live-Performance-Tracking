import logging
from apscheduler.schedulers.background import BackgroundScheduler
from app.services.candle_ingester import run_candle_ingestion
from app.services.outcome_tracker import resolve_outcomes
from app.analysis.save_trends import save_trends
from app.services.ml_predictor import predict_symbol

logger = logging.getLogger(__name__)

ASSETS = ["BITCOIN", "ETHEREUM", "IBM"]


def run_pipeline():
    logger.info("Starting hourly pipeline execution...")

    # 1. Ingest 1h candles (backfills ~500 on first run, appends latest on subsequent)
    counts = run_candle_ingestion()
    logger.info("Candle ingestion complete: %s", counts)

    # 2. Resolve outcomes for past predictions whose horizon (5 bars) has elapsed
    try:
        resolved = resolve_outcomes()
        logger.info("Outcome resolution complete: %d predictions resolved", len(resolved))
    except Exception as exc:
        logger.error("Outcome resolution failed: %s", exc)

    # 3. Run trend analysis
    try:
        save_trends()
    except Exception as exc:
        logger.warning("Trend analysis skipped: %s", exc)

    # 3. Generate ML predictions for all assets and store to DB
    for sym in ASSETS:
        try:
            res = predict_symbol(sym)
            logger.info("Live prediction for %s: %s", sym, res.get("direction"))
        except Exception as exc:
            logger.error("Failed live prediction for %s: %s", sym, exc)

    logger.info("✅ Hourly pipeline executed successfully")


def start_scheduler():
    scheduler = BackgroundScheduler()
    # Run once shortly after startup or every hour
    scheduler.add_job(run_pipeline, "interval", hours=1)
    scheduler.start()
    logger.info("Scheduler started (interval: 1 hour)")

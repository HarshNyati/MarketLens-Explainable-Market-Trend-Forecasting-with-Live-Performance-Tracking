"""
ml/train_offline.py
────────────────────
Walk-forward training pipeline for market-trend prediction.

What it does
────────────
1.  Downloads intraday candles (default: 1 h) via ml.market_data_fetcher
    (yfinance).  Falls back to a local CSV if the download fails.
2.  Engineers features via the shared ml.feature_engineering module
    (same code used at inference time – no train/serve skew).
3.  Labels rows with the ternary target from make_target():
      UP (2)      – max return over next HORIZON bars >= +UP_THRESH
      DOWN (0)    – min return over next HORIZON bars <= -DN_THRESH
      NEUTRAL (1) – neither threshold crossed
4.  Runs TimeSeriesSplit walk-forward validation across:
      • Random Forest
      • XGBoost              (skipped gracefully if not installed)
      • LightGBM             (skipped gracefully if not installed)
      • Baseline – always predict UP
      • Baseline – always predict majority class
      • Baseline – momentum (repeat last observed direction)
5.  Calibrates the best model's probabilities with CalibratedClassifierCV
    (isotonic regression, 5-fold internal CV) before saving.
6.  Primary ranking metrics: balanced_accuracy, MCC, macro-F1.
    These are not inflated by "always-UP" the way accuracy and UP-F1 are.
7.  Runs a simple backtest:
      • Enter long when model predicts UP (class 2)
      • Fee: 0.1 % per round-trip trade
      • Benchmark: buy-and-hold over the same hold-out window
8.  Saves:
      ml/models/metrics.json         – per-asset, per-model metrics
      ml/models/backtest.json        – per-asset, per-model backtest
      ml/models/<symbol>_model.pkl   – calibrated best model
      ml/models/<symbol>_meta.json   – interval, horizon, thresholds
9.  Honest verdict: prints clearly if no ML model beats all baselines.

Usage
─────
    python -m ml.train_offline
    python ml/train_offline.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

# ── Optional heavy deps – graceful degradation ────────────────────────────────
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    print("⚠  xgboost not installed – skipping XGBoost model.")

try:
    from lightgbm import LGBMClassifier
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
    print("⚠  lightgbm not installed – skipping LightGBM model.")

# ── Project imports ───────────────────────────────────────────────────────────
from ml.feature_engineering import (
    DEFAULT_DN_THRESH,
    DEFAULT_HORIZON,
    DEFAULT_UP_THRESH,
    FEATURE_COLS,
    TARGET_CLASSES,
    create_features,
)
from ml.market_data_fetcher import DEFAULT_INTERVAL, fetch_candles

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Intraday data settings ────────────────────────────────────────────────────
CANDLE_INTERVAL = DEFAULT_INTERVAL   # from .env ML_CANDLE_INTERVAL (default "1h")

# ── Target settings (pulled from feature_engineering defaults) ────────────────
HORIZON   = DEFAULT_HORIZON    # 5 bars look-ahead
UP_THRESH = DEFAULT_UP_THRESH  # 0.5 %
DN_THRESH = DEFAULT_DN_THRESH  # 0.5 %

# ── Tracked assets ────────────────────────────────────────────────────────────
ASSETS: list[str] = ["BITCOIN", "ETHEREUM", "IBM"]

N_SPLITS          = 5         # TimeSeriesSplit folds
FEE_ROUNDTRIP     = 0.001     # 0.1 % per completed round-trip trade
MAX_ROWS          = 100_000   # cap rows to keep training tractable on a laptop
SIGNAL_THRESHOLD  = 0.40      # min calibrated probability to emit a live signal



# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_models() -> dict[str, object]:
    """Return a fresh dict of model name → unfitted estimator."""
    models: dict[str, object] = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
    if _HAS_LGB:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
        )
    return models


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helper
# ─────────────────────────────────────────────────────────────────────────────

def _moving_block_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    block_length: int = 2 * HORIZON,  # block length >= 2 * HORIZON (>= 10 bars)
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Compute 95% moving-block bootstrap confidence intervals for MCC and Balanced Accuracy.

    Preserves serial correlation and temporal dependence up to `block_length` bars.
    """
    n = len(y_true)
    if n < block_length * 2:
        return {"mcc_ci": [0.0, 0.0], "bal_acc_ci": [0.0, 0.0]}

    rng = np.random.default_rng(seed)
    num_blocks = int(np.ceil(n / block_length))
    max_start = n - block_length + 1

    mcc_samples = []
    ba_samples = []

    for _ in range(n_bootstraps):
        start_indices = rng.integers(0, max_start, size=num_blocks)
        sample_indices = np.concatenate([
            np.arange(start, start + block_length) for start in start_indices
        ])[:n]

        yt_sample = y_true[sample_indices]
        yp_sample = y_pred[sample_indices]

        # Handle degenerate single-class samples
        if len(np.unique(yt_sample)) > 1 and len(np.unique(yp_sample)) > 1:
            mcc_samples.append(matthews_corrcoef(yt_sample, yp_sample))
        else:
            mcc_samples.append(0.0)
        ba_samples.append(balanced_accuracy_score(yt_sample, yp_sample))

    low_pct = 100 * (alpha / 2)
    high_pct = 100 * (1 - alpha / 2)

    return {
        "mcc_ci": [
            round(float(np.percentile(mcc_samples, low_pct)), 4),
            round(float(np.percentile(mcc_samples, high_pct)), 4),
        ],
        "bal_acc_ci": [
            round(float(np.percentile(ba_samples, low_pct)), 4),
            round(float(np.percentile(ba_samples, high_pct)), 4),
        ],
    }


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute classification metrics with moving-block bootstrap 95% confidence intervals."""
    kw = dict(zero_division=0)
    ci = _moving_block_bootstrap_ci(y_true, y_pred)

    # Confusion matrix on canonical classes [DOWN=0, NEUTRAL=1, UP=2]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    # Move vs No-Move accuracy
    # No-move is NEUTRAL (class 1); Move is directional (DOWN=0 or UP=2)
    move_true = (y_true != TARGET_CLASSES["NEUTRAL"])
    move_pred = (y_pred != TARGET_CLASSES["NEUTRAL"])
    move_acc = round(float((move_true == move_pred).mean()), 4)

    return {
        "balanced_accuracy":        round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy_ci":     ci["bal_acc_ci"],
        "mcc":                      round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "mcc_ci":                   ci["mcc_ci"],
        "macro_f1":                 round(float(f1_score(y_true, y_pred, average="macro", **kw)), 4),
        "accuracy":                 round(float((y_true == y_pred).mean()), 4),
        "confusion_matrix":         cm,
        "confusion_labels":         ["DOWN", "NEUTRAL", "UP"],
        "move_vs_nomove_accuracy":  move_acc,
    }



# ─────────────────────────────────────────────────────────────────────────────
# Backtest
# ─────────────────────────────────────────────────────────────────────────────

def _backtest(
    prices: pd.Series,
    preds: np.ndarray,
    fee: float = FEE_ROUNDTRIP,
) -> dict[str, float]:
    """Long-only out-of-sample strategy vs buy-and-hold.

    Signal is generated at bar t close -> applied at bar t+1.
    Includes round-trip transaction fees (0.1%).
    Reports cumulative return, trade count, max drawdown, and Sharpe ratio.
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    if n < 2:
        return {
            "strategy_return": 1.0,
            "buyhold_return":  1.0,
            "n_trades":        0,
            "max_drawdown":    0.0,
            "sharpe_ratio":    0.0,
        }

    # Bar return of the underlying
    bar_ret = prices[1:] / prices[:-1]   # shape (n-1,)

    # Signal: preds[t] generated at close of bar t → act at bar t+1
    # 2 = UP (long), 1 = NEUTRAL (cash), 0 = DOWN (cash)
    signal = (preds[:-1] == TARGET_CLASSES["UP"]).astype(int)

    # Detect trade entries (0 -> 1) and exits (1 -> 0)
    prev_signal = np.concatenate([[0], signal[:-1]])
    entries = (prev_signal == 0) & (signal == 1)
    exits   = (prev_signal == 1) & (signal == 0)
    n_trades = int(entries.sum())

    # Apply per-leg fees: fee/2 upon entering, fee/2 upon exiting
    fee_per_leg = fee / 2.0
    fee_mult = np.ones(n - 1, dtype=float)
    fee_mult[entries] *= (1.0 - fee_per_leg)
    fee_mult[exits]   *= (1.0 - fee_per_leg)


    strat_bar_ret = np.where(signal == 1, bar_ret, 1.0) * fee_mult
    equity_curve = np.cumprod(strat_bar_ret)
    strat_cum = float(equity_curve[-1])

    # Max Drawdown
    peaks = np.maximum.accumulate(equity_curve)
    dds = (equity_curve - peaks) / peaks
    max_dd = float(np.min(dds)) if len(dds) > 0 else 0.0

    # Sharpe Ratio (annualized for 1h bars: 8,760 hours/year)
    excess_ret = strat_bar_ret - 1.0
    std_ret = float(np.std(excess_ret))
    if std_ret > 1e-9:
        sharpe = float((np.mean(excess_ret) / std_ret) * np.sqrt(8760))
    else:
        sharpe = 0.0

    bh_cum = float(prices[-1] / prices[0])

    return {
        "strategy_return": round(strat_cum, 4),
        "buyhold_return":  round(bh_cum, 4),
        "n_trades":        n_trades,
        "max_drawdown":    round(max_dd, 4),
        "sharpe_ratio":    round(sharpe, 2),
    }



# ─────────────────────────────────────────────────────────────────────────────
# Per-asset training loop
# ─────────────────────────────────────────────────────────────────────────────

def _train_asset(symbol: str) -> tuple[dict, dict]:
    """Walk-forward training + evaluation for one asset.

    Returns
    -------
    asset_metrics  : { model_name: {balanced_accuracy, mcc, macro_f1, accuracy} }
    asset_backtest : { model_name: {strategy_return, buyhold_return, n_trades} }
    """
    print(f"\n{'═'*60}")
    print(f"  {symbol}  (interval={CANDLE_INTERVAL})")
    print(f"{'═'*60}")

    # ── 1. Load data via yfinance ─────────────────────────────────────────────
    raw = fetch_candles(symbol, interval=CANDLE_INTERVAL)

    if raw.empty:
        print(f"  ⚠  yfinance download returned no data for {symbol} – skipping.")
        return {}, {}
    else:
        print(f"  ✅ Downloaded {len(raw):,} candles from yfinance")

    raw = raw.tail(MAX_ROWS)

    # ── 2. Feature engineering (uses configurable ternary target) ─────────────
    df, _ = create_features(raw, horizon=HORIZON, up_thresh=UP_THRESH, dn_thresh=DN_THRESH)
    print(f"  Rows after feature engineering: {len(df):,}")

    # Report class distribution so we can spot severe imbalance
    counts = pd.Series(df["target"]).value_counts().sort_index()
    total  = len(df)
    dist_str = "  ".join(
        f"{['DOWN','NEUTRAL','UP'][c]}={counts.get(c, 0)} "
        f"({100*counts.get(c, 0)/total:.1f}%)"
        for c in [0, 1, 2]
    )
    print(f"  Class distribution: {dist_str}")

    X          = df[FEATURE_COLS].values
    y          = df["target"].values
    prices_all = df["Close"].values   # for backtest

    # Walk-forward validation with purge gap = HORIZON bars to prevent label leakage
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=HORIZON)

    # ── 3. Initialise accumulators ────────────────────────────────────────────
    all_true:        dict[str, list] = {}
    all_pred:        dict[str, list] = {}
    all_prices:      list            = []
    all_pred_prices: dict[str, list] = {}

    model_objects = _make_models()
    baseline_names = ["baseline_always_up", "baseline_majority", "baseline_momentum"]

    for name in list(model_objects.keys()) + baseline_names:
        all_true[name]        = []
        all_pred[name]        = []
        all_pred_prices[name] = []

    # ── 4. Walk-forward folds ─────────────────────────────────────────────────
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        p_test          = prices_all[test_idx]

        all_prices.extend(p_test.tolist())

        # ── ML models ─────────────────────────────────────────────────────────
        for name, estimator in model_objects.items():
            est = clone(estimator)   # fresh copy each fold
            est.fit(X_train, y_train)

            preds = est.predict(X_test)

            all_true[name].extend(y_test.tolist())
            all_pred[name].extend(preds.tolist())
            all_pred_prices[name].extend(preds.tolist())

        # ── Baseline: always UP ────────────────────────────────────────────────
        up_class   = TARGET_CLASSES["UP"]
        always_up  = np.full(len(y_test), up_class, dtype=int)
        all_true["baseline_always_up"].extend(y_test.tolist())
        all_pred["baseline_always_up"].extend(always_up.tolist())
        all_pred_prices["baseline_always_up"].extend(always_up.tolist())

        # ── Baseline: always majority class ────────────────────────────────────
        majority    = int(pd.Series(y_train).value_counts().idxmax())
        always_maj  = np.full(len(y_test), majority, dtype=int)
        all_true["baseline_majority"].extend(y_test.tolist())
        all_pred["baseline_majority"].extend(always_maj.tolist())
        all_pred_prices["baseline_majority"].extend(always_maj.tolist())

        # ── Baseline: momentum (sign of previous bar's return) ────────────────
        # Predict UP if previous bar return > +0.5%, DOWN if < -0.5%, else NEUTRAL
        ret_prev = X_test[:, 0]   # FEATURE_COLS[0] is 'ret_1'
        momentum = np.full(len(y_test), TARGET_CLASSES["NEUTRAL"], dtype=int)
        momentum[ret_prev > UP_THRESH]  = TARGET_CLASSES["UP"]
        momentum[ret_prev < -DN_THRESH] = TARGET_CLASSES["DOWN"]
        all_true["baseline_momentum"].extend(y_test.tolist())
        all_pred["baseline_momentum"].extend(momentum.tolist())
        all_pred_prices["baseline_momentum"].extend(momentum.tolist())

        print(f"  Fold {fold}/{N_SPLITS}  train={len(train_idx):,}  test={len(test_idx):,}  (purged {HORIZON} bars)")

    # ── 5. Aggregate metrics + comparison table ───────────────────────────────
    asset_metrics:  dict[str, dict] = {}
    asset_backtest: dict[str, dict] = {}
    prices_arr = np.array(all_prices, dtype=float)

    all_model_names = list(model_objects.keys()) + baseline_names

    # Compute metrics first for all models
    for name in all_model_names:
        yt = np.array(all_true[name])
        yp = np.array(all_pred[name])
        asset_metrics[name] = _compute_metrics(yt, yp)
        asset_backtest[name] = _backtest(prices_arr, np.array(all_pred_prices[name]))

    # Pick the serving model using ONE single rule applied during training: highest out-of-sample MCC
    ml_names = list(model_objects.keys())
    best_name = max(ml_names, key=lambda n: asset_metrics[n]["mcc"])
    asset_metrics["_served_model"] = best_name

    # Header with bootstrap CIs and full backtest stats
    w = 24
    print(f"\n  {'Model':<{w}}  {'BalAcc (95% CI)':>20}  {'MCC (95% CI)':>20}  {'MacroF1':>7}  {'Return':>7}  {'Trades':>6}  {'MaxDD':>7}  {'Sharpe':>6}")
    print(f"  {'-'*w}  {'-'*20:>20}  {'-'*20:>20}  {'-'*7:>7}  {'-'*7:>7}  {'-'*6:>6}  {'-'*7:>7}  {'-'*6:>6}")

    for name in all_model_names:
        m  = asset_metrics[name]
        bt = asset_backtest[name]

        marker = "  ← best ML (serving)" if name == best_name else ""
        is_bl  = name.startswith("baseline")
        prefix = "  [BL] " if is_bl else "  [ML] "
        ba_str  = f"{m['balanced_accuracy']:.3f} [{m['balanced_accuracy_ci'][0]:.3f}, {m['balanced_accuracy_ci'][1]:.3f}]"
        mcc_str = f"{m['mcc']:.3f} [{m['mcc_ci'][0]:.3f}, {m['mcc_ci'][1]:.3f}]"
        print(
            f"{prefix}{name:<{w-7}}  "
            f"{ba_str:>20}  "
            f"{mcc_str:>20}  "
            f"{m['macro_f1']:>7.3f}  "
            f"{bt['strategy_return']:>6.3f}x  "
            f"{bt['n_trades']:>6d}  "
            f"{bt['max_drawdown']*100:>6.1f}%  "
            f"{bt['sharpe_ratio']:>6.2f}"
            f"{marker}"
        )

    # ── 6. Honest verdict (CI lower bound vs baseline point estimate) ─────────
    ml_names = list(model_objects.keys())
    bl_names = baseline_names
    best_bl_name = max(bl_names, key=lambda n: asset_metrics[n]["mcc"])
    best_bl_mcc_point = asset_metrics[best_bl_name]["mcc"]

    best_ml_mcc_point = asset_metrics[best_name]["mcc"]
    best_ml_ci_low    = asset_metrics[best_name]["mcc_ci"][0]
    best_ml_ci_high   = asset_metrics[best_name]["mcc_ci"][1]

    best_ml_bt = asset_backtest[best_name]
    strat_ret = best_ml_bt["strategy_return"]
    bh_ret    = best_ml_bt["buyhold_return"]
    n_trades  = best_ml_bt["n_trades"]
    max_dd    = best_ml_bt["max_drawdown"]
    sharpe    = best_ml_bt["sharpe_ratio"]

    print()
    if best_ml_ci_low > best_bl_mcc_point:
        print(
            f"  ✅  Best ML model ({best_name}) BEATS all baselines:\n"
            f"     • 95% CI lower bound ({best_ml_ci_low:.4f}) is strictly above the best baseline point estimate ({best_bl_mcc_point:.4f}, {best_bl_name}).\n"
            f"     • Point MCC: {best_ml_mcc_point:.4f} [95% CI: {best_ml_ci_low:.4f}, {best_ml_ci_high:.4f}]\n"
            f"     • Backtest: {strat_ret:.4f}x ({n_trades} trades, MaxDD: {max_dd*100:.1f}%, Sharpe: {sharpe:.2f}) vs Buy & Hold: {bh_ret:.4f}x"
        )
    elif best_ml_mcc_point > best_bl_mcc_point:
        print(
            f"  ⚠️  RESULT INCONCLUSIVE for {symbol}:\n"
            f"     • Best ML model ({best_name}) point MCC ({best_ml_mcc_point:.4f}) is higher than baseline ({best_bl_mcc_point:.4f}, {best_bl_name}),\n"
            f"       BUT its 95% CI [{best_ml_ci_low:.4f}, {best_ml_ci_high:.4f}] overlaps the baseline.\n"
            f"     • Statistical superiority is not established at the 95% confidence level.\n"
            f"     • Backtest: {strat_ret:.4f}x ({n_trades} trades, MaxDD: {max_dd*100:.1f}%, Sharpe: {sharpe:.2f}) vs Buy & Hold: {bh_ret:.4f}x"
        )
    else:
        print(
            f"  ⚠️  NO ML model beats the baselines for {symbol}:\n"
            f"     • Best ML model ({best_name}) point MCC: {best_ml_mcc_point:.4f} [95% CI: {best_ml_ci_low:.4f}, {best_ml_ci_high:.4f}]\n"
            f"     • Best baseline ({best_bl_name}) point MCC: {best_bl_mcc_point:.4f}\n"
            f"     • Backtest: {strat_ret:.4f}x ({n_trades} trades, MaxDD: {max_dd*100:.1f}%, Sharpe: {sharpe:.2f}) vs Buy & Hold: {bh_ret:.4f}x"
        )


    # ── 7. Calibrate + save best model (highest MCC) ──────────────────────────
    if best_name in model_objects:
        print(f"\n  🔧 Calibrating {best_name} (highest MCC: {asset_metrics[best_name]['mcc']:.4f}) (isotonic, 5-fold CV)…")
        # Calibrate on the full dataset for maximum data use
        calibrated = CalibratedClassifierCV(
            estimator=clone(model_objects[best_name]),
            method="isotonic",
            cv=5,
        )

        calibrated.fit(X, y)

        out_path = MODELS_DIR / f"{symbol.lower()}_model.pkl"
        joblib.dump(calibrated, out_path)
        print(f"  💾 Saved calibrated {best_name} → {out_path}")

        # Save metadata so live inference uses identical config and model name
        meta = {
            "best_model":        best_name,
            "selection_metric":  "highest_mcc",
            "mcc":               asset_metrics[best_name]["mcc"],
            "interval":          CANDLE_INTERVAL,
            "horizon":           HORIZON,
            "up_thresh":         UP_THRESH,
            "dn_thresh":         DN_THRESH,
            "signal_threshold":  SIGNAL_THRESHOLD,
        }
        meta_path = MODELS_DIR / f"{symbol.lower()}_meta.json"
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)
        print(f"  📄 Meta saved  → {meta_path}")

        # Save global feature importance plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            tree_est = getattr(calibrated.calibrated_classifiers_[0], "estimator", model_objects[best_name])
            importances = getattr(tree_est, "feature_importances_", None)
            if importances is not None and len(importances) == len(FEATURE_COLS):
                indices = np.argsort(importances)
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(range(len(indices)), importances[indices], color="#2b5c8f", align="center")
                ax.set_yticks(range(len(indices)))
                ax.set_yticklabels([FEATURE_COLS[i] for i in indices], fontsize=9)
                ax.set_xlabel("Relative Feature Importance", fontsize=10)
                ax.set_title(f"Global Feature Importance — {symbol} ({best_name})", fontsize=12, fontweight="bold")
                ax.grid(axis="x", linestyle="--", alpha=0.5)
                fig.tight_layout()
                plot_path = MODELS_DIR / f"{symbol.lower()}_feature_importance.png"
                fig.savefig(plot_path, dpi=150)
                plt.close(fig)
                print(f"  📊 Feature importance plot saved → {plot_path}")
        except Exception as exc:
            print(f"  ⚠️ Could not save feature importance plot: {exc}")

    print()
    return asset_metrics, asset_backtest



# ─────────────────────────────────────────────────────────────────────────────
# Main entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    all_metrics:  dict[str, dict] = {}
    all_backtests: dict[str, dict] = {}

    for symbol in ASSETS:
        m, bt = _train_asset(symbol)
        if m:
            all_metrics[symbol]   = m
            all_backtests[symbol] = bt

    # ── Save JSON artefacts ────────────────────────────────────────────────────
    metrics_path  = MODELS_DIR / "metrics.json"
    backtest_path = MODELS_DIR / "backtest.json"

    with open(metrics_path, "w") as fh:
        json.dump(all_metrics, fh, indent=2)
    print(f"📄 Metrics  → {metrics_path}")

    with open(backtest_path, "w") as fh:
        json.dump(all_backtests, fh, indent=2)
    print(f"📄 Backtest → {backtest_path}")

    # ── Global summary ─────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  GLOBAL SUMMARY")
    print("═"*60)
    any_ml_winner = False
    for symbol, metrics in all_metrics.items():
        ml_names = [n for n in metrics if not n.startswith("baseline") and not n.startswith("_")]
        bl_names = [n for n in metrics if n.startswith("baseline")]
        best_ml  = max(ml_names, key=lambda n: metrics[n]["mcc"])
        best_bl  = max(bl_names, key=lambda n: metrics[n]["mcc"])

        best_ml_ci_low = metrics[best_ml]["mcc_ci"][0]
        best_bl_point  = metrics[best_bl]["mcc"]

        if best_ml_ci_low > best_bl_point:
            verdict = "✅ BEATS baselines (95% CI lower > baseline)"
            any_ml_winner = True
        elif metrics[best_ml]["mcc"] > best_bl_point:
            verdict = "⚠️ INCONCLUSIVE (CI overlaps baseline)"
        else:
            verdict = "❌ does NOT beat baselines"

        bt = all_backtests[symbol][best_ml]
        bt_strat = bt["strategy_return"]
        bt_bh    = bt["buyhold_return"]
        bt_sharpe = bt["sharpe_ratio"]
        bt_dd    = bt["max_drawdown"]
        print(
            f"  {symbol:<10} best={best_ml:<14} "
            f"MCC={metrics[best_ml]['mcc']:.4f} [95% CI: {metrics[best_ml]['mcc_ci'][0]:.4f}, {metrics[best_ml]['mcc_ci'][1]:.4f}] | "
            f"{verdict}"
        )
        print(
            f"  {'':10} backtest: strategy×{bt_strat:.4f} (Sharpe: {bt_sharpe:.2f}, MaxDD: {bt_dd*100:.1f}%) | "
            f"buy&hold×{bt_bh:.4f}"
        )

    if not any_ml_winner:
        print(
            "\n  ⚠️  Overall: ML models do NOT reliably beat simple baselines.\n"
            "  This is expected on next-bar direction prediction with OHLCV alone.\n"
            "  Consider: longer intraday history, wider return thresholds,\n"
            "  order-book depth, news sentiment, or ensemble stacking."
        )
    print()



if __name__ == "__main__":
    main()

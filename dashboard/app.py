# -------------------------------------------------
# Fix project import path (IMPORTANT)
# -------------------------------------------------
import sys
import os
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) in sys.path:
    sys.path.remove(str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR))

# Evict accidental collision where 'app' was resolved to dashboard/app.py instead of package app/
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

# -------------------------------------------------
# Imports
# -------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timezone

from app.database import get_db_connection

# -------------------------------------------------
# Suppress harmless warnings
# -------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning)

MODELS_DIR = ROOT_DIR / "ml" / "models"

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Market Trend AI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for executive, clean aesthetic
st.markdown("""
<style>
    /* Metric styling to prevent truncation */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: #8b949e !important;
    }
    
    /* Clean Cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }
    .badge-card {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .info-subtext {
        font-size: 0.82rem;
        color: #8b949e;
        margin-top: 4px;
        line-height: 1.4;
    }
    .disclaimer-text {
        font-size: 0.78rem;
        color: #6e7681;
        font-style: italic;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Load metrics & backtest JSON early for badge & evaluation
# -------------------------------------------------
metrics_path = MODELS_DIR / "metrics.json"
backtest_path = MODELS_DIR / "backtest.json"

all_metrics: dict = {}
all_backtests: dict = {}

if metrics_path.exists():
    try:
        with open(metrics_path) as f:
            all_metrics = json.load(f)
    except Exception:
        pass

if backtest_path.exists():
    try:
        with open(backtest_path) as f:
            all_backtests = json.load(f)
    except Exception:
        pass


def get_evidence_badge(sym: str) -> dict:
    """Derive evidence badge and backtest summary dynamically from metrics.json and backtest.json."""
    metrics = all_metrics.get(sym, {})
    backtest = all_backtests.get(sym, {})

    if not metrics:
        return {
            "badge_text": "No metrics available",
            "badge_color": "#8b949e",
            "badge_bg": "#21262d",
            "badge_border": "#30363d",
            "status_icon": "⚪",
            "backtest_summary": "No backtest data recorded",
        }

    ml_names = [n for n in metrics if not n.startswith("baseline") and not n.startswith("_") and isinstance(metrics[n], dict)]
    bl_names = [n for n in metrics if n.startswith("baseline") and isinstance(metrics[n], dict)]

    if not ml_names or not bl_names:
        return {
            "badge_text": "Insufficient comparison data",
            "badge_color": "#8b949e",
            "badge_bg": "#21262d",
            "badge_border": "#30363d",
            "status_icon": "⚪",
            "backtest_summary": "Comparison data unavailable",
        }

    best_ml = max(ml_names, key=lambda n: metrics[n].get("mcc", 0.0))
    best_bl = max(bl_names, key=lambda n: metrics[n].get("mcc", 0.0))

    best_ml_ci_low = metrics[best_ml].get("mcc_ci", [0.0, 0.0])[0]
    best_ml_point = metrics[best_ml].get("mcc", 0.0)
    best_bl_point = metrics[best_bl].get("mcc", 0.0)

    # Dynamic verdict derivation
    if best_ml_ci_low > best_bl_point:
        badge_text = "Statistically significant edge"
        badge_color = "#3fb950"
        badge_bg = "rgba(46, 160, 67, 0.15)"
        badge_border = "rgba(46, 160, 67, 0.4)"
        status_icon = "🟢"
    elif best_ml_point > best_bl_point:
        badge_text = "Inconclusive"
        badge_color = "#d29922"
        badge_bg = "rgba(187, 128, 9, 0.15)"
        badge_border = "rgba(187, 128, 9, 0.4)"
        status_icon = "🟡"
    else:
        badge_text = "No proven edge over baseline"
        badge_color = "#f85149"
        badge_bg = "rgba(248, 81, 73, 0.15)"
        badge_border = "rgba(248, 81, 73, 0.4)"
        status_icon = "🔴"

    bt = backtest.get(best_ml, {})
    strat_ret = bt.get("strategy_return", 1.0)
    bh_ret = bt.get("buyhold_return", 1.0)
    trades = bt.get("n_trades", 0)
    maxdd = bt.get("max_drawdown", 0.0)
    sharpe = bt.get("sharpe_ratio", 0.0)

    backtest_summary = (
        f"{best_ml}: {strat_ret:.3f}x return ({trades} trades, MaxDD: {maxdd*100:.1f}%, Sharpe: {sharpe:.2f}) "
        f"vs Buy & Hold: {bh_ret:.3f}x"
    )

    return {
        "badge_text": badge_text,
        "badge_color": badge_color,
        "badge_bg": badge_bg,
        "badge_border": badge_border,
        "status_icon": status_icon,
        "backtest_summary": backtest_summary,
        "best_ml": best_ml,
    }


# =================================================
# 🕒 TIMEZONE UTILITY (Indian Standard Time - IST)
# =================================================
def to_ist(val):
    """Format a timestamp or Series into Indian Standard Time (IST, UTC+5:30)."""
    if val is None:
        return "N/A"
    if isinstance(val, pd.Series):
        s = pd.to_datetime(val, utc=True)
        return s.dt.tz_convert("Asia/Kolkata").dt.strftime("%Y-%m-%d %H:%M:%S IST")
    try:
        if pd.isna(val):
            return "N/A"
        ts = pd.to_datetime(val, utc=True)
        return ts.tz_convert("Asia/Kolkata").strftime("%Y-%m-%d %H:%M:%S IST")
    except Exception:
        return str(val)


# =================================================
# 🔴 SIDEBAR: CONTROLS & ASSET SELECTOR
# =================================================
st.sidebar.markdown("### 🧭 Navigation & Asset")
symbol = st.sidebar.selectbox(
    "Target Market",
    ["BITCOIN", "ETHEREUM", "IBM"],
    index=0
)



# =================================================
# DATABASE QUERY
# =================================================
try:
    conn = get_db_connection()
    query = """
        SELECT symbol, direction, confidence, candle_timestamp, model_version,
               actual_return, actual_label, was_correct, predicted_at
        FROM prediction_results
        WHERE symbol = %s
        ORDER BY predicted_at DESC
        LIMIT 50
    """
    df = pd.read_sql(query, conn, params=(symbol,))

    # Query latest market price
    cur = conn.cursor()
    cur.execute(
        "SELECT price FROM market_data WHERE symbol = %s ORDER BY recorded_at DESC LIMIT 1",
        (symbol,)
    )
    p_row = cur.fetchone()
    latest_price = float(p_row[0]) if p_row else None
    cur.close()
    conn.close()

except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

# Notice if Predict All was clicked
if "global_notice" in st.session_state:
    st.toast(st.session_state["global_notice"])
    del st.session_state["global_notice"]

# =================================================
# 🏆 TOP HEADER & HERO SECTION
# =================================================
evidence = get_evidence_badge(symbol)

header_c1, header_c2 = st.columns([3, 1])

with header_c1:
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 6px;">
            <h2 style="margin: 0; padding: 0; font-size: 1.85rem;">{symbol} Market Forecast</h2>
            <div class="badge-card" style="background-color: {evidence['badge_bg']}; color: {evidence['badge_color']}; border: 1px solid {evidence['badge_border']};">
                {evidence['status_icon']} {evidence['badge_text']}
            </div>
        </div>
        <div class="info-subtext">
            📊 <strong>Backtest Benchmark:</strong> {evidence['backtest_summary']}
        </div>
        <div class="disclaimer-text">
            ⚠️ <em>For quantitative research only. Not financial advice.</em>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_c2:
    if st.button(f"⚡ Predict Now ({symbol})", key="hero_pred_btn", type="primary", use_container_width=True):
        with st.spinner(f"Ingesting latest candle & running {symbol} model..."):
            try:
                from app.services.candle_ingester import ingest_asset_candles
                from app.services.ml_predictor import predict_symbol
                mtype = "crypto" if symbol in ("BITCOIN", "ETHEREUM") else "stock"
                ingest_asset_candles(symbol, mtype)
                res = predict_symbol(symbol)
                if "error" in res:
                    st.error(res["error"])
                else:
                    st.session_state[f"last_action_{symbol}"] = {
                        "evaluated_at": datetime.now(timezone.utc),
                        "result": res,
                    }
                    st.toast(f"✅ Prediction refreshed for {symbol}!")
                    st.rerun()
            except Exception as exc:
                st.error(f"Prediction run failed: {exc}")

st.write("")

# Check recent manual action feedback
last_action = st.session_state.get(f"last_action_{symbol}")

# =================================================
# 📑 TABBED WORKSPACE (CLEAN & NON-MESSY)
# =================================================
tab_live, tab_track, tab_models = st.tabs([
    "🔮 Live Forecast & Drivers",
    "🎯 Live Outcome Tracker",
    "📊 Model Evidence & Diagnostics"
])

# -------------------------------------------------
# TAB 1: 🔮 LIVE FORECAST & EXPLAINABILITY
# -------------------------------------------------
with tab_live:
    if df.empty:
        st.info(f"No predictions on record for {symbol}. Click **⚡ Predict Now** to generate the first signal.")
    else:
        latest = df.iloc[0]
        direction = str(latest["direction"])
        confidence = float(latest["confidence"])

        # Top row: 3 Primary Metrics with ample room (no ellipsis truncation)
        c_price, c_dir, c_conf = st.columns([1, 1, 1])

        with c_price:
            price_val = f"${latest_price:,.2f}" if latest_price is not None else "N/A"
            st.metric("Live Market Price", price_val)

        with c_dir:
            if direction == "UP":
                st.metric("Predicted Direction", "🟢 UP", help="Expected trend over next 5 hours: > +0.5%")
            elif direction == "DOWN":
                st.metric("Predicted Direction", "🔴 DOWN", help="Expected trend over next 5 hours: < -0.5%")
            elif direction == "NEUTRAL":
                st.metric("Predicted Direction", "⚪ NEUTRAL", help="Expected trend over next 5 hours: within ±0.5%")
            else:
                st.metric("Predicted Direction", "⚠️ NO SIGNAL")

        with c_conf:
            st.metric("Model Confidence", f"{confidence:.1f}%")

        # Bottom row: Timestamps & Evaluation Status in clear metadata banner
        cand_time_str = to_ist(latest.get("candle_timestamp"))
        eval_time_str = to_ist(last_action["evaluated_at"]) if last_action else to_ist(latest.get("predicted_at"))

        st.markdown(
            f"""
            <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px 18px; margin: 12px 0 20px 0; display: flex; flex-wrap: wrap; justify-content: space-between; gap: 16px;">
                <div>
                    <span style="color: #8b949e; font-size: 0.8rem; font-weight: 500;">LAST CLOSED CANDLE (IST)</span><br>
                    <span style="color: #f0f6fc; font-size: 0.95rem; font-weight: 600;">{cand_time_str}</span>
                </div>
                <div>
                    <span style="color: #8b949e; font-size: 0.8rem; font-weight: 500;">PREDICTION RUN AT (IST)</span><br>
                    <span style="color: #f0f6fc; font-size: 0.95rem; font-weight: 600;">{eval_time_str}</span>
                </div>
                <div>
                    <span style="color: #8b949e; font-size: 0.8rem; font-weight: 500;">MODEL HORIZON</span><br>
                    <span style="color: #58a6ff; font-size: 0.95rem; font-weight: 600;">5 Bars (5 Hours)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("💡 Why does the prediction stay the same between hour marks?", expanded=False):
            st.markdown(
                """
                - Predictions are calculated strictly from **fully closed 1-hour candles** to eliminate mid-candle noise and false breakouts.
                - Between hourly boundaries (e.g. 10:00 to 11:00), the most recent closed candle is identical, so the ML model consistently returns the exact same signal until the current 1-hour candle completes.
                - When a new hourly candle closes, clicking **⚡ Predict Now** ingests the fresh candle and updates the prediction.
                """
            )

        st.markdown("---")

        # -------------------------------------------------
        # SHAP EXPLAINABILITY SECTION
        # -------------------------------------------------
        st.subheader(f"🔍 Top Feature Drivers (SHAP Explainability) — {symbol}")
        st.caption(
            f"SHAP TreeExplainer breakdown illustrating how each technical indicator pushes the model "
            f"toward or against predicted direction **{direction}** ({confidence:.1f}% confidence)."
        )

        try:
            from app.services.explainer import explain_latest_prediction
            exp_res = explain_latest_prediction(symbol, top_k=5)

            if "error" in exp_res:
                st.info(f"💡 {exp_res['error']}")
            else:
                top_drivers = exp_res.get("top_drivers", [])
                pred_cls = exp_res.get("predicted_class", direction)

                def format_feature_val(feat: str, v: float) -> str:
                    if v is None or pd.isna(v):
                        return "N/A"
                    if abs(v) < 1e-6:
                        v = 0.0
                    if feat.startswith("ret_"):
                        return f"{v * 100:+.2f}%"
                    if feat.startswith("vol_") and feat != "vol_change":
                        return f"{v * 100:.2f}%"
                    if feat.startswith("close_vs_ma"):
                        return f"{v * 100:+.2f}%"
                    if feat == "vol_change":
                        if -5.0 <= v <= 5.0:
                            pct = (np.exp(v) - 1.0) * 100.0
                            return f"{pct:+.1f}%"
                        return f"{v:+.2f}"
                    if feat.startswith("rsi_"):
                        return f"{v:.1f}"
                    if feat == "macd_hist":
                        return f"{v:+.3f}"
                    return f"{v:.4g}"

                exp_col1, exp_col2 = st.columns([3, 2])

                with exp_col1:
                    chart_rows = []
                    for d in top_drivers:
                        chart_rows.append({
                            "Feature": d["feature"],
                            "SHAP Value": d["shap_value"],
                            "Value": d["feature_value"],
                            "DisplayVal": format_feature_val(d["feature"], d["feature_value"]),
                        })
                    ch_df = pd.DataFrame(chart_rows)

                    # Dark-themed matplotlib chart
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    plt.style.use("dark_background")
                    fig, ax = plt.subplots(figsize=(6.8, 3.2), facecolor="#0e1117")
                    ax.set_facecolor("#0e1117")

                    colors = ["#2ea043" if v > 0 else "#f85149" for v in ch_df["SHAP Value"]]
                    y_pos = range(len(ch_df))
                    ax.barh(y_pos, ch_df["SHAP Value"], color=colors, height=0.6, align="center")
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(
                        [f"{r['Feature']} ({r['DisplayVal']})" for _, r in ch_df.iterrows()],
                        fontsize=9,
                        color="#c9d1d9"
                    )
                    ax.axvline(0, color="#484f58", linestyle="--", linewidth=1.0)
                    ax.set_xlabel(f"SHAP Impact toward {pred_cls}", fontsize=9, color="#8b949e")
                    ax.tick_params(axis="x", colors="#8b949e", labelsize=8)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.spines["left"].set_color("#30363d")
                    ax.spines["bottom"].set_color("#30363d")
                    ax.invert_yaxis()
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                with exp_col2:
                    st.markdown(f"**Impact Breakdown on `{pred_cls}`:**")
                    for d in top_drivers:
                        is_pos = d["shap_value"] > 0
                        icon = "🟢 ⬆️" if is_pos else "🔴 ⬇️"
                        push_action = "toward" if is_pos else "against"
                        d_val_str = format_feature_val(d["feature"], d["feature_value"])
                        st.markdown(
                            f"""
                            <div style="background: #161b22; border-left: 3px solid {'#2ea043' if is_pos else '#f85149'}; padding: 6px 12px; margin-bottom: 6px; border-radius: 4px;">
                                <span style="font-size: 0.88rem; font-weight: 600; color: #f0f6fc;">{icon} <code>{d['feature']}</code></span>
                                <span style="font-size: 0.8rem; color: #8b949e; margin-left: 6px;">(val: <strong>{d_val_str}</strong>)</span><br>
                                <span style="font-size: 0.82rem; color: {'#3fb950' if is_pos else '#f85149'}; font-weight: 600;">{d['shap_value']:+.4f}</span>
                                <span style="font-size: 0.8rem; color: #8b949e;"> — pushes <strong>{push_action}</strong> {pred_cls}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

        except Exception as exc:
            st.warning(f"SHAP explanation unavailable: {exc}")


# -------------------------------------------------
# TAB 2: 🎯 LIVE OUTCOME TRACKER
# -------------------------------------------------
with tab_track:
    st.subheader(f"🎯 Live Prediction Track Record — {symbol}")
    st.caption("Tracks the real-world accuracy of predictions after the 5-hour horizon has elapsed.")

    try:
        from app.services.outcome_tracker import get_live_performance
        perf = get_live_performance(symbol)
    except Exception as exc:
        perf = {"resolved_count": 0, "has_enough_data": False, "message": str(exc)}

    lp_col1, lp_col2, lp_col3 = st.columns(3)

    resolved_count = perf.get("resolved_count", 0)
    lp_col1.metric("Resolved Predictions", f"{resolved_count} / 30")

    if perf.get("has_enough_data", False):
        lp_col2.metric("Live Accuracy", f"{perf.get('live_accuracy_pct', 0):.1f}%")
        neutral_pct = perf.get('always_neutral_accuracy_pct', 0)
        diff_pct = perf.get('accuracy_difference_pct', 0)
        lp_col3.metric("Always NEUTRAL Baseline", f"{neutral_pct:.1f}%", delta=f"{diff_pct:+.1f}%")
    else:
        lp_col2.metric("Live Accuracy", "Not enough data yet")
        lp_col3.metric("Always NEUTRAL Baseline", "Not enough data yet")

    if not perf.get("has_enough_data", False):
        st.info(
            f"ℹ️ **Statistical Threshold**: {resolved_count} of 30 required predictions have completed their 5-hour horizon "
            f"({perf.get('unresolved_count', 0)} predictions currently maturing). "
            "Live percentage metrics unlock once 30 predictions are resolved."
        )

    st.markdown("---")
    st.subheader("📜 Recent Prediction Outcomes")

    if not df.empty:
        disp_df = df.copy()
        if "predicted_at" in disp_df.columns:
            disp_df["predicted_at (IST)"] = to_ist(disp_df["predicted_at"])
        if "candle_timestamp" in disp_df.columns:
            disp_df["candle_timestamp (IST)"] = to_ist(disp_df["candle_timestamp"])
        if "actual_return" in disp_df.columns:
            disp_df["Actual Return (5h)"] = disp_df["actual_return"].apply(
                lambda x: f"{float(x)*100:+.2f}%" if pd.notna(x) and x is not None else "⏳ Maturing"
            )
        if "actual_label" in disp_df.columns:
            disp_df["Actual Label"] = disp_df["actual_label"].fillna("⏳ Maturing")
        if "was_correct" in disp_df.columns:
            disp_df["Outcome"] = disp_df["was_correct"].apply(
                lambda x: "✅ Correct" if x is True else ("❌ Incorrect" if x is False else "⏳ Maturing")
            )

        preferred_cols = [
            "symbol", "direction", "confidence",
            "Actual Return (5h)", "Actual Label", "Outcome",
            "candle_timestamp (IST)", "predicted_at (IST)"
        ]
        show_cols = [c for c in preferred_cols if c in disp_df.columns]
        st.dataframe(disp_df[show_cols], use_container_width=True)
    else:
        st.caption("No historical predictions recorded in database.")

    if not df.empty and len(df) > 1:
        st.markdown("---")
        st.subheader("📈 Prediction Confidence Over Time")
        chart_df = df.sort_values("predicted_at").copy()
        chart_df["Time (IST)"] = to_ist(chart_df["predicted_at"])
        st.line_chart(
            chart_df.set_index("Time (IST)")["confidence"]
        )


# -------------------------------------------------
# TAB 3: 📊 MODEL EVIDENCE & DIAGNOSTICS
# -------------------------------------------------
with tab_models:
    st.subheader(f"📊 Statistical Evidence & Offline Benchmarks — {symbol}")

    asset_metrics = all_metrics.get(symbol, {})
    asset_backtest = all_backtests.get(symbol, {})
    meta_path = MODELS_DIR / f"{symbol.lower()}_meta.json"

    if not asset_metrics:
        st.info("No offline training metrics found. Run offline training: `python -m ml.train_offline`.")
    else:
        # Build comparison table
        table_rows = []
        ml_models = []
        baseline_models = []

        for model_name, m in asset_metrics.items():
            if model_name.startswith("_") or not isinstance(m, dict):
                continue
            bt = asset_backtest.get(model_name, {})
            ba_ci = m.get("balanced_accuracy_ci", [0.0, 0.0])
            mcc_ci = m.get("mcc_ci", [0.0, 0.0])

            row = {
                "Model": model_name,
                "Type": "Baseline" if model_name.startswith("baseline") else "ML Model",
                "Balanced Accuracy (95% CI)": f"{m.get('balanced_accuracy', 0.0):.3f} [{ba_ci[0]:.3f}, {ba_ci[1]:.3f}]",
                "MCC (95% CI)": f"{m.get('mcc', 0.0):.3f} [{mcc_ci[0]:.3f}, {mcc_ci[1]:.3f}]",
                "Macro F1": round(m.get("macro_f1", 0.0), 3),
                "Move vs No-Move Acc": f"{m.get('move_vs_nomove_accuracy', 0.0)*100:.1f}%" if "move_vs_nomove_accuracy" in m else "N/A",
                "Strategy Return": f"{bt.get('strategy_return', 1.0):.3f}x",
                "Buy & Hold Return": f"{bt.get('buyhold_return', 1.0):.3f}x",
                "Trades": bt.get("n_trades", 0),
                "Max Drawdown": f"{bt.get('max_drawdown', 0.0)*100:.1f}%",
                "Sharpe": round(bt.get("sharpe_ratio", 0.0), 2),
                "_mcc_point": m.get("mcc", 0.0),
                "_mcc_ci_low": mcc_ci[0],
                "_mcc_ci_high": mcc_ci[1],
                "_strat_ret": bt.get("strategy_return", 1.0),
                "_bh_ret": bt.get("buyhold_return", 1.0),
                "_trades": bt.get("n_trades", 0),
                "_maxdd": bt.get("max_drawdown", 0.0),
                "_sharpe": bt.get("sharpe_ratio", 0.0),
            }
            table_rows.append(row)
            if model_name.startswith("baseline"):
                baseline_models.append(row)
            else:
                ml_models.append(row)

        comp_df = pd.DataFrame(table_rows)

        # Highlight best ML model vs best Baseline
        best_ml = max(ml_models, key=lambda x: x["_mcc_point"]) if ml_models else None
        best_bl = max(baseline_models, key=lambda x: x["_mcc_point"]) if baseline_models else None

        if best_ml and best_bl:
            ci_low = best_ml["_mcc_ci_low"]
            ci_high = best_ml["_mcc_ci_high"]
            bl_point = best_bl["_mcc_point"]
            strat_ret = best_ml["_strat_ret"]
            bh_ret = best_ml["_bh_ret"]
            sharpe = best_ml["_sharpe"]
            maxdd = best_ml["_maxdd"]
            trades = best_ml["_trades"]

            if ci_low > bl_point:
                st.success(
                    f"**Verdict: ✅ Best ML model ({best_ml['Model']}) BEATS baselines!**\n\n"
                    f"• 95% CI lower bound (`{ci_low:.4f}`) strictly exceeds best baseline point estimate (`{bl_point:.4f}`, {best_bl['Model']}).\n"
                    f"• Point MCC: `{best_ml['_mcc_point']:.4f}` [95% CI: `{ci_low:.4f}`, `{ci_high:.4f}`].\n"
                    f"• **Backtest:** Strategy return: **{strat_ret:.4f}x** ({trades} trades, MaxDD: {maxdd*100:.1f}%, Sharpe: {sharpe:.2f}) vs Buy & Hold: **{bh_ret:.4f}x**."
                )
            elif best_ml["_mcc_point"] > bl_point:
                st.warning(
                    f"**Verdict: ⚠️ INCONCLUSIVE for {symbol}.**\n\n"
                    f"• Best ML model (**{best_ml['Model']}**) has a higher point MCC (`{best_ml['_mcc_point']:.4f}`) than best baseline (**{best_bl['Model']}**, `{bl_point:.4f}`),\n"
                    f"  **BUT** its 95% CI `[{ci_low:.4f}, {ci_high:.4f}]` overlaps the baseline. Superiority is not statistically established at 95% confidence.\n"
                    f"• **Backtest:** Strategy return: **{strat_ret:.4f}x** ({trades} trades, MaxDD: {maxdd*100:.1f}%, Sharpe: {sharpe:.2f}) vs Buy & Hold: **{bh_ret:.4f}x**."
                )
            else:
                st.error(
                    f"**Verdict: ❌ NO ML model beats simple baselines for {symbol}.**\n\n"
                    f"• Best baseline (**{best_bl['Model']}**) point MCC: `{bl_point:.4f}` outperforms best ML model (**{best_ml['Model']}**, `{best_ml['_mcc_point']:.4f}`).\n"
                    f"• **Backtest:** Strategy return: **{strat_ret:.4f}x** vs Buy & Hold: **{bh_ret:.4f}x**."
                )

        st.markdown("#### Full Cross-Validation & Baseline Comparison")
        cols_to_drop = [c for c in comp_df.columns if c.startswith("_")]
        disp_df = comp_df.drop(columns=cols_to_drop)
        st.dataframe(disp_df, use_container_width=True)

        st.markdown("---")

        # ── Served Model Diagnostics (Confusion Matrix & Move Accuracy) ──────────
        served_model_name = asset_metrics.get("_served_model") or (best_ml["Model"] if best_ml else None)
        if served_model_name and served_model_name in asset_metrics:
            sm_data = asset_metrics[served_model_name]
            cm = sm_data.get("confusion_matrix")
            move_acc = sm_data.get("move_vs_nomove_accuracy")

            st.markdown(f"#### 🎯 Served Model Diagnostics: `{served_model_name}`")
            d_col1, d_col2 = st.columns([1, 2])

            with d_col1:
                if move_acc is not None:
                    st.metric(
                        label="Move vs No-Move Accuracy",
                        value=f"{move_acc*100:.1f}%",
                        help="Binary accuracy distinguishing market trending moves (UP/DOWN) from flat consolidation (NEUTRAL)."
                    )
                st.markdown(
                    f"""
                    <div style="background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px; margin-top: 10px;">
                        <span style="font-size: 0.85rem; color: #8b949e;"><strong>Selection Rule:</strong> Highest out-of-sample MCC (<code>{sm_data.get('mcc', 0.0):.4f}</code>).</span><br>
                        <span style="font-size: 0.8rem; color: #6e7681;">Identical model is saved to <code>{symbol.lower()}_model.pkl</code> and served for live inference.</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with d_col2:
                if cm:
                    st.markdown("**Out-of-Sample Confusion Matrix (3x3):**")
                    labels = sm_data.get("confusion_labels", ["DOWN", "NEUTRAL", "UP"])
                    cm_df = pd.DataFrame(
                        cm,
                        index=[f"Actual {l}" for l in labels],
                        columns=[f"Pred {l}" for l in labels]
                    )
                    st.dataframe(cm_df, use_container_width=True)

        # ── Global Feature Importance (training time) ────────────────────────
        glob_img_path = MODELS_DIR / f"{symbol.lower()}_feature_importance.png"
        if glob_img_path.exists():
            st.markdown("---")
            with st.expander(f"🌐 View Global Feature Importance across Training History ({symbol})", expanded=False):
                st.image(str(glob_img_path), caption=f"Global Feature Importance for {symbol} (generated at training time)", use_container_width=True)

        # Metadata config
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                st.caption(
                    f"⚙️ **Trained Config:** Interval: `{meta.get('interval')}`, "
                    f"Horizon: `{meta.get('horizon')} bars`, "
                    f"Purge gap: `{meta.get('horizon')} bars`, "
                    f"UP threshold: `+{meta.get('up_thresh')*100:.1f}%`, "
                    f"DOWN threshold: `-{meta.get('dn_thresh')*100:.1f}%`, "
                    f"Signal threshold: `{meta.get('signal_threshold')*100:.0f}%`"
                )
            except Exception:
                pass

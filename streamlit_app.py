import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math
from datetime import timedelta
import matplotlib.pyplot as plt

############################################################
# CONFIG
############################################################
MODEL_PATH = "models/lgbm_weekly_tuned.pkl"          # Git LFS
FEATURE_BUFFER = "data/features_weekly.parquet"      # 13-week history
PLOT_COLORS = {"history": "#1f77b4", "forecast": "#d62728"}
MAX_LAG = 12   # highest lag referenced

############################################################
# LOAD MODEL & BUFFER (cached once per session)
############################################################
@st.cache_resource(show_spinner="Loading model …")
def load_model_and_buffer():
    model = joblib.load(MODEL_PATH)  # .pkl contains only the model

    df = pd.read_parquet(FEATURE_BUFFER)
    df = df.sort_values("week_ending")

    # Keep only last MAX_LAG+1 weeks per commodity (for lags and rolls)
    buffer = (
        df.groupby("commodity")
          .tail(MAX_LAG + 1)
          .set_index(["commodity", "week_ending"] )
    )

    # Derive feature list (all model inputs)
    drop_cols = {"commodity", "week_ending", "price_gbp_kg"}
    feat_cols = [c for c in df.columns if c not in drop_cols]
    return model, feat_cols, buffer

model, FEAT_COLS, BUFFER = load_model_and_buffer()

############################################################
# UTILITY HELPERS
############################################################

def safe_lag(series: pd.Series, k: int):
    """Return value k steps back; if not enough history, use earliest."""
    return series.iloc[-k] if len(series) >= k else series.iloc[0]


def safe_tail_sum(series: pd.Series, w: int):
    """Sum of last w items; if shorter, sum whole series."""
    return series.tail(w).sum() if len(series) >= w else series.sum()

############################################################
# FEATURE BUILDER
############################################################

def build_feature_row(hist: pd.DataFrame) -> pd.Series:
    """Build one feature row from the most-recent history."""
    row = {}
    # price lags
    for k in (1, 2, 4, 8, 12):
        row[f"price_lag_{k}"] = safe_lag(hist.price_gbp_kg, k)

    # rolling price
    row["price_roll_4"] = hist.price_gbp_kg.tail(4).mean()

    # weather aggregates
    for w in (4, 8):
        row[f"rain_sum_{w}"] = safe_tail_sum(hist.rain_sum, w)
        row[f"sun_sum_{w}"]  = safe_tail_sum(hist.sun_sum, w)

    # last week's weather
    row["tmax_mean"] = hist.tmax_mean.iloc[-1]
    row["tmin_mean"] = hist.tmin_mean.iloc[-1]

    # calendar features
    # get the actual date value from index
    if isinstance(hist.index, pd.MultiIndex):
        last_date_val = hist.index.get_level_values(-1)[-1]
    else:
        last_date_val = hist.index[-1]
    # convert to pandas Timestamp
    last_date_ts = pd.to_datetime(last_date_val)
    week_no = last_date_ts.isocalendar().week
    row["week_num"] = week_no
    row["sin_week"] = math.sin(2 * math.pi * week_no / 52)
    row["cos_week"] = math.cos(2 * math.pi * week_no / 52)
    row["month"] = last_date_ts.month

    # macro & holiday (carry forward last known)
    for col in ("fx_usd_gbp", "brent_usd_bbl", "is_holiday"):  
        row[col] = hist[col].iloc[-1]

    # ensure order matches training
    return pd.Series(row)[FEAT_COLS]

############################################################
# FORECASTING & PLOTTING
############################################################

def forecast_prices(veg: str, horizon: int) -> list[float]:
    """Generate forecasts for `horizon` weeks ahead for one veg."""
    veg = veg.upper()
    if veg not in BUFFER.index.get_level_values(0):
        st.error(f"Commodity '{veg}' not found in buffer.")
        st.stop()

    hist = BUFFER.xs(veg).copy()
    preds = []
    for _ in range(horizon):
        feats = build_feature_row(hist)
        log_pred = model.predict(feats.to_frame().T)[0]
        price = round(math.exp(log_pred), 3)
        preds.append(price)

        # append pseudo-row for next iteration
        next_week = hist.index[-1] + timedelta(days=7)
        pseudo = feats.to_frame().T.assign(price_gbp_kg=price)
        pseudo.index = [next_week]
        hist = pd.concat([hist, pseudo]).tail(MAX_LAG + 1)
    return preds


def plot_history_and_forecast(hist: pd.DataFrame, preds: list[float]):
    plt.figure(figsize=(9, 4))
    plt.plot(hist.index, hist.price_gbp_kg, label="History", color=PLOT_COLORS["history"])
    future_idx = [hist.index[-1] + timedelta(days=7 * (i + 1)) for i in range(len(preds))]
    plt.plot(future_idx, preds, label="Forecast", color=PLOT_COLORS["forecast"], marker="o")
    plt.xlabel("Week ending")
    plt.ylabel("Price (£/kg)")
    plt.legend()
    st.pyplot(plt.gcf())

############################################################
# STREAMLIT UI
############################################################

st.title("🇬🇧 UK Vegetable Price Forecaster (all-in-one)")

veg_options = sorted(BUFFER.index.get_level_values(0).unique())
veg = st.selectbox("Select vegetable", veg_options)
horizon = st.slider("Forecast horizon (weeks)", 1, 4, 1)

if st.button("Forecast"):
    history = BUFFER.xs(veg)
    predictions = forecast_prices(veg, horizon)
    st.subheader(f"Next {horizon}-week forecast for {veg.title()}")
    st.write({f"Week +{i+1}": p for i, p in enumerate(predictions)})
    plot_history_and_forecast(history, predictions)

st.markdown("---")
st.caption("Model: LightGBM tuned, log-target. Data window: Jun-2018 → Dec-2024")

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
MODEL_PATH = "models/lgbm_weekly_tuned.pkl"          # tracked with Git LFS
FEATURE_BUFFER = "data/features_weekly.parquet"      # 13‑week history used for lags
PLOT_COLORS = {
    "history": "#1f77b4",  # blue
    "forecast": "#d62728",  # red
}

############################################################
# LOAD MODEL & BUFFER (cached once per session)
############################################################
@st.cache_resource(show_spinner="Loading model …")
def load_model_and_buffer():
    model = joblib.load(MODEL_PATH)           # one object only
    df = pd.read_parquet(FEATURE_BUFFER)

    df = df.sort_values("week_ending")
    buffer = (df.groupby("commodity")
                .tail(13)
                .set_index(["commodity", "week_ending"]))

    # Derive feature list: everything except ID/target columns
    FEAT_EXCLUDE = {"commodity", "week_ending", "price_gbp_kg"}
    feat_cols = [c for c in df.columns if c not in FEAT_EXCLUDE]

    return model, feat_cols, buffer


model, FEAT_COLS, BUFFER = load_model_and_buffer()

############################################################
# UTILS
############################################################

def _week_features(hist: pd.DataFrame) -> pd.Series:
    """Build one feature row from the most‑recent history (expects 13 rows)."""
    row = {}
    # price lags
    for k in (1, 2, 4, 8, 12):
        row[f"price_lag_{k}"] = hist.price_gbp_kg.iloc[-k]
    # rolling
    row["price_roll_4"] = hist.price_gbp_kg.tail(4).mean()
    for w in (4, 8):
        row[f"rain_sum_{w}"] = hist.rain_sum.tail(w).sum()
        row[f"sun_sum_{w}"] = hist.sun_sum.tail(w).sum()
    # weather last week
    row["tmax_mean"] = hist.tmax_mean.iloc[-1]
    row["tmin_mean"] = hist.tmin_mean.iloc[-1]
    # calendar
    week_no = hist.index.get_level_values(1)[-1].isocalendar().week
    row["week_num"] = week_no
    row["sin_week"] = math.sin(2 * math.pi * week_no / 52)
    row["cos_week"] = math.cos(2 * math.pi * week_no / 52)
    row["month"] = hist.index.get_level_values(1)[-1].month
    # macro + holiday from last week
    for col in ("fx_usd_gbp", "brent_usd_bbl", "is_holiday"):
        row[col] = hist[col].iloc[-1]
    return pd.Series(row)[FEAT_COLS]


def forecast_prices(veg: str, horizon: int) -> list[float]:
    """Generate forecasts for `horizon` weeks ahead for one veg."""
    veg = veg.upper()
    if veg not in BUFFER.index.get_level_values(0):
        st.error(f"Commodity '{veg}' not found in buffer.")
        st.stop()
    hist = BUFFER.xs(veg)
    preds = []
    for _ in range(horizon):
        feats = _week_features(hist)
        log_pred = model.predict(feats.values.reshape(1, -1))[0]
        price = round(math.exp(log_pred), 3)
        preds.append(price)
        # append pseudo row to hist to keep lags rolling
        next_wk = hist.index[-1] + timedelta(days=7)
        new_row = feats.to_frame().T.assign(week_ending=next_wk, price_gbp_kg=price)
        new_row = new_row.set_index("week_ending")
        hist = pd.concat([hist, new_row]).tail(13)
    return preds


def plot_history_and_forecast(hist: pd.DataFrame, preds: list[float]):
    plt.figure(figsize=(9, 4))
    plt.plot(hist.index, hist.price_gbp_kg, label="History", color=PLOT_COLORS["history"])
    future_idx = [hist.index[-1] + timedelta(days=7 * (i + 1)) for i in range(len(preds))]
    plt.plot(future_idx, preds, label="Forecast", color=PLOT_COLORS["forecast"], marker="o")
    plt.xlabel("Week ending")
    plt.ylabel("Price (₤/kg)")
    plt.legend()
    st.pyplot(plt.gcf())

############################################################
# STREAMLIT UI
############################################################

st.title("🇬🇧 UK Vegetable Price Forecaster (all‑in‑one)")

veg_options = sorted(BUFFER.index.get_level_values(0).unique())
veg = st.selectbox("Select vegetable", veg_options)
horizon = st.slider("Forecast horizon (weeks)", 1, 4, 1)

if st.button("Forecast"):
    hist = BUFFER.xs(veg)
    preds = forecast_prices(veg, horizon)
    st.subheader(f"Next {horizon}‑week forecast for {veg.title()}")
    st.write({f"Week +{i+1}": p for i, p in enumerate(preds)})
    plot_history_and_forecast(hist, preds)

st.markdown("---")
st.caption("Model: LightGBM tuned, log‑target.  Data window: Jun‑2018 → Dec‑2024")

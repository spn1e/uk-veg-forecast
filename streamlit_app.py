import streamlit as st
import pandas as pd
import numpy as np
import joblib, math
from datetime import timedelta
import altair as alt

# ───── CONFIG ─────
MODEL_PATH  = "models/lgbm_weekly_tuned.pkl"
BUFFER_PATH = "data/features_weekly.parquet"
MAX_LAG     = 12
COLOR_HIST, COLOR_FORE = "#1f77b4", "#d62728"

# ───── LOAD ─────
@st.cache_resource(show_spinner="Loading model & buffer…")
def load_assets():
    model = joblib.load(MODEL_PATH)

    df = pd.read_parquet(BUFFER_PATH)
    df["week_ending"] = pd.to_datetime(df["week_ending"])
    df = df.sort_values(["commodity", "week_ending"])

    buffer = (df.groupby("commodity")
                .tail(MAX_LAG + 1)
                .set_index(["commodity", "week_ending"]))

    feat_cols = model.feature_name_  # saved when model trained
    return model, list(feat_cols), buffer

model, FEAT_COLS, BUFFER = load_assets()

# ───── HELPERS ─────
def safe_lag(s, k):  return s.iloc[-k] if len(s) >= k else s.iloc[0]
def safe_sum(s, w):  return s.tail(w).sum() if len(s) >= w else s.sum()
def safe_mean(s,w):  return s.tail(w).mean() if len(s) >= w else s.mean()

def build_features(hist: pd.DataFrame) -> pd.Series:
    last = hist.index.get_level_values(-1)[-1]
    last = pd.to_datetime(last)
    wk   = int(last.isocalendar().week)

    row = {
        "tmax_mean":  hist.tmax_mean.iloc[-1],
        "tmin_mean":  hist.tmin_mean.iloc[-1],
        "rain_sum":   hist.rain_sum.iloc[-1],
        "sun_sum":    hist.sun_sum.iloc[-1],
        "rain_sum_4": safe_sum(hist.rain_sum, 4),
        "sun_sum_4":  safe_sum(hist.sun_sum, 4),
        "rain_sum_8": safe_sum(hist.rain_sum, 8),
        "sun_sum_8":  safe_sum(hist.sun_sum, 8),
        "price_roll_4": safe_mean(hist.price_gbp_kg, 4),
        "week_num": wk,
        "month":    int(last.month),
        "sin_week": math.sin(2*math.pi*wk/52),
        "cos_week": math.cos(2*math.pi*wk/52),
        "fx_usd_gbp":   hist.fx_usd_gbp.iloc[-1],
        "brent_usd_bbl":hist.brent_usd_bbl.iloc[-1],
        "is_holiday": int(hist.is_holiday.iloc[-1]),
    }
    for k in (1,2,4,8,12):
        row[f"price_lag_{k}"] = safe_lag(hist.price_gbp_kg, k)

    return pd.Series(row).reindex(FEAT_COLS, fill_value=np.nan)

def forecast(veg, horizon):
    hist  = BUFFER.xs(veg.upper()).copy()
    preds = []
    for _ in range(horizon):
        feats = build_features(hist)
        log_p = model.predict(feats.to_frame().T, validate_features=False)[0]
        price = round(math.exp(log_p), 3)
        preds.append(price)

        next_wk = hist.index[-1] + timedelta(days=7)
        pseudo  = hist.iloc[-1:].copy()
        pseudo.index = [next_wk]
        pseudo.price_gbp_kg = price
        hist = pd.concat([hist, pseudo]).tail(MAX_LAG+1)
    return preds, hist

# ───── UI ─────
st.title("🇬🇧 UK Vegetable Price Forecaster")

veg = st.selectbox("Choose a vegetable", sorted(BUFFER.index.get_level_values(0).unique()))
horizon = st.slider("Forecast horizon (weeks)", 1, 12, 4)

if st.button("Generate forecast"):
    preds, hist = forecast(veg, horizon)
    st.json({f"week+{i+1}": p for i,p in enumerate(preds)})

    hist_df = hist.reset_index(names=["commodity","week_ending"])
    fut_df  = pd.DataFrame({
        "week_ending": [hist.index[-1] + timedelta(days=7*(i+1)) for i in range(horizon)],
        "price_gbp_kg": preds,
        "type": "Forecast"
    })
    hist_df = hist_df[["week_ending","price_gbp_kg"]].assign(type="History")
    plot_df = pd.concat([hist_df.tail(20), fut_df])

    alt_chart = alt.Chart(plot_df).mark_line(point=True).encode(
        x="week_ending:T",
        y="price_gbp_kg:Q",
        color=alt.Color("type", scale=alt.Scale(domain=["History","Forecast"],
                                                range=[COLOR_HIST, COLOR_FORE]))
    ).properties(width=700, height=400)
    st.altair_chart(alt_chart, use_container_width=True)

st.caption("LightGBM log‑price model · data Jun‑2018 → Dec‑2024")

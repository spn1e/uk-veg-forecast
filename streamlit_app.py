import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math
from datetime import timedelta
import matplotlib.pyplot as plt

############################
# --- CONFIG ---
############################
MODEL_PATH = "models/lgbm_weekly_tuned.pkl"   # committed to repo via Git LFS
FEATURE_TABLE_PATH = "data/features_weekly.parquet"  # 13‑week buffer

############################
# --- UTILITIES ---
############################
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model, feat_cols = joblib.load(MODEL_PATH)
    buf = pd.read_parquet(FEATURE_TABLE_PATH)
    # keep most‑recent 13 weeks per commodity
    buf = (buf.sort_values("week_ending")
              .groupby("commodity").tail(13)
              .set_index(["commodity","week_ending"]))
    return model, feat_cols, buf

model, feat_cols, buffer = load_artifacts()


def build_next_features(hist: pd.DataFrame) -> pd.Series:
    row = {}
    for k in [1,2,4,8,12]:
        row[f"price_lag_{k}"] = hist.price_gbp_kg.iloc[-k]
    for w in [4,8]:
        row[f"rain_sum_{w}"] = hist.rain_sum.tail(w).sum()
        row[f"sun_sum_{w}"] = hist.sun_sum.tail(w).sum()
    row["price_roll_4"] = hist.price_gbp_kg.tail(4).mean()
    row["tmax_mean"] = hist.tmax_mean.iloc[-1]
    row["tmin_mean"] = hist.tmin_mean.iloc[-1]
    # calendar
    week_no = hist.index.get_level_values(1)[-1].isocalendar().week
    row["week_num"] = week_no
    row["sin_week"] = math.sin(2*math.pi*week_no/52)
    row["cos_week"] = math.cos(2*math.pi*week_no/52)
    row["month"] = hist.index.get_level_values(1)[-1].month
    # macro & holiday
    for col in ["fx_usd_gbp","brent_usd_bbl","is_holiday"]:
        row[col] = hist[col].iloc[-1]
    return pd.Series(row)[feat_cols]


def multi_week_forecast(veg: str, horizon: int = 1):
    veg = veg.upper()
    if (veg,) not in buffer.index.droplevel(1).unique():
        st.error(f"{veg} not in feature table")
        return []
    hist = buffer.xs(veg)
    preds, dates = [], []
    for _ in range(horizon):
        feats = build_next_features(hist)
        log_pred = model.predict(feats.values.reshape(1,-1))[0]
        price = math.exp(log_pred)
        next_date = hist.index[-1] + timedelta(days=7)
        # append pseudo row
        new_row = feats.to_frame().T.assign(price_gbp_kg=price)
        new_row.index = pd.MultiIndex.from_product([[veg],[next_date]],
                                                   names=["commodity","week_ending"])
        hist = pd.concat([hist,new_row]).tail(13)
        preds.append(price)
        dates.append(next_date)
    return dates, preds

############################
# --- STREAMLIT UI ---
############################
st.set_page_config(page_title="UK Veg Price Forecaster", layout="wide")
st.title("🥕 UK Vegetable Price Forecaster")

left, right = st.columns([2,1], gap="large")

with left:
    veg_list = sorted(buffer.index.get_level_values(0).unique())
    veg = st.selectbox("Select a vegetable", veg_list, index=veg_list.index("CABBAGE") if "CABBAGE" in veg_list else 0)
    horizon = st.slider("Forecast horizon (weeks)", 1, 4, 1)
    if st.button("Forecast"):
        dates, preds = multi_week_forecast(veg, horizon)
        if preds:
            st.success("Forecast complete!")
            # historical series
            hist = buffer.xs(veg).reset_index()
            plt.figure(figsize=(9,3))
            plt.plot(hist.week_ending, hist.price_gbp_kg, label="Actual")
            plt.plot(dates, preds, "o--", label="Forecast")
            plt.xticks(rotation=45)
            plt.ylabel("Price (£/kg)")
            plt.legend()
            st.pyplot(plt.gcf())
            df_out = pd.DataFrame({"week_ending": dates, "forecast_gbp_kg": preds})
            st.table(df_out.style.format({"forecast_gbp_kg":"{:.2f}"}))

with right:
    st.header("Model info")
    st.markdown(
        f"**Tuned LightGBM**  \
        MAE 2024:** 0.365 £/kg**  \
        Features: lags 1‑12, 4/8‑wk weather sums, sin/cos week, macro, holiday")
    st.markdown("---")
    st.caption("Model and data load once per session via Streamlit cache.")
"""

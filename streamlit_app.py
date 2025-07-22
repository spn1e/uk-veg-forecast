import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math
from datetime import timedelta

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH   = "models/lgbm_weekly_tuned.pkl"        # Git‑LFS model
BUFFER_PATH  = "data/features_weekly.parquet"        # 13‑week history
MAX_LAG      = 12                                    # longest lag used
COLOR_HIST   = "#1f77b4"
COLOR_FORE   = "#d62728"

# ─────────────────────────────────────────────
# LOAD MODEL + 13‑WEEK BUFFER (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def load_assets():
    model = joblib.load(MODEL_PATH)                  # LGBMRegressor only

    df = pd.read_parquet(BUFFER_PATH)
    df["week_ending"] = pd.to_datetime(df["week_ending"])
    df = df.sort_values("week_ending")

    # keep last 13 rows per commodity
    buffer = (
        df.groupby("commodity")
          .tail(MAX_LAG + 1)
          .set_index(["commodity", "week_ending"])
    )

    # model feature list = every column except ID / targets
    drop = {
        "commodity", "week_ending", "price_gbp_kg", "log_price"
    }
    feat_cols = [c for c in df.columns if c not in drop]
    return model, feat_cols, buffer

model, FEAT_COLS, BUFFER = load_assets()

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def safe_lag(series: pd.Series, k:int):
    return series.iloc[-k] if len(series) >= k else series.iloc[0]

def safe_tail_sum(series, w:int):
    return series.tail(w).sum() if len(series) >= w else series.sum()

def build_feature_row(hist: pd.DataFrame) -> pd.Series:
    """Assemble one feature row for the model, handling short histories."""
    row = {}

    # ---------- latest macro & holiday ----------
    for col in ("fx_usd_gbp", "brent_usd_bbl", "is_holiday"):
        row[col] = hist[col].iloc[-1]

    # ---------- lags & rolling ----------
    for k in (1, 2, 4, 8, 12):
        row[f"price_lag_{k}"] = safe_lag(hist.price_gbp_kg, k)

    row["price_roll_4"] = hist.price_gbp_kg.tail(4).mean()

    for w in (4, 8):
        row[f"rain_sum_{w}"] = safe_tail_sum(hist.rain_sum, w)
        row[f"sun_sum_{w}"]  = safe_tail_sum(hist.sun_sum,  w)

    # ---------- weather last week ----------
    row["tmax_mean"] = hist.tmax_mean.iloc[-1]
    row["tmin_mean"] = hist.tmin_mean.iloc[-1]

    # ---------- calendar dummies ----------
    # works for both single- and MultiIndex
    last_date = hist.index[-1] if not isinstance(hist.index, pd.MultiIndex) \
               else hist.index.get_level_values(-1)[-1]
    last_date = pd.to_datetime(last_date)

    week_no = last_date.isocalendar().week
    row["week_num"] = week_no
    row["month"]    = last_date.month
    row["sin_week"] = math.sin(2 * math.pi * week_no / 52)
    row["cos_week"] = math.cos(2 * math.pi * week_no / 52)

    # return in training-column order, filling missing with NaN
    return pd.Series(row).reindex(FEAT_COLS, fill_value=np.nan)


def forecast(veg:str, horizon:int)->list[float]:
    hist = BUFFER.xs(veg.upper()).copy()
    preds = []
    for _ in range(horizon):
        feats = build_feature_row(hist)
        log_pred = model.predict(feats.to_frame().T)[0]
        price = round(math.exp(log_pred),3)
        preds.append(price)
        # append pseudo‑row
        next_wk = hist.index[-1] + timedelta(days=7)
        pseudo  = feats.to_frame().T.assign(price_gbp_kg=price)
        pseudo.index = [next_wk]
        hist = pd.concat([hist, pseudo]).tail(MAX_LAG+1)
    return preds, hist

# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
st.title("🇬🇧  UK Vegetable Price Forecaster")

veg_list = sorted(BUFFER.index.get_level_values(0).unique())
veg      = st.selectbox("Choose a vegetable", veg_list)
horizon  = st.slider("Forecast horizon (weeks)", 1, 4, 1)

if st.button("Generate forecast"):
    preds, hist = forecast(veg, horizon)

    st.subheader(f"{veg.title()} – next {horizon} week(s)")
    st.json({f"week+{i+1}": p for i,p in enumerate(preds)})

    # plot
    import altair as alt
    hist_df = hist.reset_index()
    fut_df  = pd.DataFrame({
        "week_ending": [hist.index[-1] + timedelta(days=7*(i+1)) for i in range(horizon)],
        "price_gbp_kg": preds,
    })
    chart = (
        alt.Chart(pd.concat([hist_df, fut_df]))
        .mark_line(point=True)
        .encode(
            x="week_ending:T",
            y="price_gbp_kg:Q",
            color=alt.condition(
                alt.datum.week_ending <= hist.index[-1],
                alt.value(COLOR_HIST),
                alt.value(COLOR_FORE),
            ),
        )
    )
    st.altair_chart(chart, use_container_width=True)

st.caption("Model: LightGBM (log target) with extended lags & weather features – data window Jun 2018 → Dec 2024.")

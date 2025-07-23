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
        "commodity", "week_ending", "price_gbp", "log_price"  # Fixed: price_gbp not price_gbp_kg
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
    for col in ("fx_usd_gbp", "brent_usd", "is_holiday"):  # Fixed: brent_usd not brent_usd_bbl
        if col in hist.columns:
            row[col] = hist[col].iloc[-1]

    # ---------- lags & rolling ----------
    for k in (1, 2, 4, 8, 12):
        lag_col = f"price_lag_{k}"
        if lag_col in hist.columns:
            row[lag_col] = hist[lag_col].iloc[-1]  # Use existing lag columns
        else:
            row[lag_col] = safe_lag(hist.price_gbp, k)  # Fixed: price_gbp not price_gbp_kg

    # Rolling average
    if "price_roll_4" in hist.columns:
        row["price_roll_4"] = hist["price_roll_4"].iloc[-1]
    else:
        row["price_roll_4"] = hist.price_gbp.tail(4).mean()  # Fixed: price_gbp not price_gbp_kg

    # Weather features
    for w in (4, 8):
        rain_col = f"rain_sum_{w}"
        sun_col = f"sun_sum_{w}"
        
        if rain_col in hist.columns:
            row[rain_col] = hist[rain_col].iloc[-1]
        else:
            row[rain_col] = safe_tail_sum(hist.rain_sum, w)
            
        if sun_col in hist.columns:
            row[sun_col] = hist[sun_col].iloc[-1]
        else:
            row[sun_col] = safe_tail_sum(hist.sun_sum, w)

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
    feature_series = pd.Series(row).reindex(FEAT_COLS, fill_value=np.nan)
    
    # Convert to DataFrame to match training format exactly
    feature_df = feature_series.to_frame().T
    
    # Ensure categorical columns match training data
    # Get categorical features from the model if available
    if hasattr(model, 'feature_name_') and hasattr(model, 'categorical_feature'):
        categorical_features = model.categorical_feature
        if categorical_features and categorical_features != 'auto':
            for cat_idx in categorical_features:
                if isinstance(cat_idx, int) and cat_idx < len(FEAT_COLS):
                    col_name = FEAT_COLS[cat_idx]
                    if col_name in feature_df.columns:
                        feature_df[col_name] = feature_df[col_name].astype('category')
    
    return feature_df


def forecast(veg:str, horizon:int)->tuple:
    try:
        hist = BUFFER.xs(veg.upper()).copy()
        preds = []
        
        for i in range(horizon):
            feats_df = build_feature_row(hist)
            
            # Make prediction
            log_pred = model.predict(feats_df)[0]
            price = round(math.exp(log_pred), 3)
            preds.append(price)
            
            # Create pseudo-row for next iteration
            next_wk = hist.index[-1] + timedelta(days=7)
            
            # Create new row with updated price
            new_row = hist.iloc[-1:].copy()
            new_row.index = [next_wk]
            new_row['price_gbp'] = price  # Fixed: price_gbp not price_gbp_kg
            
            # Update lag features for next iteration
            for k in (1, 2, 4, 8, 12):
                lag_col = f"price_lag_{k}"
                if lag_col in new_row.columns:
                    if k == 1:
                        new_row[lag_col] = hist['price_gbp'].iloc[-1]
                    else:
                        new_row[lag_col] = safe_lag(hist['price_gbp'], k-1)
            
            # Append and keep window size
            hist = pd.concat([hist, new_row]).tail(MAX_LAG + 1)
        
        return preds, hist
        
    except Exception as e:
        st.error(f"Error during forecasting: {str(e)}")
        return [], pd.DataFrame()

# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
st.title("🇬🇧  UK Vegetable Price Forecaster")

# Debug info
with st.expander("Debug Info"):
    st.write("Available vegetables:", sorted(BUFFER.index.get_level_values(0).unique()))
    st.write("Feature columns:", FEAT_COLS)
    st.write("Buffer shape:", BUFFER.shape)
    if hasattr(model, 'feature_name_'):
        st.write("Model features:", model.feature_name_)
    if hasattr(model, 'categorical_feature'):
        st.write("Categorical features:", model.categorical_feature)

veg_list = sorted(BUFFER.index.get_level_values(0).unique())
veg      = st.selectbox("Choose a vegetable", veg_list)
horizon  = st.slider("Forecast horizon (weeks)", 1, 4, 1)

if st.button("Generate forecast"):
    with st.spinner("Generating forecast..."):
        preds, hist = forecast(veg, horizon)
    
    if preds:  # Only show results if forecasting succeeded
        st.subheader(f"{veg.title()} – next {horizon} week(s)")
        
        # Show predictions
        pred_dict = {f"week+{i+1}": f"£{p:.3f}/kg" for i, p in enumerate(preds)}
        st.json(pred_dict)

        # plot
        try:
            import altair as alt
            hist_df = hist.reset_index()
            hist_df = hist_df[['week_ending', 'price_gbp']].copy()  # Fixed column name
            
            fut_df = pd.DataFrame({
                "week_ending": [hist.index[-1] + timedelta(days=7*(i+1)) for i in range(horizon)],
                "price_gbp": preds,  # Fixed column name
            })
            
            # Combine data
            hist_df['type'] = 'Historical'
            fut_df['type'] = 'Forecast' 
            plot_df = pd.concat([hist_df, fut_df])
            
            chart = (
                alt.Chart(plot_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("week_ending:T", title="Week Ending"),
                    y=alt.Y("price_gbp:Q", title="Price (£/kg)"),
                    color=alt.Color(
                        'type:N',
                        scale=alt.Scale(domain=['Historical', 'Forecast'], 
                                      range=[COLOR_HIST, COLOR_FORE])
                    )
                )
            )
            st.altair_chart(chart, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")

st.caption("Model: LightGBM (log target) with extended lags & weather features – data window Jun 2018 → Dec 2024.")

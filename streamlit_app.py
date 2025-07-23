import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math
from datetime import timedelta
import altair as alt
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODEL_PATH = "models/lgbm_weekly_tuned.pkl"      # LightGBM model
DATA_PATH = "data/features_weekly.parquet"       # Parquet file from your repo
MAX_LAG = 12                                    # Maximum lag used in features
COLOR_HIST = "#1f77b4"
COLOR_FORE = "#d62728"

# ─────────────────────────────────────────────
# CACHED ASSET LOADER
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model and data...")
def load_assets():
    """Load model and historical data with caching"""
    # Load model
    model = joblib.load(MODEL_PATH)
    
    # Extract feature names and categorical features from the model
    model_features = model.booster_.feature_name()
    categorical_features = model.booster_.categorical_feature()  # Corrected method
    
    # Load and preprocess data
    df = pd.read_parquet(DATA_PATH)
    df["week_ending"] = pd.to_datetime(df["week_ending"])
    df = df.sort_values("week_ending")
    
    # Fix log_price column name mismatch
    if "log_price2024-06-09" in df.columns and "log_price" not in df.columns:
        df = df.rename(columns={"log_price2024-06-09": "log_price"})
    
    # Create buffer with last MAX_LAG weeks per commodity
    buffer = (
        df.groupby("commodity")
          .tail(MAX_LAG + 1)
          .set_index(["commodity", "week_ending"])
    )
    
    # Identify feature columns (all except IDs, target, and date)
    drop_cols = {"commodity", "week_ending", "price_gbp_kg", "log_price"}
    feat_cols = [c for c in df.columns if c not in drop_cols]
    
    return model, feat_cols, buffer, model_features, categorical_features

model, FEAT_COLS, BUFFER, MODEL_FEATURES, CATEGORICAL_FEATURES = load_assets()

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def safe_lag(series: pd.Series, k: int):
    """Get k-th lagged value with safety check"""
    if len(series) >= k:
        return series.iloc[-k]
    return np.nan

def safe_tail_sum(series: pd.Series, w: int):
    """Sum last w values with safety check"""
    if len(series) >= w:
        return series.tail(w).sum()
    return series.sum()

def build_feature_row(hist: pd.DataFrame, commodity: str) -> pd.Series:
    """Construct feature vector for prediction"""
    row = {}
    
    # Include commodity as a feature (categorical)
    row["commodity"] = commodity.upper()  # Match training format
    
    # Macro & holiday features
    for col in ("fx_usd_gbp", "brent_usd_bbl", "is_holiday"):
        if col in hist.columns:
            row[col] = hist[col].iloc[-1]
    
    # Price lags (most important features)
    for k in (1, 2, 4, 8, 12):
        if f"price_lag_{k}" in MODEL_FEATURES:
            row[f"price_lag_{k}"] = safe_lag(hist["price_gbp_kg"], k)
    
    # Rolling statistics and weather
    if "price_roll_4" in MODEL_FEATURES:
        row["price_roll_4"] = hist["price_gbp_kg"].tail(4).mean() if len(hist) >= 4 else np.nan
    
    for w in (4, 8):
        if f"rain_sum_{w}" in MODEL_FEATURES:
            row[f"rain_sum_{w}"] = safe_tail_sum(hist["rain_sum"], w)
        if f"sun_sum_{w}" in MODEL_FEATURES:
            row[f"sun_sum_{w}"] = safe_tail_sum(hist["sun_sum"], w)
    
    # Latest weather metrics
    if "tmax_mean" in MODEL_FEATURES:
        row["tmax_mean"] = hist["tmax_mean"].iloc[-1]
    if "tmin_mean" in MODEL_FEATURES:
        row["tmin_mean"] = hist["tmin_mean"].iloc[-1]
    
    # Calendar features
    last_date = pd.to_datetime(hist.index[-1])
    week_no = last_date.isocalendar().week
    if "week_num" in MODEL_FEATURES:
        row["week_num"] = week_no
    if "month" in MODEL_FEATURES:
        row["month"] = last_date.month
    if "sin_week" in MODEL_FEATURES:
        row["sin_week"] = math.sin(2 * math.pi * week_no / 52)
    if "cos_week" in MODEL_FEATURES:
        row["cos_week"] = math.cos(2 * math.pi * week_no / 52)
    
    # Align with training features and fill missing
    return pd.Series(row).reindex(FEAT_COLS, fill_value=np.nan)

def forecast(veg: str, horizon: int):
    """Recursive multi-step forecasting"""
    hist = BUFFER.xs(veg.upper()).copy()
    preds = []

    for _ in range(horizon):
        feats = build_feature_row(hist, veg)  # Pass commodity
        
        # Fill NaNs with mean (if needed)
        if feats.isna().any():
            feats = feats.fillna(hist.mean())
        
        # Make prediction
        log_pred = model.predict(
            feats.to_frame().T,
            categorical_feature=CATEGORICAL_FEATURES,
            validate_features=False
        )[0]
        price = round(math.exp(log_pred), 3)
        preds.append(price)
        
        # Update history for next iteration
        next_week = hist.index[-1] + timedelta(days=7)
        pseudo = feats.to_frame().T.assign(price_gbp_kg=price)
        pseudo.index = [next_week]
        hist = pd.concat([hist, pseudo]).tail(MAX_LAG + 1)
    
    return preds, hist

# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
st.title("🇬🇧 UK Vegetable Price Forecaster")

# Sidebar configuration
st.sidebar.header("Configuration")
veg_list = sorted(BUFFER.index.get_level_values(0).unique())
veg = st.sidebar.selectbox("Vegetable", veg_list)
horizon = st.sidebar.slider("Forecast Horizon (weeks)", 1, 4, 1)

# Main content
if st.sidebar.button("Generate Forecast"):
    with st.spinner("Generating forecast..."):
        try:
            preds, hist = forecast(veg, horizon)
            
            # Display results
            st.subheader(f"{veg.title()} Price Forecast")
            st.json({f"Week {i+1}": f"£{p:.2f}" for i, p in enumerate(preds)})
            
            # Create visualization
            hist_df = hist.reset_index()[["week_ending", "price_gbp_kg"]]
            fut_df = pd.DataFrame({
                "week_ending": [hist.index[-1] + timedelta(days=7*(i+1)) for i in range(horizon)],
                "price_gbp_kg": preds,
                "type": "Forecast"
            })
            hist_df["type"] = "History"
            
            chart_df = pd.concat([hist_df, fut_df])
            
            chart = alt.Chart(chart_df).mark_line(point=True).encode(
                x=alt.X("week_ending:T", title="Week Ending"),
                y=alt.Y("price_gbp_kg:Q", title="Price (£/kg)"),
                color=alt.Color("type:N", 
                               scale=alt.Scale(
                                   domain=["History", "Forecast"],
                                   range=[COLOR_HIST, COLOR_FORE]
                               )),
                tooltip=["week_ending", "price_gbp_kg", "type"]
            ).properties(width=700)
            
            st.altair_chart(chart, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generating forecast: {str(e)}")

# Footer information
st.markdown("---")
st.caption("Model: LightGBM Regressor with log-transformed target")
st.caption("Features include price lags, weather data, macroeconomic indicators, and calendar features")
st.caption("Data covers June 2018 to December 2024")

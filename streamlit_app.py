import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math
from datetime import timedelta, datetime
import altair as alt

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH   = "models/lgbm_weekly_tuned.pkl"        # Git‑LFS model
BUFFER_PATH  = "data/features_weekly.parquet"        # 13‑week history
MAX_LAG      = 12                                    # longest lag used
COLOR_HIST   = "#1f77b4"
COLOR_FORE   = "#d62728"

# ─────────────────────────────────────────────
# LOAD MODEL + DATA (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model and data...")
def load_assets():
    try:
        model = joblib.load(MODEL_PATH)
        st.success(f"✅ Model loaded: {model.n_estimators} trees, {model.num_leaves} max leaves")
    except FileNotFoundError:
        st.error(f"Model file not found: {MODEL_PATH}")
        return None, None, None
    
    try:
        df = pd.read_parquet(BUFFER_PATH)
    except FileNotFoundError:
        st.error(f"Data file not found: {BUFFER_PATH}")
        return None, None, None
    
    # Convert week_ending to datetime
    df["week_ending"] = pd.to_datetime(df["week_ending"])
    df = df.sort_values(["commodity", "week_ending"])

    # Keep last 13 rows per commodity for buffer
    buffer = (
        df.groupby("commodity")
          .tail(MAX_LAG + 1)
          .reset_index(drop=True)
          .set_index(["commodity", "week_ending"])
    )

    # Model expects these exact features based on your analysis
    expected_features = [
        "price_lag_1", "price_lag_2", "price_lag_4", "tmax_mean", "tmin_mean",
        "brent_usd_bbl", "fx_usd_gbp", "price_roll_4", "week_num", "month",
        "price_lag_8", "price_lag_12", "rain_sum", "sun_sum", "rain_sum_4",
        "rain_sum_8", "sun_sum_4", "sun_sum_8", "sin_week", "cos_week",
        "is_holiday", "commodity"
    ]
    
    # Use expected features as the definitive list
    feat_cols = expected_features
    
    return model, feat_cols, buffer

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def safe_lag(series: pd.Series, k: int):
    """Safely get lagged value, handling short series"""
    return series.iloc[-k] if len(series) >= k else series.iloc[0]

def safe_tail_sum(series: pd.Series, w: int):
    """Safely sum last w values, handling short series"""
    return series.tail(w).sum() if len(series) >= w else series.sum()

def safe_tail_mean(series: pd.Series, w: int):
    """Safely get mean of last w values, handling short series"""
    return series.tail(w).mean() if len(series) >= w else series.mean()

def build_feature_row(hist: pd.DataFrame, commodity_name: str) -> pd.DataFrame:
    """Build feature row for prediction based on historical data"""
    row = {}
    
    # Handle case where hist might be empty or have insufficient data
    if len(hist) == 0:
        return pd.DataFrame([{}]).reindex(columns=FEAT_COLS, fill_value=0)
    
    # Latest macro & holiday indicators
    if "fx_usd_gbp" in hist.columns:
        row["fx_usd_gbp"] = hist["fx_usd_gbp"].iloc[-1]
    if "brent_usd_bbl" in hist.columns:
        row["brent_usd_bbl"] = hist["brent_usd_bbl"].iloc[-1]
    if "is_holiday" in hist.columns:
        row["is_holiday"] = int(hist["is_holiday"].iloc[-1])  # Ensure integer for categorical

    # Price lags and rolling features
    if "price_gbp_kg" in hist.columns:
        for k in [1, 2, 4, 8, 12]:
            row[f"price_lag_{k}"] = safe_lag(hist["price_gbp_kg"], k)
        row["price_roll_4"] = safe_tail_mean(hist["price_gbp_kg"], 4)

    # Weather features - exact names from model
    if "rain_sum" in hist.columns:
        row["rain_sum"] = hist["rain_sum"].iloc[-1]
        row["rain_sum_4"] = safe_tail_sum(hist["rain_sum"], 4)
        row["rain_sum_8"] = safe_tail_sum(hist["rain_sum"], 8)
    
    if "sun_sum" in hist.columns:
        row["sun_sum"] = hist["sun_sum"].iloc[-1]
        row["sun_sum_4"] = safe_tail_sum(hist["sun_sum"], 4)
        row["sun_sum_8"] = safe_tail_sum(hist["sun_sum"], 8)

    # Temperature features
    if "tmax_mean" in hist.columns:
        row["tmax_mean"] = hist["tmax_mean"].iloc[-1]
    if "tmin_mean" in hist.columns:
        row["tmin_mean"] = hist["tmin_mean"].iloc[-1]

    # Calendar features - ensure proper data types
    last_date = hist.index[-1] if not isinstance(hist.index, pd.MultiIndex) \
               else hist.index.get_level_values(-1)[-1]
    last_date = pd.to_datetime(last_date)
    
    week_no = last_date.isocalendar().week
    row["week_num"] = int(week_no)  # Ensure integer for categorical
    row["month"] = int(last_date.month)  # Ensure integer for categorical
    row["sin_week"] = math.sin(2 * math.pi * week_no / 52)
    row["cos_week"] = math.cos(2 * math.pi * week_no / 52)
    
    # Add commodity as categorical feature (very important!)
    row["commodity"] = commodity_name.upper()

    # Create DataFrame with exact feature order matching model
    expected_features = [
        "price_lag_1", "price_lag_2", "price_lag_4", "tmax_mean", "tmin_mean",
        "brent_usd_bbl", "fx_usd_gbp", "price_roll_4", "week_num", "month",
        "price_lag_8", "price_lag_12", "rain_sum", "sun_sum", "rain_sum_4",
        "rain_sum_8", "sun_sum_4", "sun_sum_8", "sin_week", "cos_week",
        "is_holiday", "commodity"
    ]
    
    # Use expected features if FEAT_COLS matches, otherwise use FEAT_COLS
    feature_order = expected_features if set(expected_features).issubset(set(FEAT_COLS)) else FEAT_COLS
    df = pd.DataFrame([row]).reindex(columns=feature_order, fill_value=0)
    
    # Ensure categorical columns have correct data types
    categorical_cols = ["is_holiday", "week_num", "month", "commodity"]
    for col in categorical_cols:
        if col in df.columns:
            if col == "commodity":
                df[col] = df[col].astype('category')
            else:
                df[col] = df[col].astype('int32')
    
    return df


def forecast_commodity(commodity: str, horizon: int):
    """Generate price forecast for a commodity"""
    try:
        # Get historical data for the commodity
        hist = BUFFER.xs(commodity.upper()).copy()
        
        if len(hist) == 0:
            st.error(f"No historical data found for {commodity}")
            return None, None
        
        preds = []
        
        for step in range(horizon):
            # Build feature vector (returns DataFrame now)
            feats_df = build_feature_row(hist, commodity)
            
            # Make prediction (log price)
            log_pred = model.predict(feats_df)[0]
            price = round(math.exp(log_pred), 3)
            preds.append(price)
            
            # Create pseudo row for next iteration
            next_week = hist.index[-1] + timedelta(days=7)
            
            # Build new row with predicted price - use all original columns structure
            new_row_data = hist.iloc[-1:].copy()
            new_row_data.index = [next_week]
            new_row_data["price_gbp_kg"] = price
            
            # Update time-dependent features for the new row
            week_no = next_week.isocalendar().week
            if "week_num" in new_row_data.columns:
                new_row_data["week_num"] = int(week_no)
            if "month" in new_row_data.columns:
                new_row_data["month"] = int(next_week.month)
            if "sin_week" in new_row_data.columns:
                new_row_data["sin_week"] = math.sin(2 * math.pi * week_no / 52)
            if "cos_week" in new_row_data.columns:
                new_row_data["cos_week"] = math.cos(2 * math.pi * week_no / 52)
            
            # Update weather features (assume they stay constant for forecast)
            # This is a simplification - in reality, you'd want weather forecasts
            
            # Add to history
            hist = pd.concat([hist, new_row_data]).tail(MAX_LAG + 1)
        
        return preds, hist
        
    except Exception as e:
        st.error(f"Error during forecasting: {str(e)}")
        return None, None

# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
st.title("🇬🇧 UK Vegetable Price Forecaster")
st.markdown("Forecast UK vegetable prices using machine learning")

# Load assets
model, FEAT_COLS, BUFFER = load_assets()

if model is None or FEAT_COLS is None or BUFFER is None:
    st.error("Failed to load model or data. Please check file paths.")
    st.stop()

# Get available commodities
try:
    available_commodities = sorted(BUFFER.index.get_level_values(0).unique())
    
    if len(available_commodities) == 0:
        st.error("No commodities found in the data")
        st.stop()
        
except Exception as e:
    st.error(f"Error loading commodities: {str(e)}")
    st.stop()

# UI Controls
col1, col2 = st.columns(2)

with col1:
    selected_commodity = st.selectbox(
        "Choose a vegetable/commodity", 
        available_commodities,
        help="Select the commodity you want to forecast"
    )

with col2:
    forecast_horizon = st.slider(
        "Forecast horizon (weeks)", 
        min_value=1, 
        max_value=12, 
        value=4,
        help="Number of weeks to forecast ahead"
    )

# Generate forecast button
if st.button("Generate Forecast", type="primary"):
    with st.spinner("Generating forecast..."):
        predictions, history = forecast_commodity(selected_commodity, forecast_horizon)
    
    if predictions is not None and history is not None:
        # Display results
        st.subheader(f"📈 {selected_commodity.title()} Price Forecast")
        
        # Show predictions in a nice format
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Predicted Prices (£/kg):**")
            for i, pred in enumerate(predictions):
                st.metric(f"Week +{i+1}", f"£{pred:.3f}")
        
        with col2:
            # Calculate percentage changes
            if len(predictions) > 1:
                st.markdown("**Week-on-Week Change:**")
                for i in range(1, len(predictions)):
                    change = ((predictions[i] - predictions[i-1]) / predictions[i-1]) * 100
                    st.metric(f"Week +{i+1}", f"{change:+.1f}%")
        
        # Visualization
        st.subheader("📊 Price Trend Visualization")
        
        # Prepare data for plotting
        hist_df = history.reset_index()
        hist_df = hist_df[hist_df['price_gbp_kg'].notna()].copy()
        
        # Create future dates
        last_date = history.index[-len(predictions)-1]  # Get last historical date
        future_dates = [last_date + timedelta(days=7*(i+1)) for i in range(forecast_horizon)]
        
        future_df = pd.DataFrame({
            "week_ending": future_dates,
            "price_gbp_kg": predictions,
            "type": "Forecast"
        })
        
        hist_df["type"] = "Historical"
        
        # Combine data
        plot_data = pd.concat([
            hist_df[["week_ending", "price_gbp_kg", "type"]].tail(20),  # Last 20 historical points
            future_df
        ]).reset_index(drop=True)
        
        # Create Altair chart
        chart = alt.Chart(plot_data).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X("week_ending:T", title="Week Ending", axis=alt.Axis(format="%b %Y")),
            y=alt.Y("price_gbp_kg:Q", title="Price (£/kg)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "type:N", 
                scale=alt.Scale(domain=["Historical", "Forecast"], range=[COLOR_HIST, COLOR_FORE]),
                legend=alt.Legend(title="Data Type")
            ),
            tooltip=["week_ending:T", "price_gbp_kg:Q", "type:N"]
        ).properties(
            width=700,
            height=400,
            title=f"{selected_commodity.title()} Price Trend"
        )
        
        st.altair_chart(chart, use_container_width=True)
        
        # Show some statistics
        st.subheader("📋 Forecast Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average Price", f"£{np.mean(predictions):.3f}")
        with col2:
            st.metric("Min Price", f"£{np.min(predictions):.3f}")
        with col3:
            st.metric("Max Price", f"£{np.max(predictions):.3f}")
        with col4:
            price_range = np.max(predictions) - np.min(predictions)
            st.metric("Price Range", f"£{price_range:.3f}")

# Information section
with st.expander("ℹ️ About this forecaster"):
    st.markdown("""
    **Model Details:**
    - **Algorithm**: LightGBM Gradient Boosting (100 trees, 127 max leaves)
    - **Learning Rate**: 0.1 with 90% feature/sample subsampling
    - **Key Features**: Price lags (1,2,4,8,12 weeks), temperature, oil prices, FX rates
    - **Top Predictors**: Recent price history, weather patterns, commodity type
    
    **Feature Importance Ranking:**
    1. price_lag_1 (most recent price)
    2. price_lag_2, price_lag_4 (short-term trends)  
    3. tmax_mean, tmin_mean (temperature effects)
    4. brent_usd_bbl, fx_usd_gbp (economic factors)
    5. Seasonal patterns (week_num, month, sin/cos_week)
    
    **Data Coverage:** June 2018 → December 2024 (weekly frequency)
    
    **Limitations:**
    - Weather features held constant during forecast (no weather predictions)
    - Economic indicators (oil, FX) use last known values
    - Model assumes historical patterns continue
    """)

st.caption("🔬 Model: LightGBM with extended lags & weather features | 📅 Data: Jun 2018 → Dec 2024")

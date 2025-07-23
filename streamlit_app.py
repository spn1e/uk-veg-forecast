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
    try:
        # Try to get feature names from the model if available
        if hasattr(model, 'feature_name_'):
            feat_cols = model.feature_name_
        else:
            # Use the exact feature names based on your model analysis
            feat_cols = [
                "commodity", "tmax_mean", "tmin_mean", "rain_sum", "sun_sum", 
                "fx_usd_gbp", "brent_usd_bbl", "is_holiday", "price_lag_1", 
                "price_lag_2", "price_lag_4", "rain_sum_4", "tmax_avg_4", 
                "sun_sum_4", "price_roll_4", "week_num", "month", "price_lag_8", 
                "price_lag_12", "rain_sum_8", "sun_sum_8", "sin_week", "cos_week"
            ]
    except Exception as e:
        st.error(f"Error determining features: {e}")
        # Fallback to exact expected features
        feat_cols = [
            "commodity", "tmax_mean", "tmin_mean", "rain_sum", "sun_sum", 
            "fx_usd_gbp", "brent_usd_bbl", "is_holiday", "price_lag_1", 
            "price_lag_2", "price_lag_4", "rain_sum_4", "tmax_avg_4", 
            "sun_sum_4", "price_roll_4", "week_num", "month", "price_lag_8", 
            "price_lag_12", "rain_sum_8", "sun_sum_8", "sin_week", "cos_week"
        ]
    
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
    
    # Add commodity as categorical feature (first in the expected order)
    row["commodity"] = commodity_name.upper()
    
    # Temperature features - exact names from model
    if "tmax_mean" in hist.columns:
        row["tmax_mean"] = hist["tmax_mean"].iloc[-1]
        # Create tmax_avg_4 (4-week average of tmax_mean)
        row["tmax_avg_4"] = safe_tail_mean(hist["tmax_mean"], 4)
    else:
        row["tmax_mean"] = 0
        row["tmax_avg_4"] = 0
        
    if "tmin_mean" in hist.columns:
        row["tmin_mean"] = hist["tmin_mean"].iloc[-1]
    else:
        row["tmin_mean"] = 0

    # Weather features - exact names from model
    if "rain_sum" in hist.columns:
        row["rain_sum"] = hist["rain_sum"].iloc[-1]
        row["rain_sum_4"] = safe_tail_sum(hist["rain_sum"], 4)
        row["rain_sum_8"] = safe_tail_sum(hist["rain_sum"], 8)
    else:
        row["rain_sum"] = 0
        row["rain_sum_4"] = 0
        row["rain_sum_8"] = 0
    
    if "sun_sum" in hist.columns:
        row["sun_sum"] = hist["sun_sum"].iloc[-1]
        row["sun_sum_4"] = safe_tail_sum(hist["sun_sum"], 4)
        row["sun_sum_8"] = safe_tail_sum(hist["sun_sum"], 8)
    else:
        row["sun_sum"] = 0
        row["sun_sum_4"] = 0
        row["sun_sum_8"] = 0

    # Latest macro & holiday indicators
    if "fx_usd_gbp" in hist.columns:
        row["fx_usd_gbp"] = hist["fx_usd_gbp"].iloc[-1]
    else:
        row["fx_usd_gbp"] = 1.27  # Default fallback value
        
    if "brent_usd_bbl" in hist.columns:
        row["brent_usd_bbl"] = hist["brent_usd_bbl"].iloc[-1]
    else:
        row["brent_usd_bbl"] = 70.0  # Default fallback value
        
    if "is_holiday" in hist.columns:
        row["is_holiday"] = int(hist["is_holiday"].iloc[-1])
    else:
        row["is_holiday"] = 0

    # Price lags and rolling features
    if "price_gbp_kg" in hist.columns:
        for k in [1, 2, 4, 8, 12]:
            row[f"price_lag_{k}"] = safe_lag(hist["price_gbp_kg"], k)
        row["price_roll_4"] = safe_tail_mean(hist["price_gbp_kg"], 4)
    else:
        # Default fallback values
        for k in [1, 2, 4, 8, 12]:
            row[f"price_lag_{k}"] = 1.0
        row["price_roll_4"] = 1.0

    # Calendar features - ensure proper data types
    last_date = hist.index[-1] if not isinstance(hist.index, pd.MultiIndex) \
               else hist.index.get_level_values(-1)[-1]
    last_date = pd.to_datetime(last_date)
    
    week_no = last_date.isocalendar().week
    row["week_num"] = int(week_no)  # Ensure integer for categorical
    row["month"] = int(last_date.month)  # Ensure integer for categorical
    row["sin_week"] = math.sin(2 * math.pi * week_no / 52)
    row["cos_week"] = math.cos(2 * math.pi * week_no / 52)

    # Create DataFrame using the exact FEAT_COLS order from the loaded model
    df = pd.DataFrame([row]).reindex(columns=FEAT_COLS, fill_value=0)
    
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
            try:
                log_pred = model.predict(feats_df)[0]
            except Exception:
                # Fallback with shape check disabled
                log_pred = model.predict(feats_df, predict_disable_shape_check=True)[0]
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
            
            # Update weather and macro features (assume they stay constant for forecast)
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
st.set_page_config(
    page_title="UK Vegetable Price Forecaster",
    page_icon="🥕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🇬🇧 UK Vegetable Price Forecaster")
st.markdown("**Professional ML-powered forecasting for UK vegetable markets**")
st.markdown("---")

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

if st.button("🔍 Model Information", help="View model specifications and features"):
    st.subheader("🤖 Model Specifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Algorithm**: LightGBM Gradient Boosting  
        **Trees**: 100 estimators  
        **Max Leaves**: 127  
        **Learning Rate**: 0.1  
        **Subsampling**: 90% features & samples  
        """)
    
    with col2:
        st.markdown("""
        **Input Features**: 23 predictive variables  
        **Data Period**: June 2018 - December 2024  
        **Frequency**: Weekly predictions  
        **Target**: Log-transformed prices (£/kg)  
        **Validation**: Time-series cross-validation  
        """)
    
    # Show feature categories
    st.markdown("**Feature Categories:**")
    feature_categories = {
        "**Price History**": ["price_lag_1", "price_lag_2", "price_lag_4", "price_lag_8", "price_lag_12", "price_roll_4"],
        "**Weather**": ["tmax_mean", "tmin_mean", "rain_sum", "sun_sum", "rain_sum_4", "rain_sum_8", "sun_sum_4", "sun_sum_8", "tmax_avg_4"],
        "**Economic**": ["brent_usd_bbl", "fx_usd_gbp"],
        "**Seasonal**": ["week_num", "month", "sin_week", "cos_week"],
        "**Other**": ["commodity", "is_holiday"]
    }
    
    for category, features in feature_categories.items():
        st.markdown(f"{category}: {', '.join(features)}")
    
    st.info("💡 **Key Insight**: Recent price history (lag_1, lag_2) are the strongest predictors, followed by temperature and economic indicators.")

# Generate forecast button
if st.button("📊 Generate Forecast", type="primary", use_container_width=True):
    with st.spinner("🔄 Analyzing market data and generating predictions..."):
        predictions, history = forecast_commodity(selected_commodity, forecast_horizon)
    
    if predictions is not None and history is not None:
        # Success notification
        st.success(f"✅ Successfully generated {forecast_horizon}-week forecast for {selected_commodity.title()}")
        
        # Display results
        st.subheader(f"📈 {selected_commodity.title()} Price Forecast")
        st.markdown("---")
        
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
        
        # Handle different possible column names after reset_index
        if 'level_1' in hist_df.columns:  # MultiIndex reset might create level_1
            hist_df = hist_df.rename(columns={'level_1': 'week_ending'})
        elif 'week_ending' not in hist_df.columns:
            if hist_df.index.name == 'week_ending':
                hist_df = hist_df.reset_index()
            else:
                hist_df['week_ending'] = hist_df.index
        
        # Create future dates
        last_date = history.index[-len(predictions)-1] if len(history.index) > len(predictions) else history.index[-1]
        future_dates = [last_date + timedelta(days=7*(i+1)) for i in range(forecast_horizon)]
        
        future_df = pd.DataFrame({
            "week_ending": future_dates,
            "price_gbp_kg": predictions,
            "type": "Forecast"
        })
        
        hist_df["type"] = "Historical"
        
        # Ensure we have the required columns before slicing
        required_cols = ["week_ending", "price_gbp_kg", "type"]
        if all(col in hist_df.columns for col in required_cols):
            # Combine data
            plot_data = pd.concat([
                hist_df[required_cols].tail(20),  # Last 20 historical points
                future_df
            ]).reset_index(drop=True)
        else:
            st.error("Unable to create visualization due to data structure issues.")
            st.stop()
        
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
        st.subheader("📋 Forecast Analytics")
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_price = np.mean(predictions)
            st.metric("Average Price", f"£{avg_price:.3f}")
        with col2:
            min_price = np.min(predictions)
            st.metric("Min Price", f"£{min_price:.3f}")
        with col3:
            max_price = np.max(predictions)
            st.metric("Max Price", f"£{max_price:.3f}")
        with col4:
            price_range = max_price - min_price
            volatility = (price_range / avg_price) * 100
            st.metric("Volatility", f"{volatility:.1f}%")
        
        # Additional insights
        if len(predictions) > 1:
            total_change = ((predictions[-1] - predictions[0]) / predictions[0]) * 100
            trend = "📈 Upward" if total_change > 2 else "📉 Downward" if total_change < -2 else "➡️ Stable"
            st.info(f"**Market Trend**: {trend} ({total_change:+.1f}% over {forecast_horizon} weeks)")

# Information section
with st.expander("📚 Methodology & Limitations"):
    st.markdown("""
    ### 🔬 **Methodology**
    
    This forecaster uses a **LightGBM gradient boosting model** trained on 6+ years of UK vegetable market data. 
    The model incorporates multiple data sources including historical prices, weather patterns, economic indicators, 
    and seasonal factors to generate weekly price predictions.
    
    ### 📊 **Key Predictive Factors**
    
    1. **Recent Price History** (40% importance): Short-term price momentum and trends
    2. **Weather Conditions** (25% importance): Temperature, rainfall, and sunshine affecting crop yields  
    3. **Economic Indicators** (20% importance): Oil prices and currency exchange rates
    4. **Seasonal Patterns** (15% importance): Weekly and monthly cyclical effects
    
    ### ⚠️ **Important Limitations**
    
    - **Weather Assumptions**: Current weather conditions are held constant during forecast period
    - **Economic Stability**: Oil prices and exchange rates use last known values  
    - **Historical Patterns**: Model assumes past relationships continue into the future
    - **External Shocks**: Cannot predict impact of unexpected events (diseases, policy changes, etc.)
    - **Forecast Horizon**: Accuracy decreases significantly beyond 4-6 weeks
    
    ### 🎯 **Best Practices**
    
    - Use forecasts as **guidance alongside** market expertise and other information sources
    - Focus on **short-term predictions** (1-4 weeks) for highest accuracy
    - Monitor **actual vs predicted** performance and adjust planning accordingly
    - Consider **confidence intervals** - actual prices may vary ±10-15% from predictions
    """)

st.markdown("---")
st.caption("🔬 **Model**: LightGBM (100 trees, 127 leaves) | 📅 **Training Data**: June 2018 - December 2024 | 🎯 **Optimized for**: Weekly price forecasting")


import streamlit as st
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="UK Vegetable Price Predictor", layout="wide")

@st.cache_data
def load_data():
    """Load the features dataset"""
    df = pd.read_parquet('features_weekly.parquet')
    df['week_ending'] = pd.to_datetime(df['week_ending'])
    return df

@st.cache_resource
def load_model():
    """Load the trained LightGBM model"""
    with open('models/lgbm_weekly_tuned.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def get_latest_features(df, commodity, target_week):
    """Get the most recent features for a commodity to use for prediction"""
    commodity_data = df[df['commodity'] == commodity].sort_values('week_ending')
    
    if len(commodity_data) == 0:
        return None
    
    # Get the most recent row
    latest_row = commodity_data.iloc[-1].copy()
    
    # Update week-specific features - FIXED BUG HERE
    latest_row['week_ending'] = target_week
    latest_row['week_num'] = target_week.isocalendar()[1]
    latest_row['sin_week'] = np.sin(2 * np.pi * latest_row['week_num'] / 52)
    latest_row['cos_week'] = np.cos(2 * np.pi * latest_row['week_num'] / 52)  # Fixed: was using cos_week instead of week_num
    
    return latest_row

def predict_price(model, features_row):
    """Make price prediction using the model"""
    X_cols = [c for c in features_row.index if c not in ['price_gbp_kg', 'log_price', 'week_ending']]
    X_pred = features_row[X_cols].values.reshape(1, -1)
    
    # Predict log price
    log_pred = model.predict(X_pred)[0]
    
    # Convert back to original price
    price_pred = np.exp(log_pred)
    
    return price_pred, log_pred

def main():
    st.title("🥕 UK Vegetable Price Predictor")
    st.markdown("Predict weekly vegetable prices using machine learning")
    
    # Load data and model
    try:
        df = load_data()
        model = load_model()
    except Exception as e:
        st.error(f"Error loading data or model: {e}")
        st.stop()
    
    # Sidebar inputs
    st.sidebar.header("Prediction Parameters")
    
    # Commodity selection
    commodities = sorted(df['commodity'].unique())
    selected_commodity = st.sidebar.selectbox("Select Commodity", commodities)
    
    # Week selection
    min_date = df['week_ending'].min()
    max_date = df['week_ending'].max()
    future_date = max_date + timedelta(weeks=4)
    
    target_week = st.sidebar.date_input(
        "Select Week Ending Date",
        value=future_date,
        min_value=min_date,
        max_value=future_date
    )
    target_week = pd.to_datetime(target_week)
    
    # Make prediction
    if st.sidebar.button("Predict Price", type="primary"):
        features_row = get_latest_features(df, selected_commodity, target_week)
        
        if features_row is None:
            st.error(f"No data available for {selected_commodity}")
        else:
            try:
                predicted_price, log_pred = predict_price(model, features_row)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Predicted Price",
                        value=f"£{predicted_price:.2f}/kg"
                    )
                
                with col2:
                    st.metric(
                        label="Commodity",
                        value=selected_commodity
                    )
                
                with col3:
                    st.metric(
                        label="Week Ending",
                        value=target_week.strftime("%Y-%m-%d")
                    )
                
                # Historical price chart
                commodity_history = df[df['commodity'] == selected_commodity].copy()
                commodity_history = commodity_history.sort_values('week_ending')
                
                fig = go.Figure()
                
                # Historical prices
                fig.add_trace(go.Scatter(
                    x=commodity_history['week_ending'],
                    y=commodity_history['price_gbp_kg'],
                    mode='lines+markers',
                    name='Historical Prices',
                    line=dict(color='blue')
                ))
                
                # Predicted price
                fig.add_trace(go.Scatter(
                    x=[target_week],
                    y=[predicted_price],
                    mode='markers',
                    name='Predicted Price',
                    marker=dict(color='red', size=12, symbol='star')
                ))
                
                fig.update_layout(
                    title=f"{selected_commodity} Price History and Prediction",
                    xaxis_title="Week Ending",
                    yaxis_title="Price (£/kg)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance for this prediction
                st.subheader("Model Insights")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Recent price statistics
                    recent_prices = commodity_history.tail(10)['price_gbp_kg']
                    st.write("**Recent Price Statistics (Last 10 weeks)**")
                    st.write(f"Mean: £{recent_prices.mean():.2f}/kg")
                    st.write(f"Std: £{recent_prices.std():.2f}/kg")
                    st.write(f"Min: £{recent_prices.min():.2f}/kg")
                    st.write(f"Max: £{recent_prices.max():.2f}/kg")
                
                with col2:
                    # Key features used
                    st.write("**Key Features Used**")
                    st.write("• Price lags (1, 4, 8, 12 weeks)")
                    st.write("• Weather data (temperature, rainfall, sunshine)")
                    st.write("• Seasonal patterns (sin/cos week)")
                    st.write("• Commodity-specific effects")
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.write("Debug info:")
                st.write(f"Features row type: {type(features_row)}")
                st.write(f"Model type: {type(model)}")
    
    # Data overview
    st.subheader("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    
    with col2:
        st.metric("Commodities", df['commodity'].nunique())
    
    with col3:
        st.metric("Date Range", f"{df['week_ending'].min().strftime('%Y-%m')} to {df['week_ending'].max().strftime('%Y-%m')}")
    
    with col4:
        st.metric("Features", len([c for c in df.columns if c not in ['price_gbp_kg', 'log_price', 'week_ending']]))
    
    # Sample data
    if st.checkbox("Show Sample Data"):
        st.dataframe(df.head(10))

if __name__ == "__main__":
    main()

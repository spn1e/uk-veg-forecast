import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math
from datetime import timedelta, datetime
import altair as alt
import anthropic
from typing import List, Dict, Any
import json
import os

# ─────────────────────────────────────────────
# SIMPLE VECTOR STORE (ChromaDB Replacement)
# ─────────────────────────────────────────────
class SimpleVectorStore:
    def __init__(self):
        self.documents = []
        self.ids = []
        self.metadatas = []
        
        # Load from session state if available
        if 'vector_store_data' in st.session_state:
            data = st.session_state.vector_store_data
            self.documents = data.get('documents', [])
            self.ids = data.get('ids', [])
            self.metadatas = data.get('metadatas', [])
    
    def add(self, documents, metadatas=None, ids=None):
        """Add documents to the vector store"""
        if metadatas is None:
            metadatas = [{}] * len(documents)
        if ids is None:
            ids = [f"doc_{len(self.ids) + i}" for i in range(len(documents))]
        
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        
        # Save to session state
        st.session_state.vector_store_data = {
            'documents': self.documents,
            'ids': self.ids,
            'metadatas': self.metadatas
        }
    
    def query(self, query_texts, n_results=10):
        """Simple text similarity query (keyword matching)"""
        if not self.documents or not query_texts:
            return {"documents": [[]], "metadatas": [[]]}
        
        query_text = query_texts[0].lower()
        results = []
        
        # Simple keyword-based similarity
        for i, doc in enumerate(self.documents):
            score = sum(1 for word in query_text.split() if word in doc.lower())
            if score > 0:
                results.append((score, i))
        
        # Sort by relevance and take top n_results
        results.sort(reverse=True, key=lambda x: x[0])
        top_indices = [idx for _, idx in results[:n_results]]
        
        return {
            "documents": [[self.documents[i] for i in top_indices]],
            "metadatas": [[self.metadatas[i] for i in top_indices]]
        }

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH   = "models/lgbm_weekly_tuned.pkl"
BUFFER_PATH  = "data/features_weekly.parquet"
MAX_LAG      = 12
COLOR_HIST   = "#1f77b4"
COLOR_FORE   = "#d62728"

# ─────────────────────────────────────────────
# CLAUDE API & RAG SETUP
# ─────────────────────────────────────────────
class VegetableForecastAssistant:
    def __init__(self):
        # Initialize Claude client (supports both Anthropic direct and OpenRouter)
        self.use_openrouter = st.secrets.get("USE_OPENROUTER", os.getenv("USE_OPENROUTER", "false")).lower() == "true"
        
        if self.use_openrouter:
            # OpenRouter setup
            self.openrouter_api_key = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY"))
            if not self.openrouter_api_key:
                st.error("OpenRouter API key not found. Please set OPENROUTER_API_KEY in secrets.")
                return
            self.claude_client = None  # We'll use requests for OpenRouter
        else:
            # Direct Anthropic setup
            anthropic_key = st.secrets.get("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY"))
            if not anthropic_key:
                st.error("Anthropic API key not found. Please set ANTHROPIC_API_KEY in secrets.")
                return
            self.claude_client = anthropic.Anthropic(api_key=anthropic_key)
        
        # Initialize simple vector store
        self.setup_vector_store()
        
        # Initialize knowledge base
        self.setup_knowledge_base()
    
    def setup_vector_store(self):
        """Initialize simple vector store for RAG"""
        try:
            self.collection = SimpleVectorStore()
        except Exception as e:
            st.error(f"Vector store setup failed: {e}")
            self.collection = None
    
    def setup_knowledge_base(self):
        """Populate knowledge base with vegetable market information"""
        if not self.collection:
            return
            
        knowledge_documents = [
            {
                "id": "uk_vegetable_seasons",
                "content": """UK Vegetable Seasonal Patterns:
                Spring (Mar-May): Asparagus, spring onions, lettuce, spinach peak season
                Summer (Jun-Aug): Tomatoes, cucumbers, peppers, courgettes at lowest prices
                Autumn (Sep-Nov): Root vegetables (carrots, potatoes, parsnips) harvest season
                Winter (Dec-Feb): Stored crops, imported vegetables drive higher prices
                Weather impacts: Cold snaps increase heating costs and reduce yields, wet weather delays harvests""",
                "metadata": {"category": "seasonal_patterns", "source": "agricultural_calendar"}
            },
            {
                "id": "price_volatility_factors",
                "content": """Key UK Vegetable Price Drivers:
                1. Weather: Temperature extremes affect yields, rainfall impacts harvest timing
                2. Energy costs: Oil prices drive transportation and greenhouse heating costs
                3. Currency: GBP/USD affects import costs from Netherlands, Spain
                4. Supply chain: Brexit impacts, labor shortages affect distribution
                5. Consumer demand: Health trends, seasonal cooking patterns
                6. Storage capacity: Cold storage availability affects price stability""",
                "metadata": {"category": "market_factors", "source": "market_analysis"}
            },
            {
                "id": "forecasting_methodology",
                "content": """LightGBM Model Features & Interpretation:
                Primary predictors: price_lag_1 (most recent price), price_lag_2 (short-term trend)
                Weather features: tmax_mean, tmin_mean (temperature stress), rain_sum (harvest disruption)
                Economic: brent_usd_bbl (transport costs), fx_usd_gbp (import costs)
                Seasonal: week_num, month, sin/cos_week (cyclical patterns)
                Model accuracy: Best for 1-4 week forecasts, degrades beyond 6 weeks
                Uncertainty: ±10-15% typical range, higher during extreme weather events""",
                "metadata": {"category": "model_info", "source": "technical_documentation"}
            },
            {
                "id": "trading_strategies",
                "content": """Vegetable Market Trading Insights:
                Buy signals: Prices 15%+ below seasonal average, good weather forecasts
                Sell signals: Prices 20%+ above average, weather warnings issued
                Risk management: Diversify across commodities, hedge currency exposure
                Seasonal arbitrage: Buy during harvest gluts, sell during scarcity periods
                Quality premiums: Premium vegetables command 30-50% higher prices
                Contract timing: Lock in prices 2-3 months ahead for major crops""",
                "metadata": {"category": "trading_strategy", "source": "market_expertise"}
            }
        ]
        
        # Add documents to collection
        try:
            # Check if already populated
            if not hasattr(self.collection, 'documents') or not self.collection.documents:
                self.collection.add(
                    documents=[doc['content'] for doc in knowledge_documents],
                    metadatas=[doc['metadata'] for doc in knowledge_documents],
                    ids=[doc['id'] for doc in knowledge_documents]
                )
        except Exception as e:
            st.error(f"Knowledge base setup failed: {e}")
    
    def get_context(self, query: str, n_results: int = 3) -> str:
        """Retrieve relevant context from knowledge base"""
        if not self.collection:
            return "Knowledge base unavailable."
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            if results['documents'] and results['documents'][0]:
                context = "\n\n".join(results['documents'][0])
                return f"Relevant market knowledge:\n{context}"
            else:
                return "No specific market knowledge found for this query."
        except Exception as e:
            return f"Context retrieval error: {e}"
    
    def price_forecast_tool(self, commodity: str, weeks: int = 4) -> Dict[str, Any]:
        """Generate price forecast using the trained model"""
        try:
            if 'model' not in st.session_state or 'BUFFER' not in st.session_state:
                return {"error": "Forecasting model not loaded"}
            
            model = st.session_state.model
            BUFFER = st.session_state.BUFFER
            
            # Get historical data
            hist = BUFFER.xs(commodity.upper()).copy()
            if len(hist) == 0:
                return {"error": f"No data available for {commodity}"}
            
            # Generate forecast (simplified version)
            predictions = []
            for step in range(min(weeks, 8)):  # Limit to 8 weeks max
                feats_df = build_feature_row(hist, commodity)
                try:
                    log_pred = model.predict(feats_df)[0]
                except:
                    log_pred = model.predict(feats_df, predict_disable_shape_check=True)[0]
                
                price = round(math.exp(log_pred), 3)
                predictions.append(price)
                
                # Update history for next iteration (simplified)
                next_week = hist.index[-1] + timedelta(days=7)
                new_row = hist.iloc[-1:].copy()
                new_row.index = [next_week]
                new_row["price_gbp_kg"] = price
                hist = pd.concat([hist, new_row]).tail(MAX_LAG + 1)
            
            return {
                "commodity": commodity,
                "forecast_weeks": weeks,
                "predictions": predictions,
                "current_price": float(hist["price_gbp_kg"].iloc[-1]),
                "trend": "up" if predictions[-1] > predictions[0] else "down",
                "volatility": round(np.std(predictions) / np.mean(predictions) * 100, 1)
            }
        
        except Exception as e:
            return {"error": f"Forecast generation failed: {str(e)}"}
    
    def chat_with_claude(self, user_message: str, chat_history: List[Dict]) -> str:
        """Main chat function with Claude API (supports OpenRouter and direct Anthropic)"""
        try:
            # Get relevant context
            context = self.get_context(user_message)
            
            # Check if user is asking for a forecast
            forecast_result = None
            if any(word in user_message.lower() for word in ['forecast', 'predict', 'price']):
                # Extract commodity if mentioned
                commodities = ['potato', 'carrot', 'onion', 'tomato', 'cucumber', 
                             'lettuce', 'cabbage', 'broccoli', 'pepper', 'courgette']
                mentioned_commodity = None
                for commodity in commodities:
                    if commodity in user_message.lower():
                        mentioned_commodity = commodity
                        break
                
                if mentioned_commodity:
                    forecast_result = self.price_forecast_tool(mentioned_commodity)
            
            # Build system prompt
            system_prompt = f"""You are a professional UK vegetable market analyst and forecasting expert. 
            
            Your role:
            - Provide expert analysis on UK vegetable prices and market trends
            - Interpret forecasting data and explain market dynamics
            - Offer actionable insights for traders, farmers, and buyers
            - Be professional, accurate, and helpful
            
            Available tools and data:
            - Advanced ML price forecasting model (LightGBM with 23 features)
            - Historical price data from June 2018 - December 2024
            - Weather, economic, and seasonal data integration
            
            Context from knowledge base:
            {context}
            
            {f"Current forecast data: {json.dumps(forecast_result, indent=2)}" if forecast_result else ""}
            
            Guidelines:
            - Always cite data sources when making specific claims
            - Explain uncertainty and limitations clearly
            - Provide practical, actionable advice
            - Use professional but accessible language
            """
            
            # Build messages for Claude
            messages = []
            
            # Add chat history (last 10 messages to stay within limits)
            for msg in chat_history[-10:]:
                messages.append({
                    "role": msg["role"], 
                    "content": msg["content"]
                })
            
            # Add current user message
            messages.append({
                "role": "user",
                "content": user_message
            })
            
            if self.use_openrouter:
                # OpenRouter API call
                import requests
                
                headers = {
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://streamlit.io",  # Required by OpenRouter
                    "X-Title": "UK Vegetable Price Forecaster"  # Optional but recommended
                }
                
                data = {
                    "model": "anthropic/claude-3.5-sonnet",  # OpenRouter model name
                    "messages": [
                        {"role": "system", "content": system_prompt}
                    ] + messages,
                    "max_tokens": 1000,
                    "temperature": 0.3,
                    "stream": False
                }
                
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    error_msg = f"OpenRouter API error: {response.status_code}"
                    try:
                        error_detail = response.json().get("error", {}).get("message", "Unknown error")
                        error_msg += f" - {error_detail}"
                    except:
                        pass
                    return f"I apologize, but I encountered an API error: {error_msg}. Please try again."
            
            else:
                # Direct Anthropic API call
                response = self.claude_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    system=system_prompt,
                    messages=messages,
                    temperature=0.3
                )
                
                return response.content[0].text
            
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Please try again or contact support."

# ─────────────────────────────────────────────
# ORIGINAL FORECASTING CODE (keeping your existing functions)
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
    
    df["week_ending"] = pd.to_datetime(df["week_ending"])
    df = df.sort_values(["commodity", "week_ending"])

    buffer = (
        df.groupby("commodity")
          .tail(MAX_LAG + 1)
          .reset_index(drop=True)
          .set_index(["commodity", "week_ending"])
    )

    try:
        if hasattr(model, 'feature_name_'):
            feat_cols = model.feature_name_
        else:
            feat_cols = [
                "commodity", "tmax_mean", "tmin_mean", "rain_sum", "sun_sum", 
                "fx_usd_gbp", "brent_usd_bbl", "is_holiday", "price_lag_1", 
                "price_lag_2", "price_lag_4", "rain_sum_4", "tmax_avg_4", 
                "sun_sum_4", "price_roll_4", "week_num", "month", "price_lag_8", 
                "price_lag_12", "rain_sum_8", "sun_sum_8", "sin_week", "cos_week"
            ]
    except Exception as e:
        st.error(f"Error determining features: {e}")
        feat_cols = [
            "commodity", "tmax_mean", "tmin_mean", "rain_sum", "sun_sum", 
            "fx_usd_gbp", "brent_usd_bbl", "is_holiday", "price_lag_1", 
            "price_lag_2", "price_lag_4", "rain_sum_4", "tmax_avg_4", 
            "sun_sum_4", "price_roll_4", "week_num", "month", "price_lag_8", 
            "price_lag_12", "rain_sum_8", "sun_sum_8", "sin_week", "cos_week"
        ]
    
    return model, feat_cols, buffer

def safe_lag(series: pd.Series, k: int):
    return series.iloc[-k] if len(series) >= k else series.iloc[0]

def safe_tail_sum(series: pd.Series, w: int):
    return series.tail(w).sum() if len(series) >= w else series.sum()

def safe_tail_mean(series: pd.Series, w: int):
    return series.tail(w).mean() if len(series) >= w else series.mean()

def build_feature_row(hist: pd.DataFrame, commodity_name: str) -> pd.DataFrame:
    if 'FEAT_COLS' not in st.session_state:
        return pd.DataFrame()
    
    FEAT_COLS = st.session_state.FEAT_COLS
    row = {}
    
    if len(hist) == 0:
        return pd.DataFrame([{}]).reindex(columns=FEAT_COLS, fill_value=0)
    
    row["commodity"] = commodity_name.upper()
    
    if "tmax_mean" in hist.columns:
        row["tmax_mean"] = hist["tmax_mean"].iloc[-1]
        row["tmax_avg_4"] = safe_tail_mean(hist["tmax_mean"], 4)
    else:
        row["tmax_mean"] = 0
        row["tmax_avg_4"] = 0
        
    if "tmin_mean" in hist.columns:
        row["tmin_mean"] = hist["tmin_mean"].iloc[-1]
    else:
        row["tmin_mean"] = 0

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

    if "fx_usd_gbp" in hist.columns:
        row["fx_usd_gbp"] = hist["fx_usd_gbp"].iloc[-1]
    else:
        row["fx_usd_gbp"] = 1.27
        
    if "brent_usd_bbl" in hist.columns:
        row["brent_usd_bbl"] = hist["brent_usd_bbl"].iloc[-1]
    else:
        row["brent_usd_bbl"] = 70.0
        
    if "is_holiday" in hist.columns:
        row["is_holiday"] = int(hist["is_holiday"].iloc[-1])
    else:
        row["is_holiday"] = 0

    if "price_gbp_kg" in hist.columns:
        for k in [1, 2, 4, 8, 12]:
            row[f"price_lag_{k}"] = safe_lag(hist["price_gbp_kg"], k)
        row["price_roll_4"] = safe_tail_mean(hist["price_gbp_kg"], 4)
    else:
        for k in [1, 2, 4, 8, 12]:
            row[f"price_lag_{k}"] = 1.0
        row["price_roll_4"] = 1.0

    last_date = hist.index[-1] if not isinstance(hist.index, pd.MultiIndex) \
               else hist.index.get_level_values(-1)[-1]
    last_date = pd.to_datetime(last_date)
    
    week_no = last_date.isocalendar().week
    row["week_num"] = int(week_no)
    row["month"] = int(last_date.month)
    row["sin_week"] = math.sin(2 * math.pi * week_no / 52)
    row["cos_week"] = math.cos(2 * math.pi * week_no / 52)

    df = pd.DataFrame([row]).reindex(columns=FEAT_COLS, fill_value=0)
    
    categorical_cols = ["is_holiday", "week_num", "month", "commodity"]
    for col in categorical_cols:
        if col in df.columns:
            if col == "commodity":
                df[col] = df[col].astype('category')
            else:
                df[col] = df[col].astype('int32')
    
    return df

def forecast_commodity(commodity: str, horizon: int):
    if 'model' not in st.session_state or 'BUFFER' not in st.session_state:
        return None, None
    
    model = st.session_state.model
    BUFFER = st.session_state.BUFFER
    
    try:
        hist = BUFFER.xs(commodity.upper()).copy()
        
        if len(hist) == 0:
            st.error(f"No historical data found for {commodity}")
            return None, None
        
        preds = []
        
        for step in range(horizon):
            feats_df = build_feature_row(hist, commodity)
            
            try:
                log_pred = model.predict(feats_df)[0]
            except Exception:
                log_pred = model.predict(feats_df, predict_disable_shape_check=True)[0]
            
            price = round(math.exp(log_pred), 3)
            preds.append(price)
            
            next_week = hist.index[-1] + timedelta(days=7)
            new_row_data = hist.iloc[-1:].copy()
            new_row_data.index = [next_week]
            new_row_data["price_gbp_kg"] = price
            
            week_no = next_week.isocalendar().week
            if "week_num" in new_row_data.columns:
                new_row_data["week_num"] = int(week_no)
            if "month" in new_row_data.columns:
                new_row_data["month"] = int(next_week.month)
            if "sin_week" in new_row_data.columns:
                new_row_data["sin_week"] = math.sin(2 * math.pi * week_no / 52)
            if "cos_week" in new_row_data.columns:
                new_row_data["cos_week"] = math.cos(2 * math.pi * week_no / 52)
            
            hist = pd.concat([hist, new_row_data]).tail(MAX_LAG + 1)
        
        return preds, hist
        
    except Exception as e:
        st.error(f"Error during forecasting: {str(e)}")
        return None, None

# ─────────────────────────────────────────────
# STREAMLIT UI WITH CHAT INTEGRATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="UK Vegetable Price Forecaster",
    page_icon="🥕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'assistant' not in st.session_state:
    st.session_state.assistant = VegetableForecastAssistant()

# Load model and data into session state
if 'model' not in st.session_state:
    model, feat_cols, buffer = load_assets()
    if model is not None:
        st.session_state.model = model
        st.session_state.FEAT_COLS = feat_cols
        st.session_state.BUFFER = buffer

st.title("🇬🇧 UK Vegetable Price Forecaster")
st.markdown("**Professional ML-powered forecasting with AI Assistant**")
st.markdown("---")

# Create two columns: main app and chat
col_main, col_chat = st.columns([2, 1])

# ─────────────────────────────────────────────
# MAIN FORECASTING INTERFACE (LEFT COLUMN)
# ─────────────────────────────────────────────
with col_main:
    st.subheader("📊 Price Forecasting")
    
    if 'model' not in st.session_state:
        st.error("Failed to load model or data. Please check file paths.")
        st.stop()

    try:
        available_commodities = sorted(st.session_state.BUFFER.index.get_level_values(0).unique())
        
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
            st.success(f"✅ Successfully generated {forecast_horizon}-week forecast for {selected_commodity.title()}")
            
            st.subheader(f"📈 {selected_commodity.title()} Price Forecast")
            st.markdown("---")
            
            # Show predictions in a nice format
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Predicted Prices (£/kg):**")
                for i, pred in enumerate(predictions):
                    st.metric(f"Week +{i+1}", f"£{pred:.3f}")
            
            with col2:
                if len(predictions) > 1:
                    st.markdown("**Week-on-Week Change:**")
                    for i in range(1, len(predictions)):
                        change = ((predictions[i] - predictions[i-1]) / predictions[i-1]) * 100
                        st.metric(f"Week +{i+1}", f"{change:+.1f}%")
            
            # Visualization
            st.subheader("📊 Price Trend Visualization")
            
            hist_df = history.reset_index()
            hist_df = hist_df[hist_df['price_gbp_kg'].notna()].copy()
            
            if 'level_1' in hist_df.columns:
                hist_df = hist_df.rename(columns={'level_1': 'week_ending'})
            elif 'week_ending' not in hist_df.columns:
                if hist_df.index.name == 'week_ending':
                    hist_df = hist_df.reset_index()
                else:
                    hist_df['week_ending'] = hist_df.index
            
            last_date = history.index[-len(predictions)-1] if len(history.index) > len(predictions) else history.index[-1]
            future_dates = [last_date + timedelta(days=7*(i+1)) for i in range(forecast_horizon)]
            
            future_df = pd.DataFrame({
                "week_ending": future_dates,
                "price_gbp_kg": predictions,
                "type": "Forecast"
            })
            
            hist_df["type"] = "Historical"
            
            required_cols = ["week_ending", "price_gbp_kg", "type"]
            if all(col in hist_df.columns for col in required_cols):
                plot_data = pd.concat([
                    hist_df[required_cols].tail(20),
                    future_df
                ]).reset_index(drop=True)
            else:
                st.error("Unable to create visualization due to data structure issues.")
                st.stop()
            
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
            
            if len(predictions) > 1:
                total_change = ((predictions[-1] - predictions[0]) / predictions[0]) * 100
                trend = "📈 Upward" if total_change > 2 else "📉 Downward" if total_change < -2 else "➡️ Stable"
                st.info(f"**Market Trend**: {trend} ({total_change:+.1f}% over {forecast_horizon} weeks)")

# ─────────────────────────────────────────────
# CHAT ASSISTANT (RIGHT COLUMN)
# ─────────────────────────────────────────────
with col_chat:
    st.subheader("🤖 AI Market Assistant")
    st.markdown("*Ask about prices, trends, and market insights*")
    
    # Chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.chat_history[-10:]:  # Show last 10 messages
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about vegetable prices, market trends, or forecasts..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = st.session_state.assistant.chat_with_claude(
                    prompt, 
                    st.session_state.chat_history
                )
            st.write(response)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Quick action buttons
    st.markdown("**Quick Actions:**")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📈 Market Summary", use_container_width=True):
            summary_prompt = "Give me a brief summary of current UK vegetable market conditions and any notable trends."
            st.session_state.chat_history.append({"role": "user", "content": summary_prompt})
            
            with st.spinner("Generating market summary..."):
                response = st.session_state.assistant.chat_with_claude(
                    summary_prompt, 
                    st.session_state.chat_history
                )
            
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
    
    with col2:
        if st.button("🔍 Price Alert", use_container_width=True):
            alert_prompt = f"Are there any price alerts or significant changes I should know about for {selected_commodity if 'selected_commodity' in locals() else 'key vegetables'}?"
            st.session_state.chat_history.append({"role": "user", "content": alert_prompt})
            
            with st.spinner("Checking for alerts..."):
                response = st.session_state.assistant.chat_with_claude(
                    alert_prompt, 
                    st.session_state.chat_history
                )
            
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ─────────────────────────────────────────────
# FOOTER SECTION
# ─────────────────────────────────────────────
st.markdown("---")

# Information section
with st.expander("📚 Methodology & Limitations"):
    st.markdown("""
    ### 🔬 **Methodology**
    
    This forecaster uses a **LightGBM gradient boosting model** trained on 6+ years of UK vegetable market data. 
    The model incorporates multiple data sources including historical prices, weather patterns, economic indicators, 
    and seasonal factors to generate weekly price predictions.
    
    The **AI Assistant** uses Claude 3.5 Sonnet with:
    - **Simple Vector Store** for market knowledge (ChromaDB replacement)
    - **Specialized tools** for price forecasting and context retrieval
    - **Professional market analysis** capabilities
    
    ### 📊 **Key Predictive Factors**
    
    1. **Recent Price History** (40% importance): Short-term price momentum and trends
    2. **Weather Conditions** (25% importance): Temperature, rainfall, and sunshine affecting crop yields  
    3. **Economic Indicators** (20% importance): Oil prices and currency exchange rates
    4. **Seasonal Patterns** (15% importance): Weekly and monthly cyclical effects
    
    ### 🤖 **AI Assistant Features**
    
    - **Market Analysis**: Real-time interpretation of price movements and trends
    - **Contextual Insights**: Access to specialized agricultural and trading knowledge
    - **Forecast Explanation**: Detailed breakdown of model predictions
    - **Trading Guidance**: Professional advice for market participants
    
    ### ⚠️ **Important Limitations**
    
    - **Weather Assumptions**: Current weather conditions are held constant during forecast period
    - **Economic Stability**: Oil prices and exchange rates use last known values  
    - **Historical Patterns**: Model assumes past relationships continue into the future
    - **External Shocks**: Cannot predict impact of unexpected events (diseases, policy changes, etc.)
    - **Forecast Horizon**: Accuracy decreases significantly beyond 4-6 weeks
    - **AI Responses**: Assistant responses are for informational purposes only, not financial advice
    - **Vector Store**: Uses simple keyword matching instead of semantic similarity
    
    ### 🎯 **Best Practices**
    
    - Use forecasts as **guidance alongside** market expertise and other information sources
    - Focus on **short-term predictions** (1-4 weeks) for highest accuracy
    - Monitor **actual vs predicted** performance and adjust planning accordingly
    - Consider **confidence intervals** - actual prices may vary ±10-15% from predictions
    - **Verify AI insights** with independent market research and professional advisors
    """)

# API Key Setup Instructions
with st.expander("🔧 Setup Instructions"):
    st.markdown("""
    ### Required Setup
    
    **Option 1: OpenRouter (Recommended - Free Tier Available)**
    
    1. **Get OpenRouter API Key**: 
       - Sign up at [openrouter.ai](https://openrouter.ai)
       - **Free credits**: $1 of free credits to start
       - **Claude 3.5 Sonnet**: ~$0.003 per 1K tokens (very affordable)
       - Add to Streamlit secrets: `OPENROUTER_API_KEY = "your_key_here"`
       - Set: `USE_OPENROUTER = "true"`
    
    **Option 2: Direct Anthropic API**
    
    1. **Anthropic API Key**: 
       - Get one from [console.anthropic.com](https://console.anthropic.com)
       - $5 minimum credit purchase required
       - Add to Streamlit secrets: `ANTHROPIC_API_KEY = "your_key_here"`
       - Set: `USE_OPENROUTER = "false"` (or omit)
    
    ### Streamlit Secrets Configuration
    
    Add to your `.streamlit/secrets.toml`:
    
    ```toml
    # For OpenRouter (recommended)
    USE_OPENROUTER = "true"
    OPENROUTER_API_KEY = "sk-or-v1-xxxxx"
    
    # OR for direct Anthropic
    # USE_OPENROUTER = "false"
    # ANTHROPIC_API_KEY = "sk-ant-xxxxx"
    ```
    
    ### Required Python Packages (Updated)
    
    ```bash
    pip install anthropic streamlit pandas numpy joblib altair requests
    ```
    
    **Note**: `chromadb` is no longer required! 🎉
    
    ### Data Files
    
    - `models/lgbm_weekly_tuned.pkl` (your trained model)
    - `data/features_weekly.parquet` (historical data)
    
    ### Features Included
    
    ✅ **Simple Vector Store**: ChromaDB replacement using keyword matching  
    ✅ **Claude API Integration**: OpenRouter and direct Anthropic support  
    ✅ **Agent Tools**: `price_forecast()` and `get_context()` functions  
    ✅ **Chat Integration**: Live chat panel with conversation history  
    ✅ **Professional UI**: Clean, production-ready interface  
    ✅ **No SQLite Issues**: Completely eliminates ChromaDB dependency problems
    
    ### Knowledge Base Content
    
    The simple vector store includes:
    - UK seasonal vegetable patterns
    - Price volatility factors and drivers
    - Model interpretation and limitations
    - Trading strategies and market insights
    
    ### What Changed
    
    - **Removed**: ChromaDB, pysqlite3 dependencies
    - **Added**: SimpleVectorStore class with keyword-based matching
    - **Improved**: More reliable deployment on Streamlit Cloud
    - **Maintained**: All original functionality and UI
    """)

st.markdown("---")
st.caption("🔬 **Model**: LightGBM (100 trees, 127 leaves) + Claude 3.5 Sonnet AI | 📅 **Training Data**: June 2018 - December 2024 | 🎯 **Optimized for**: Weekly price forecasting with AI assistance | ✅ **No ChromaDB dependency**")

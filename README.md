# 🇬🇧 UK Vegetable Price Forecaster + AI Market Assistant

## Live Demo

[👉 Try the app here](https://uk-veg-forecast-hjmyrqy3pb8ryk3hekha2i.streamlit.app/)

![Streamlit](https://img.shields.io/badge/Built with-Streamlit-fc4c02?logo=streamlit)
![License](https://img.shields.io/badge/license-MIT-blue)
![Status](https://img.shields.io/badge/status-Alpha-yellow)

A one‑page Streamlit web‑app that **forecasts weekly wholesale prices for UK vegetables** and pairs the prediction engine with a **conversational AI analyst** (Claude 3.5 Sonnet) that can answer market questions, explain the model and surface domain knowledge.

<img src="/assests/screenshot.png" alt="app screenshot" width="800"/>

---

## ✨ Key Features
| Category | What it does |
|----------|--------------|
| Forecasting | LightGBM model (<0.36 £/kg MAE) predicts 1‑12 weeks ahead |
| Interactive UI | Select crop & horizon, view chart, table, analytics |
| AI Assistant | Claude-powered chat with price\_forecast & get\_context tools |
| Vector RAG | Simple in‑memory keyword store (no SQLite/Chroma issues) |
| Zero‑Ops | Runs entirely on free Streamlit Cloud – no FastAPI back‑end |

---

## 🗂️ Project Structure
```
.
├── streamlit_app.py          ← main app (this repo)
├── models/
│   └── lgbm_weekly_tuned.pkl ← 100‑tree LightGBM model (Git LFS)
├── data/
│   └── features_weekly.parquet ← last 13 weeks buffer for lags
├── assets/
│   └── screenshot.png
└── requirements.txt
```
## 🗺️  Data Pipeline

<p align="center">
  <img src="docs/dataset_collection.png" width="350">
</p>

| Source (raw) | Rows | Period | Notes |
|--------------|------|--------|-------|
| **UK Fruit & Veg Prices** | 24 k | 2014‑2024 | Weekly DEFRA prices |
| **UK Horticulture Dataset**  | 14 k | 2014‑2025 | Additional crop lines |
| **UK Holidays**  | 4 k | 2014‑2025 | Bank‑holiday flags |
| **USD/GBP FX**  | 2.9 k | 2014‑2025 | Daily → weekly AVG |
| **Brent Oil**  | 2.9 k | 2014‑2025 | Energy cost proxy |
| **Weather** (5 UK Met stations, 11 yrs) | 165 × 365 | 2014‑2025 | Temp / Rain / Sunshine |

All are merged into **`features_weekly.parquet`** – the model‑ready feature matrix containing 23 engineered predictors (lags, rolling stats, seasonality, weather, macro).

---

## ⚙️  Tech Stack

<p align="center">

</p>

| Layer | Library / Service | Why |
|-------|-------------------|-----|
| **Model** | **LightGBM** + joblib | Fast tabular GBM, 13 % MAE improvement after tuning |
| **UI / API** | **Streamlit** | One‑file deploy on Streamlit Cloud |
| **RAG** | **SimpleVectorStore** (in‑memory) | Zero external DB; avoids ChromaDB/SQLite issues |
| **LLM** | **Anthropic Claude 3.5 Sonnet** via OpenRouter _(or direct API)_ | Cheap, fast, state‑of‑the‑art reasoning |
| **Vis** | **Altair** | Interactive charts with declarative grammar |
---
  <img src="/assests/techstack.png" width="880">
## 🚀 Quick Start

```bash
# 1. Clone & install
git clone https://github.com/<your‑handle>/uk-veg-forecast.git
cd uk-veg-forecast
pip install -r requirements.txt   # Python ≥3.10

# 2. Add secrets (choose ONE provider)
mkdir -p .streamlit
cat > .streamlit/secrets.toml <<'EOF'
# ▶ Option A – OpenRouter (free credits)
USE_OPENROUTER = "true"
OPENROUTER_API_KEY = "sk-or-v1-xxxxxxxx"

# ▶ Option B – Direct Anthropic
#USE_OPENROUTER = "false"
#ANTHROPIC_API_KEY = "sk-ant-xxxxxxxx"
EOF

# 3. Run locally
streamlit run streamlit_app.py
```

---

## 🛠️ Configuration

| Variable | Where | Purpose |
|----------|-------|---------|
| `MODEL_PATH` | `streamlit_app.py` | points to LightGBM `.pkl` |
| `BUFFER_PATH` | `streamlit_app.py` | 13‑week lag feature cache |
| `OPENROUTER_API_KEY` / `ANTHROPIC_API_KEY` | `secrets.toml` | LLM access |
| `USE_OPENROUTER` | `secrets.toml` | toggle provider |

---

## 🧠 How It Works

1. **Model loading** – `joblib` reads the tuned LightGBM regressor and the last‑13‑weeks feature buffer.  
2. **User selects** commodity & horizon → `_forecast_commodity` iteratively builds feature rows, predicts log‑price, exponentiates.  
3. **Plot & metrics** – Altair chart + average/min/max/volatility cards.  
4. **Chat** – Messages stream through `VegetableForecastAssistant`, which:
   - pulls keyword‑matched docs from the simple vector store,
   - optionally calls `price_forecast_tool`,
   - sends everything to Claude 3.5 Sonnet (via OpenRouter **or** Anthropic API).

---

## 📈 Model Performance

| Split | MAE ( £/kg ) |
|-------|-------------|
| Train (2018‑06→2023‑12) | 0.365 |
| Test (2024) | 0.365 |
| **Overall** | **0.364 (‑13 % vs baseline)** |

*Target: log(price\_gbp\_kg); features: price lags 1/2/4/8/12, 4‑wk roll, weather aggregates, FX & Brent, sine/cosine week, holiday flag.*

---

## 🧩 Extending

* ✅ swap keyword store for real vector DB (ChromaDB, LanceDB)  
* ✅ schedule nightly model retrains (GitHub Actions)  
* ⏳ docker‑compose recipe  
* ⏳ import user CSVs for ad‑hoc commodities  
* ⏳ upgrade to semantic search (OpenAI embeddings)  

---

## 📄 License
MIT – see [`LICENSE`](LICENSE).

---

> Built with ❤️ & 🥕 by spn1e*.

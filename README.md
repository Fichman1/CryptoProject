# Binance Bitcoin Prediction Project

The main objective of this project is to build a full machine-learning pipeline that predicts future cryptocurrency prices (starting with Bitcoin).
The project begins with a reliable data-collection layer from the Binance API, and will include preprocessing, feature engineering (technical indicators), 
model training (LSTM/Random Forest Regressor,XGBoost Regressor),evaluation, and a live forecasting module.
---
## 📌 Features (Current)
* Fetches historical candlestick (kline) data from Binance.
* Supports configurable symbol and interval (e.g., `BTCUSDT`, `5m`).
* Handles Binance API rate limits with retries.
* Uses automatic pagination in chunks of 1000 rows.
* Converts timestamps into `datetime`.
* Converts numeric columns to proper numeric types.
* Removes overlapping candles (duplicate timestamps).
* Saves final data as a CSV inside the `data/` directory.
---
## 📁 Project Structure (Current & Future)
```
project/
│
├── data/                         # Stored CSV output
├── src/
│   ├── data_loader.py            # BinanceDataLoader (current)
│   ├── preprocessing.py          # (future)
│   ├── feature_engineering.py    # (future)
│   ├── model_training.py         # (future)
│   ├── inference.py              # (future)
│   └── utils/                    # Helpers (future)
│
└── README.md
```
---
## 📊 Data Collection (Current)
Example usage:
```python
from data_loader import BinanceDataLoader

loader = BinanceDataLoader(symbol="BTCUSDT", interval="5m")
df = loader.fetch_history(start_str="2017-09-09")
```
The script will download all kline data from the specified start date until now and save it under:
```
data/BTCUSDT_5m_data.csv
```
---
## 🧼 Data Cleaning (Current)
The loader applies minimal, safe cleaning:
* Duplicate candle removal (overlaps returned by Binance)
* Sorting candles by time
* Numeric type conversion
* Optional detection of missing candles (printed only)
Real price spikes are **not removed**, as they represent legitimate market behavior.
---
## 📦 Output (Current)
Each row in the CSV contains:
* `open_time`, `close_time`
* `open`, `high`, `low`, `close`, `volume`
* `quote_asset_volume`
* `number_of_trades`
* `taker_buy_base_asset_volume`
* `taker_buy_quote_asset_volume`
---
# 🔮 **Future Sections (Empty for now)**

## 🧹 Data Preprocessing (Future)

*(To be added later)*
---
## ⚙️ Feature Engineering (Future)
*(To be added later)*
---
## 🧠 Model Training (Future)
*(To be added later)*
---
## 📈 Model Evaluation (Future)
*(To be added later)*
---
## 🔍 Inference / Prediction (Future)
*(To be added later)*
---
## 💾 Model Saving & Loading (Future)
*(To be added later)*
---
## 📉 Live Forecasting Script (Future)
*(To be added later)*
---
## 🧪 Backtesting (Future)
*(To be added later)*
---
## 🛠️ Requirements
```
pandas
requests
```
Install with:
```bash
pip install pandas/ requests --no-warn-script-location
```
---

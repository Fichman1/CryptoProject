import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os

# --- קבועים והגדרות ---
BASE_URL = "https://api.binance.com/api/v3/klines"
COLUMNS = [
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
]

# נתיב דינמי (כדי שיעבוד לכל חברי הצוות)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(BASE_DIR, 'data')

class BinanceDataLoader:
    def __init__(self, symbol='BTCUSDT', interval='5m', save_path=SAVE_PATH):
        self.symbol = symbol
        self.interval = interval
        self.save_path = save_path
        
        # יצירת תיקייה אם לא קיימת
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def _fetch_chunk(self, start_time, limit=1000):
        params = {
            'symbol': self.symbol,
            'interval': self.interval,
            'startTime': start_time,
            'limit': limit
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(BASE_URL, params=params, timeout=10)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    print(f"Rate limit hit. Sleeping for 60 seconds...")
                    time.sleep(60)
                else:
                    print(f"Error {response.status_code}: {response.text}")
            except requests.exceptions.RequestException as e:
                print(f"Connection error (Attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(2)
        return None

    def fetch_history(self, start_str, end_str=None):
        print(f"--- Starting data collection for {self.symbol} ({self.interval}) ---")
        
        start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_str).timestamp() * 1000) if end_str else int(time.time() * 1000)
        
        all_data = []
        current_start = start_ts
        
        while current_start < end_ts:
            print(f"Fetching data from: {datetime.fromtimestamp(current_start / 1000)}")
            
            data_chunk = self._fetch_chunk(current_start)
            
            if not data_chunk or len(data_chunk) == 0:
                break
            
            all_data.extend(data_chunk)
            last_close_time = data_chunk[-1][6]
            current_start = last_close_time + 1
            time.sleep(0.1)

        # עיבוד ל-DataFrame
        df = pd.DataFrame(all_data, columns=COLUMNS)
        
        # המרת טיפוסי נתונים
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
        
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # --- חזרה לשמירה בפורמט CSV ---
        self._save_to_csv(df)
        
        print(f"--- Data Collection Complete. Fetched {len(df)} rows. ---")
        return df

    def _save_to_csv(self, df):
        filename = os.path.join(self.save_path, f"{self.symbol}_{self.interval}_data.csv")
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

if __name__ == "__main__":
    loader = BinanceDataLoader(symbol='BTCUSDT', interval='5m')
    
    df = loader.fetch_history(start_str='2017-09-09', end_str=None)
    
    expected_file = os.path.join(SAVE_PATH, f"BTCUSDT_{loader.interval}_data.csv")
    if os.path.exists(expected_file):
        print(f"Success! CSV file found at: {expected_file}")
    else:
        print("Error: File was not created.")
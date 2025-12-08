import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os

BASE_URL = "https://api.binance.com/api/v3/klines"
COLUMNS = [
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
]

class BinanceDataLoader:
    def __init__(self, symbol='BTCUSDT', interval='5m', save_path='data'):#manage paths and parameters from binance
        self.symbol = symbol
        self.interval = interval
        self.save_path = save_path

        # create directory if not exists - under data
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def _fetch_chunk(self, start_time, limit=1000):#API call to fetch data chunk
        params = {
            'symbol': self.symbol,
            'interval': self.interval,
            'startTime': start_time,
            'limit': limit
        }
        # Retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(BASE_URL, params=params, timeout=10)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429: # Rate Limit
                    print(f"Rate limit hit. Sleeping for 60 seconds...")
                    time.sleep(60)
                else:
                    print(f"Error {response.status_code}: {response.text}")
            except requests.exceptions.RequestException as e:
                print(f"Connection error (Attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(2) # waiting before retrying
        return None # if failed after retries

    def fetch_history(self, start_str, end_str=None): #main function to fetch historical data
        print(f"--- Starting data collection for {self.symbol} ---")
        
        #making timestamps
        start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_str).timestamp() * 1000) if end_str else int(time.time() * 1000)
        
        all_data = []
        current_start = start_ts
        
        while current_start < end_ts:
            print(f"Fetching data from: {datetime.fromtimestamp(current_start / 1000)}")
            data_chunk = self._fetch_chunk(current_start)
            if not data_chunk:
                print("Failed to retrieve data chunk. Stopping.")
                break
            if len(data_chunk) == 0:
                print("No more data available.")
                break
            all_data.extend(data_chunk)
    
            #overlap prevention
            last_close_time = data_chunk[-1][6]
            current_start = last_close_time + 1
           #pause to respect rate limits
            time.sleep(0.1)
       #dataframe creation
        df = pd.DataFrame(all_data, columns=COLUMNS)
        
        #convert columns to appropriate types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        #CLEANING BLOCK 
        #Convert ALL numeric columns
        numeric_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
        ]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        #Remove duplicated candles (API overlap)
        df.drop_duplicates(subset='open_time', inplace=True)
        #Sort candles by time
        df.sort_values('open_time', inplace=True)
        #Reset index after cleaning
        df.reset_index(drop=True, inplace=True)
       
        self._save_to_csv(df)
        print(f"--- Data Collection Complete. Fetched {len(df)} rows. ---")
        return df

    def _save_to_csv(self, df): #save raw data to csv
        filename = f"{self.save_path}/{self.symbol}_{self.interval}_data.csv"
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

if __name__ == "__main__": #what to collect
    loader = BinanceDataLoader(symbol='BTCUSDT', interval='5m')
    df = loader.fetch_history(start_str='2017-09-09')
    print(df.head())
    print(df.tail())
    
    
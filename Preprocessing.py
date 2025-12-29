import pandas as pd
import numpy as np
import os

# --- הגדרות ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# קובץ הנתונים (5 דקות)
DATA_PATH = os.path.join(BASE_DIR, 'data', 'BTCUSDT_5m_data.csv') 

# --- השינוי הגדול: חלון זמן של יממה שלמה ---
# 288 נרות של 5 דקות = 24 שעות אחורה
SEQ_LENGTH = 288  

PREDICT_AHEAD = 1 # חיזוי של נר אחד קדימה (5 דקות קדימה)
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

# עמודות לאימון
FEATURE_COLS = ['log_ret', 'volume']
TARGET_COL = 'log_ret'

class DataPreprocessor:
    def __init__(self):
        # ללא סקיילר (משתמשים ב-Log Returns גולמי)
        pass

    def load_and_clean_data(self, filepath):
        print(f"Loading data from {filepath}...")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}. Please run API_test.py first.")

        df = pd.read_csv(filepath)
        
        df['open_time'] = pd.to_datetime(df['open_time'])
        df = df.sort_values('open_time')
        df = df.drop_duplicates(subset=['open_time'])
        df = df.ffill().dropna()

        # חישוב תשואה לוגריתמית
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        
        # טיפול ב-Volume (הקטנה סדרי גודל)
        df['volume'] = np.log(df['volume'] + 1) 
        
        df = df.dropna()
        
        print(f"Data converted to Log Returns. Rows: {len(df)}")
        return df

    def create_sequences(self, data, target, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length - PREDICT_AHEAD + 1):
            x = data[i:(i + seq_length)]
            y = target[i + seq_length + PREDICT_AHEAD - 1]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def process(self):
        # 1. טעינה ועיבוד
        try:
            df = self.load_and_clean_data(DATA_PATH)
        except FileNotFoundError as e:
            print(e)
            return

        # המרה ל-Numpy
        data = df[FEATURE_COLS].values
        target = df[TARGET_COL].values

        # 2. חלוקה ל-Train/Val/Test
        n = len(data)
        train_end = int(n * TRAIN_SPLIT)
        val_end = int(n * (TRAIN_SPLIT + VAL_SPLIT))

        train_data = data[:train_end]
        train_target = target[:train_end]
        
        val_data = data[train_end:val_end]
        val_target = target[train_end:val_end]
        
        test_data = data[val_end:]
        test_target = target[val_end:]

        # 3. יצירת רצפים (Sequences)
        print(f"Creating sliding windows (SEQ_LENGTH={SEQ_LENGTH} steps = 24h context)...")
        X_train, y_train = self.create_sequences(train_data, train_target, SEQ_LENGTH)
        X_val, y_val = self.create_sequences(val_data, val_target, SEQ_LENGTH)
        X_test, y_test = self.create_sequences(test_data, test_target, SEQ_LENGTH)

        # 4. שמירה
        save_dir = os.path.join(BASE_DIR, 'processed_data')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(save_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(save_dir, 'y_val.npy'), y_val)
        np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(save_dir, 'y_test.npy'), y_test)

        print("--- Preprocessing Complete ---")
        print(f"X_train shape: {X_train.shape}") # אמור להיות (Samples, 288, 2)
        print(f"X_test shape: {X_test.shape}")

if __name__ == "__main__":
    processor = DataPreprocessor()
    processor.process()
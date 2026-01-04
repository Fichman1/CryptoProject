import pandas as pd
import numpy as np
import os
import ta # ספריית הניתוח הטכני

# --- הגדרות ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# קובץ הנתונים (5 דקות)
DATA_PATH = os.path.join(BASE_DIR, 'data', 'BTCUSDT_5m_data.csv')

# חלון זמן של יממה שלמה (24 שעות * 12 נרות בשעה)
SEQ_LENGTH = 120

PREDICT_AHEAD = 1
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

# --- רשימת הפיצ'רים המעודכנת ---
# המודל יקבל עכשיו 5 נתונים בכל צעד זמן במקום 2
FEATURE_COLS = ['log_ret', 'rsi', 'rsi_change', 'macd_diff', 'macd_slope', 'bb_pband_change', 'volume']
TARGET_COL = 'log_ret'

class DataPreprocessor:
    def __init__(self):
        pass

    def load_and_clean_data(self, filepath):
        print(f"Loading data from {filepath}...")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath)

        df['open_time'] = pd.to_datetime(df['open_time'])
        df = df.sort_values('open_time')
        df = df.drop_duplicates(subset=['open_time'])
        df = df.ffill().dropna()

        # --- 1. חישוב Log Returns (הלב של המודל) ---
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

        # --- 2. חישוב אינדיקטורים טכניים (Technical Indicators) ---

        # A. RSI (Relative Strength Index) - מומנטום
        # טווח מקורי: 0-100. נחלק ב-100 כדי שיהיה בין 0-1
        df['rsi'] = ta.momentum.rsi(df['close'], window=14) / 100.0
        df['rsi_change'] = df['rsi'].diff(periods=3) # Change in RSI over last 3 periods
        # B. MACD (Moving Average Convergence Divergence) - זיהוי מגמה
        # נשתמש בהפרש (MACD Diff) שמראה את עוצמת המגמה
        macd = ta.trend.MACD(df['close'])
        macd_raw = macd.macd_diff()
        df['macd'] = (macd_raw - macd_raw.rolling(window=100).mean()) / (macd_raw.rolling(window=100).std() + 1e-9)
        df['macd_diff'] = macd.macd_diff()
        df['macd_slope'] = df['macd_diff'].diff(periods=2) # Velocity of trend change

        # C. Bollinger Bands - תנודתיות
        # נשתמש ברוחב הרצועה (Band Width) כדי לדעת אם השוק "רגוע" או "סוער"
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_width'] = bb.bollinger_pband()
        df['bb_pband_change'] = df['bb_width'].diff(periods=1)


        # Log Returns and Direction (Sign)
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        # Optional: Add a 'lagged' return so the model sees recent volatility
        df['log_ret_lag'] = df['log_ret'].shift(1)
        # --- 3. טיפול ב-Volume ---
        # Log scaling volume is good, but we also standardize it to match the other features
        df['volume'] = np.log(df['volume'] + 1)
        df['volume'] = (df['volume'] - df['volume'].mean()) / (df['volume'].std() + 1e-9)

        # האינדיקטורים יוצרים ערכי NaN בהתחלה (כי צריך היסטוריה כדי לחשב אותם)
        # למשל RSI צריך 14 נרות אחורה. אז נמחק את השורות הריקות.
        df = df.dropna()

        print(f"Features added: RSI, MACD, BB_Width. Total Rows: {len(df)}")
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

        # 3. יצירת רצפים
        print(f"Creating sliding windows (SEQ_LENGTH={SEQ_LENGTH}, Features={len(FEATURE_COLS)})...")
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

        print("--- Preprocessing Complete (With Indicators) ---")
        print(f"X_train shape: {X_train.shape}") # אמור להיות (Samples, 288, 5)
        print(f"X_test shape: {X_test.shape}")

if __name__ == "__main__":
    processor = DataPreprocessor()
    processor.process()
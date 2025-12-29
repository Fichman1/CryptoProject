import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# --- הגדרות ---
# שימוש בנתיב דינמי כדי למנוע שגיאות אצל חברי הצוות
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'BTCUSDT_1h_data.csv')

SEQ_LENGTH = 60   # אורך החלון (כמה נרות אחורה המודל רואה)
PREDICT_AHEAD = 1 # חיזוי של צעד אחד קדימה
TRAIN_SPLIT = 0.8 # 80% לאימון
VAL_SPLIT = 0.1   # 10% לולידציה

# כעת אנחנו מתמקדים בחיזוי התשואה הלוגריתמית
# אנחנו נשתמש בתשואה ובנפח המסחר כפיצ'רים
FEATURE_COLS = ['log_ret', 'volume'] 
TARGET_COL = 'log_ret' 

class DataPreprocessor:
    def __init__(self):
        # סקיילר לנרמול הנתונים לטווח 0-1 (חשוב ל-LSTM)
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_and_clean_data(self, filepath):
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        initial_len = len(df)
        
        # המרת תאריך לפורמט זמן
        df['open_time'] = pd.to_datetime(df['open_time'])
        df = df.sort_values('open_time')
        
        # הסרת כפילויות
        df = df.drop_duplicates(subset=['open_time'])
        
        # [cite: 41] טיפול בערכים חסרים (Fill-Forward)
        df = df.ffill().dropna()
        
        print(f"Cleaned data: {len(df)} rows (removed {initial_len - len(df)} duplicates/NaNs)")
        return df

    def compute_log_returns(self, data):
        """
        מחשב log returns עבור הנתונים הפיננסיים
        """
        # העתקה של הדאטה
        returns_data = data.copy()

        # חישוב log returns עבור מחירים (open, high, low, close)
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            col_idx = FEATURE_COLS.index(col)
            # log(P_t / P_{t-1}) = log(P_t) - log(P_{t-1})
            returns_data[1:, col_idx] = np.log(data[1:, col_idx] / data[:-1, col_idx])

        # חישוב log עבור volume (מוסיף 1 כדי למנוע log(0))
        volume_idx = FEATURE_COLS.index('volume')
        returns_data[:, volume_idx] = np.log(data[:, volume_idx] + 1)

        # הסרת השורה הראשונה (NaN מה-log returns)
        returns_data = returns_data[1:]

        return returns_data

    def create_sequences(self, data, target, seq_length):
        """
        יצירת Sliding Windows
        ממיר את הדאטה למבנה תלת-ממדי: (Samples, Time Steps, Features)
        """
        xs, ys = [], []
        for i in range(len(data) - seq_length - PREDICT_AHEAD + 1):
            x = data[i:(i + seq_length)]
            y = target[i + seq_length + PREDICT_AHEAD - 1]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def process(self):
        # 1. טעינה וחישוב Log Returns
        df = self.load_and_clean_data(DATA_PATH)
        
        # שמירת רק העמודות הרלוונטיות
        data = df[FEATURE_COLS].values
        target = df[[TARGET_COL]].values # משתמשים בסוגריים כפולים כדי לשמור על מימד

        # 2. חלוקה ל-Train/Val/Test לפני הנרמול (כדי למנוע Data Leakage)
        n = len(data)
        train_end = int(n * TRAIN_SPLIT)
        val_end = int(n * (TRAIN_SPLIT + VAL_SPLIT))

        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        # 3.  נרמול הנתונים (לומדים את הסקאלה רק מה-Train!)
        print("Normalizing data...")
        self.scaler.fit(train_data) # לומד מינימום ומקסימום רק מהאימון
        
        # שמירת הסקיילר לשימוש עתידי (בשלב החיזוי בזמן אמת)
        if not os.path.exists('models'):
            os.makedirs('models')
        joblib.dump(self.scaler, os.path.join(BASE_DIR, 'models', 'scaler.pkl'))
        
        # ביצוע הטרנספורמציה
        train_scaled = self.scaler.transform(train_data)
        val_scaled = self.scaler.transform(val_data)
        test_scaled = self.scaler.transform(test_data)

        # נרמול ה-Target בנפרד (כדי שנוכל להפוך את הנרמול בסוף ולדעת מחיר אמיתי)
        # הערה: כרגע אנחנו משתמשים באותו סקיילר לכל הפיצ'רים. 
        # לצורך הפשטות נשתמש בעמודת ה-Close מתוך המידע המנורמל כ-Target.
        target_col_idx = FEATURE_COLS.index(TARGET_COL)
        
        # 4. יצירת רצפים (Windows)
        print("Creating sliding windows...")
        X_train, y_train = self.create_sequences(train_scaled, train_scaled[:, target_col_idx], SEQ_LENGTH)
        X_val, y_val = self.create_sequences(val_scaled, val_scaled[:, target_col_idx], SEQ_LENGTH)
        X_test, y_test = self.create_sequences(test_scaled, test_scaled[:, target_col_idx], SEQ_LENGTH)

        # 5. שמירת קבצי NumPy
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
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")

if __name__ == "__main__":
    processor = DataPreprocessor()
    processor.process()
   
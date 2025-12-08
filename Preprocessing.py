import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# --- הגדרות ---
DATA_PATH = 'data/BTCUSDT_1h_data.csv'  # הנתיב לקובץ שיצרנו בשלב הקודם
SEQ_LENGTH = 60  # אורך החלון (כמה נרות אחורה המודל רואה כדי לחזות)
PREDICT_AHEAD = 1 # כמה צעדים קדימה חוזים (1 = הנר הבא)
TRAIN_SPLIT = 0.8 # 80% לאימון
VAL_SPLIT = 0.1   # 10% לולידציה (נשאר 10% לטסט)

# עמודות שאנחנו משתמשים בהן לאימון (Features)
FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume']
# העמודה אותה אנחנו מנסים לחזות (Target)
TARGET_COL = 'close'

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_and_clean_data(self, filepath):
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # המרת תאריך לפורמט זמן
        df['open_time'] = pd.to_datetime(df['open_time'])
        df = df.sort_values('open_time')

        # [cite: 42] הסרת כפילויות
        initial_len = len(df)
        df = df.drop_duplicates(subset=['open_time'])
        
        # [cite: 41] טיפול בערכים חסרים (Fill-Forward)
        df = df.ffill().dropna()
        
        print(f"Cleaned data: {len(df)} rows (removed {initial_len - len(df)} duplicates/NaNs)")
        return df

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
        # 1. טעינה וניקוי
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
        joblib.dump(self.scaler, 'models/scaler.pkl')
        print("Scaler saved to models/scaler.pkl")

        # ביצוע הנרמול בפועל
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

        # 5. שמירת התוצאות
        if not os.path.exists('processed_data'):
            os.makedirs('processed_data')

        np.save('processed_data/X_train.npy', X_train)
        np.save('processed_data/y_train.npy', y_train)
        np.save('processed_data/X_val.npy', X_val)
        np.save('processed_data/y_val.npy', y_val)
        np.save('processed_data/X_test.npy', X_test)
        np.save('processed_data/y_test.npy', y_test)

        print("--- Preprocessing Complete ---")
        print(f"Train shape: {X_train.shape}") # (דוגמאות, 60, 5)
        print(f"Validation shape: {X_val.shape}")
        print(f"Test shape: {X_test.shape}")

if __name__ == "__main__":
    processor = DataPreprocessor()
    processor.process()
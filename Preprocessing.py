import pandas as pd
import numpy as np
import os
import ta # ספריית הניתוח הטכני
from sklearn.preprocessing import StandardScaler


# --- הגדרות ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# קובץ הנתונים (5 דקות)
DATA_PATH = os.path.join(BASE_DIR, 'data', 'BTCUSDT_5m_data.csv')

# חלון זמן של יממה שלמה (24 שעות * 12 נרות בשעה)
SEQ_LENGTH = 120

PREDICT_AHEAD = 1
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
STEP = 5

# --- רשימת הפיצ'רים המעודכנת ---
# המודל יקבל עכשיו 5 נתונים בכל צעד זמן במקום 2
FEATURE_COLS = ['log_ret', 'log_ret_lag_2', 'rsi', 'rsi_change', 'macd', 'macd_slope', 'bb_pband_change', 'volume', 'volume_z', 'vol_spike']
#TARGET_COL = 'log_ret'
TARGET_COL = 'bb_pband' # במקום 'log_ret'

class DataPreprocessor:
    def __init__(self):
        pass

    def check_feature_correlations(self, df, feature_cols):
        """
        Check and print correlations between features to identify redundancy.
        """
        print("\n--- Feature Correlations ---")
        # חישוב מטריצת הקורלציה בין הפיצ'רים
        corr_matrix = df[feature_cols].corr()
        print(corr_matrix)
        
        print("\nHigh correlations (>0.8 or <-0.8) may indicate redundancy:")
        for i in range(len(feature_cols)):
            for j in range(i+1, len(feature_cols)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    print(f"  {feature_cols[i]} vs {feature_cols[j]}: {corr_val:.3f}")

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
        df['log_ret'] = np.log((df['close'] / df['close'].shift(1))* 10) # הוספת ערך קטן למניעת בעיות לוג

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
        df['bb_pband'] = bb.bollinger_pband()
        df['bb_pband_change'] = df['bb_pband'].diff(periods=1)


        # Log Returns and Direction (Sign)
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        # Optional: Add a 'lagged' return so the model sees recent volatility
        df['log_ret_lag'] = df['log_ret'].shift(1)
        df['log_ret_lag_2'] = df['log_ret'].shift(2)
        # --- 3. טיפול ב-Volume ---
        # Log scaling volume is good, but we also standardize it to match the other features
        df['volume'] = np.log(df['volume'] + 1)
        df['volume'] = (df['volume'] - df['volume'].mean()) / (df['volume'].std() + 1e-9)
        df['vol_ma'] = df['volume'].rolling(window=20).mean()
        df['vol_std'] = df['volume'].rolling(window=20).std()
        
        # Z-Score של הנפח: כמה סטיות תקן הנפח הנוכחי רחוק מהממוצע
        df['volume_z'] = (df['volume'] - df['vol_ma']) / (df['vol_std'] + 1e-9)
        
        # Binary Spike: 1 אם הנפח גבוה פי 2 מהממוצע, אחרת 0
        df['vol_spike'] = (df['volume'] > (df['vol_ma'] * 2)).astype(float)

        # האינדיקטורים יוצרים ערכי NaN בהתחלה (כי צריך היסטוריה כדי לחשב אותם)
        # למשל RSI צריך 14 נרות אחורה. אז נמחק את השורות הריקות.
        df = df.dropna()

        print(f"Features added: RSI, MACD, BB_Width. Total Rows: {len(df)}")
        return df

    def create_sequences(self, data, target, seq_length, step=1):
        xs, ys = [], []
        # אנחנו רצים בקפיצות של 'step' כדי להוריד כפילות נתונים
        for i in range(0, len(data) - seq_length - PREDICT_AHEAD + 1, step):
            # ה-X מכיל את ההיסטוריה עד הנקודה הנוכחית
            x = data[i : (i + seq_length)]
            
            # ה-y הוא בדיוק הנר שאנחנו רוצים לחזות בעתיד
            # למשל: אם seq_length=120 ו-predict_ahead=1, נחזה את נר 121
            y = target[i + seq_length + PREDICT_AHEAD - 1]
            
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def process(self):
        # 1. טעינה ועיבוד ראשוני של הנתונים (כולל אינדיקטורים)
        try:
            df = self.load_and_clean_data(DATA_PATH)
        except FileNotFoundError as e:
            print(e)
            return

        # בדיקת קורלציות (אופציונלי)
        self.check_feature_correlations(df, FEATURE_COLS)

        # 2. הפרדה לפיצ'רים ומטרות (X ו-y)
        data = df[FEATURE_COLS].values
        target = df[TARGET_COL].values

        # 3. חלוקה כרונולוגית ל-Train/Val/Test
        n = len(data)
        train_end = int(n * TRAIN_SPLIT)
        val_end = int(n * (TRAIN_SPLIT + VAL_SPLIT))

        train_data = data[:train_end]
        train_target = target[:train_end]

        val_data = data[train_end:val_end]
        val_target = target[train_end:val_end]

        test_data = data[val_end:]
        test_target = target[val_end:]

        # --- שלב קריטי: נרמול הנתונים (Standardization) ---
        # אנחנו מאמנים את הסקיילר רק על נתוני ה-Train כדי למנוע Data Leakage
        scaler = StandardScaler()
        train_data_scaled = scaler.fit_transform(train_data)
        
        
        # משתמשים באותו סקיילר עבור ה-Validation וה-Test
        val_data_scaled = scaler.transform(val_data)
        test_data_scaled = scaler.transform(test_data)

        # 4. יצירת רצפים (Sliding Windows)
        # שים לב: אנחנו משתמשים בנתונים המנורמלים
        print(f"Creating sliding windows (SEQ_LENGTH={SEQ_LENGTH}, Features={len(FEATURE_COLS)})...")
        X_train, y_train = self.create_sequences(train_data_scaled, train_target, SEQ_LENGTH,STEP)
        X_val, y_val = self.create_sequences(val_data_scaled, val_target, SEQ_LENGTH,STEP)
        X_test, y_test = self.create_sequences(test_data_scaled, test_target, SEQ_LENGTH,STEP)

        # 5. שמירת הנתונים המעובדים לתיקיית processed_data
        save_dir = os.path.join(BASE_DIR, 'processed_data')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(save_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(save_dir, 'y_val.npy'), y_val)
        np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(save_dir, 'y_test.npy'), y_test)

        print("--- Preprocessing Complete (Normalized & Sequenced) ---")
        print(f"X_train shape: {X_train.shape}") 
        print(f"X_test shape: {X_test.shape}")

if __name__ == "__main__":
    processor = DataPreprocessor()
    processor.process()
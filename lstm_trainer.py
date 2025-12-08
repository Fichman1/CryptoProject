import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# --- הגדרות נתיבים (דינמי - יעבוד לכולם) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'processed_data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# --- פרמטרים לאימון (Hyperparameters) ---
BATCH_SIZE = 64      # כמה דוגמאות המודל רואה במכה אחת [cite: 126]
EPOCHS = 50          # מספר מקסימלי של פעמים שהמודל עובר על כל הדאטה
LEARNING_RATE = 0.001 # קצב הלמידה [cite: 127]

def load_data():
    """טעינת הנתונים המעובדים (קבצי .npy)"""
    print("Loading data from:", DATA_DIR)
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
    y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
    return X_train, y_train, X_val, y_val, X_test, y_test

def build_model(input_shape):
    """בניית ארכיטקטורת LSTM"""
    model = Sequential()
    
    # שכבת כניסה
    model.add(Input(shape=input_shape))
    
    # שכבת LSTM ראשונה (מחזירה רצף לשכבה הבאה)
    model.add(LSTM(units=64, return_sequences=True)) 
    model.add(Dropout(0.2)) # מניעת Overfitting

    # שכבת LSTM שנייה (לא מחזירה רצף, אלא וקטור סופי)
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))

    # שכבת יציאה (ניבוי ערך אחד: מחיר הסגירה)
    model.add(Dense(units=1))

    # קומפילציה [cite: 90]
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), 
                  loss='mean_squared_error')
    
    return model

def train():
    # 1. טעינת הנתונים
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    print(f"Data loaded! Train shape: {X_train.shape}")

    # 2. בניית המודל
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.summary()

    # 3. הכנת Callbacks (שמירה ועצירה)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # שמירת המודל הטוב ביותר (Checkpoints) [cite: 130]
    checkpoint_path = os.path.join(MODEL_DIR, 'best_lstm_model.keras')
    checkpoint = ModelCheckpoint(
        checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1
    )
    
    # עצירה מוקדמת אם אין שיפור (Early Stopping)
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    # 4. ביצוע האימון (Training Loop) [cite: 59]
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )

    # 5. הצגת גרף הלמידה (Loss)
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training Process (Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 6. הערכה סופית על ה-Test Set [cite: 92]
    print("Evaluating on Test Set...")
    predictions = model.predict(X_test)
    
    # המרת התחזיות חזרה למחיר דולרי אמיתי (Inverse Transform)
    # נטען את הסקיילר ששמרנו בשלב הקודם
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        # טריק: הסקיילר מצפה ל-5 עמודות, אנחנו צריכים "לזייף" אותן כדי להמיר רק את ה-Close
        # (בפרויקט מורכב יותר בונים סקיילר נפרד ל-Target, כאן נעשה קירוב לצורך התצוגה)
        # נציג כרגע את הגרף בערכים מנורמלים (0-1) כדי לוודא שהמודל עובד
        pass 

    # חישוב מדדים [cite: 134]
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    print(f"Final Test RMSE: {rmse:.4f}")
    print(f"Final Test R^2: {r2:.4f}")

    # 7. ויזואליזציה של התחזית מול האמת [cite: 141]
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[:100], label='Actual Price (Normalized)', color='blue') # מציג רק 100 ראשונים כדי שיהיה ברור
    plt.plot(predictions[:100], label='Predicted Price (Normalized)', color='red', linestyle='--')
    plt.title('LSTM Prediction vs Actual (First 100 Test Samples)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train()
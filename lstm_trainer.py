import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mplfinance as mpf
import pandas as pd
from sklearn.preprocessing import StandardScaler
import ta
# import joblib # לא בשימוש כרגע כי אנחנו עובדים עם לוג ללא סקיילר

# --- הגדרות ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'processed_data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

# --- Fine Tuning Hyperparameters ---
BATCH_SIZE = 16         # הקטנו כדי להוסיף רעש חיובי לאימון
EPOCHS = 100
LEARNING_RATE = 0.0005  # הקטנו קצת כדי שהלימוד יהיה עדין יותר
HIDDEN_DIM = 128        # הגדלנו את "המוח" של המודל
NUM_LAYERS = 2
DROPOUT = 0.1     # ביטלנו את ה-Dropout כדי לא לאבד מידע עדין

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_loss = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_lstm_model.pth'))
        self.best_loss = val_loss

def load_data():
    print("Loading data...")
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
    y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
    return X_train, y_train, X_val, y_val, X_test, y_test

def build_demo_loaders(fraction=0.05):
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Subsample
    train_size = int(len(X_train) * fraction)
    val_size = int(len(X_val) * fraction)
    test_size = int(len(X_test) * fraction)

    X_train = X_train[-train_size:]
    y_train = y_train[-train_size:]
    X_val = X_val[-val_size:]
    y_val = y_val[-val_size:]
    X_test = X_test[-test_size:]
    y_test = y_test[-test_size:]
    return X_train, y_train, X_val, y_val, X_test, y_test


class DirectionalLogCoshLoss(nn.Module):
    def __init__(self, directional_penalty=0.5):
        super(DirectionalLogCoshLoss, self).__init__()
        self.directional_penalty = directional_penalty

    def forward(self, y_pred, y_true):
        # 1. Log-Cosh Loss
        # log(cosh(x)) is smooth and robust to outliers
        loss = torch.log(torch.cosh(y_pred - y_true + 1e-12))

        # 2. Directional Penalty
        # If signs are different, product is negative
        # We penalize when sign(pred) != sign(true)
        penalty = torch.where(y_pred * y_true < 0, self.directional_penalty, 0.0)

        return torch.mean(loss + penalty)

# --- פונקציה לחישוב דיוק כיווני ---
def calculate_directional_accuracy(y_true, y_pred, threshold=0.001):
    # חישוב השינוי באחוזים
    true_change = np.diff(y_true) / y_true[:-1]
    pred_change = np.diff(y_pred) / y_pred[:-1]
    
    # מסנן: נחשב דיוק רק כשהמודל חזה שינוי משמעותי (מעל הסף)
    # זה מדמה מסחר אמיתי: אנחנו לא נכנסים לעסקה על כל פיפס קטן
    significant_moves_mask = np.abs(pred_change) > threshold
    
    if np.sum(significant_moves_mask) == 0:
        print("No significant moves predicted.")
        return 0.0

    # סינון הנתונים
    true_direction = np.sign(true_change[significant_moves_mask])
    pred_direction = np.sign(pred_change[significant_moves_mask])
    
    correct = np.sum(true_direction == pred_direction)
    total = len(true_direction)
    
    accuracy = (correct / total) * 100
    print(f"\n--- Accuracy on Strong Moves (Threshold {threshold}): {accuracy:.2f}% ({total}/{len(y_pred)} candles) ---")
    return accuracy

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. טעינת הנתונים הגולמיים
    #X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test = build_demo_loaders(0.5)

    # --- שיפור קריטי: נרמול (Standardization) ---
    # אנחנו מנרמלים את כל הפיצ'רים כדי שכולם יהיו באותה סקאלה (Mean=0, Std=1)
    # זה מונע מה-Volume "לחנוק" את ה-Log Returns הקטנים
    scaler = StandardScaler()
    
    # משטחים את הנתונים לצורך נרמול ומחזירים לצורה המקורית
    N_samples, Seq_len, N_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, N_features)
    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(N_samples, Seq_len, N_features)
    
    # משתמשים באותו סקיילר (Fit על Train בלבד!) עבור Val ו-Test
    X_val_scaled = scaler.transform(X_val.reshape(-1, N_features)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, N_features)).reshape(X_test.shape)

    # --- הסרת ה-SCALE_FACTOR הידני ---
    # המודל ילמד את ה-y המקורי. Log-Cosh יודע להתמודד עם זה.
    
    # המרה ל-Tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor), batch_size=BATCH_SIZE, shuffle=False)

    # מודל קטן יותר (2 שכבות, 64 יחידות) כדי למנוע Underfitting עצלן
    model = LSTMModel(input_size=N_features, hidden_size=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
    
    criterion = DirectionalLogCoshLoss(directional_penalty=2) # עונש כיווני חזק יותר
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # --- REDUCE LR ON PLATEAU אגרסיבי ---
    # מחכה רק 3 אפוקים (Patience) וחותך את ה-LR ב-70% (Factor=0.3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=7, threshold=1e-5,min_lr=0.0001
    )

    early_stopping = EarlyStopping(patience=25, verbose=True, delta=1e-6)

    # ... (המשך לולאת האימון כפי שהיה, רק בלי חלוקה ב-SCALE_FACTOR בסוף) ...

    print("Starting training (with Target Scaling * 100)...")
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        print(f'Current LR: {optimizer.param_groups[0]["lr"]}')

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # --- הערכה ---
    print("\nEvaluating on Test Set...")
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_lstm_model.pth')))
    model.eval()

    predictions = []
    with torch.no_grad():
        for X_batch, in test_loader: # שים לב: הטסט לא מכיל y בקריאה הזו
            outputs = model(X_batch)
            predictions.extend(outputs.squeeze().cpu().numpy())

    predictions = np.array(predictions)

    # --- חלוקה חזרה בפקטור ---
    

    # חישוב מדדים על המידע המקורי
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"Final Test RMSE (Original Scale): {rmse:.6f}")

    # 8. Visualization (OHLCV Candles with Predicted Price Overlay)
    print("\nGenerating Candle Visualization...")

    # --- A. Load Original Data for OHLC context ---
    csv_path = os.path.join(BASE_DIR, 'data', 'BTCUSDT_5m_data.csv')
    df_full = pd.read_csv(csv_path)
    df_full['open_time'] = pd.to_datetime(df_full['open_time'])
    df_full.set_index('open_time', inplace=True)

    # --- B. Reconstruct Indices to match Preprocessing.py ---
    # We must replicate the split logic to find exactly which candles are in the Test Set
    # Note: Ensure these match the constants in Preprocessing.py
    SEQ_LENGTH = 120
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1

    n = len(df_full)
    train_end = int(n * TRAIN_SPLIT)
    val_end = int(n * (TRAIN_SPLIT + VAL_SPLIT))

    # The test set in Preprocessing starts at val_end
    # But the first TARGET (y) appears after SEQ_LENGTH steps
    test_start_index = val_end + SEQ_LENGTH

    # Slice the original dataframe to match the size of 'predictions'
    # We limit it to the number of predictions we actually have
    df_test_candles = df_full.iloc[test_start_index : test_start_index + len(predictions)].copy()

   # --- C. שחזור מחיר חזוי מתוך Bollinger %B ---
    # אנחנו צריכים את הרצועות המקוריות כדי להמיר מ-0..1 חזרה למחיר בדולרים
    bb = ta.volatility.BollingerBands(df_full['close'], window=20, window_dev=2)
    upper_band = bb.bollinger_hband().iloc[test_start_index : test_start_index + len(predictions)].values
    lower_band = bb.bollinger_lband().iloc[test_start_index : test_start_index + len(predictions)].values

    # נוסחת השחזור: מחיר = רצועה תחתונה + (תחזית * רוחב הרצועות)
    predicted_bb_price = lower_band + (np.array(predictions) * (upper_band - lower_band))

    # Add to DataFrame for plotting
    df_test_candles['Predicted_Close'] = pd.Series(predicted_bb_price).rolling(window=3).mean().fillna(method='bfill').values
    # --- D. Plotting with mplfinance ---
    # We zoom in on the first 150 candles for clarity
    ZOOM_SAMPLES = 576
    df_plot = df_test_candles.head(ZOOM_SAMPLES)

    # Create the overlay plot (The predicted price line)
    # panel=0 means it draws on the main chart
    apdict = mpf.make_addplot(df_plot['Predicted_Close'], type='line', color='red', linestyle='--', width=1.5, panel=0)

    # Plot Settings
    title_text = f'BTC/USDT Prediction vs Actual\nRMSE (Log Ret): {rmse:.5f}'

    mpf.plot(
        df_plot,
        type='candle',        # Show candles
        style='yahoo',        # Green/Red style
        addplot=apdict,       # Add the prediction line
        volume=True,          # Show volume at the bottom
        title=title_text,
        ylabel='Price (USDT)',
        figsize=(14, 8),
        tight_layout=True
    )
    # --- ויזואליזציה 1: גרף תהליך הלמידה (Loss) ---
    # מראה האם המודל למד והאם היה Overfitting
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training Process (Scaled Huber Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()



    # --- ויזואליזציה 2: גרף קווי של התחזית מול המציאות ---
    # מראה את הביצועים על הנתונים הגולמיים (ללא נרות, רק קו)
    # אנו מתמקדים ב-150 הצעדים הראשונים כדי לראות פרטים
    plt.figure(figsize=(14, 7))
    # y_test - המציאות (כחול)
    plt.plot(y_test[:576], label='Actual Log Return', color='blue', alpha=0.7)

    # predictions - התחזית של המודל (אדום)
    plt.plot(predictions[:576], label='Predicted Log Return', color='red', linewidth=1.5)

    plt.title(f'LSTM Prediction Performance (Bollinger %B)\nRMSE: {rmse:.5f}')
    plt.xlabel('Time Steps (5m intervals)')
    plt.ylabel('Bollinger %B Value (0-1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    print("Visualization displayed.")

   
        # --- הוסף את הקריאה הזו בסוף פונקציית train, אחרי חישוב ה-RMSE ---
    # שים לב: אנחנו שולחים את המחירים המשוחזרים (predicted_prices) ואת המחירים האמיתיים
    actual_prices = df_test_candles['close'].values
    # predicted_prices כבר מחושב אצלך בקוד הויזואליזציה
    calculate_directional_accuracy(actual_prices, predicted_bb_price)


if __name__ == "__main__":
    train()

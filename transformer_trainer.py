import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import mplfinance as mpf
import pandas as pd
import math

# --- הגדרות ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'processed_data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# --- Hyperparameters (מותאם ל-Transformer) ---
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.0005  # טרנספורמרים צריכים קצב לימוד קצת יותר עדין בהתחלה
SCALE_FACTOR = 500.0    # שומרים על המודל "אגרסיבי"

# הגדרות הרשת
D_MODEL = 64           # גודל הייצוג הפנימי (צריך להיות זוגי)
N_HEADS = 4            # מספר ה"עיניים" שדרכן המודל מסתכל על הנתונים
NUM_LAYERS = 2         # מספר שכבות הטרנספורמר
DROPOUT = 0.2

# ---------------------------------------------------------
# 1. מנגנון הזרקת הזמן (Positional Encoding)
# ---------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

# ---------------------------------------------------------
# 2. מודל ה-Transformer שלנו
# ---------------------------------------------------------
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super(TimeSeriesTransformer, self).__init__()

        # שלב 1: הרחבת הפיצ'רים (12) לגודל ה-d_model (64)
        self.input_linear = nn.Linear(input_dim, d_model)

        # שלב 2: הוספת הבנת הזמן
        self.pos_encoder = PositionalEncoding(d_model)

        # שלב 3: ליבת הטרנספורמר (Encoder)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True # קריטי כדי שיעבוד עם (Batch, Seq, Feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # שלב 4: כיווץ חזרה לתחזית אחת (מחיר סגירה)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (Batch, Seq_Length, Input_Dim) -> (64, 120, 12)

        x = self.input_linear(x)          # -> (64, 120, 64)
        x = self.pos_encoder(x)           # מוסיפים את חותמת הזמן

        # מעבר בטרנספורמר
        output = self.transformer_encoder(x) # -> (64, 120, 64)

        # לוקחים רק את התובנה של הצעד האחרון בסיקוונס (הכי קרוב לעתיד)
        last_step_output = output[:, -1, :]  # -> (64, 64)

        # חיזוי המחיר
        prediction = self.fc(last_step_output) # -> (64, 1)
        return prediction

# ---------------------------------------------------------
# שאר רכיבי המערכת (ללא שינוי, עבדו מעולה)
# ---------------------------------------------------------
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
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_transformer_model.pth'))
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

class DirectionalLogCoshLoss(nn.Module):
    def __init__(self, directional_penalty=10.0): # העונש האגרסיבי נשאר!
        super(DirectionalLogCoshLoss, self).__init__()
        self.directional_penalty = directional_penalty

    def forward(self, y_pred, y_true):
        loss = torch.log(torch.cosh(y_pred - y_true + 1e-12))
        penalty = torch.where(y_pred * y_true < 0, self.directional_penalty, 0.0)
        return torch.mean(loss + penalty)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    y_train = y_train * SCALE_FACTOR
    y_val = y_val * SCALE_FACTOR

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test), batch_size=BATCH_SIZE, shuffle=False)

    # יצירת מודל הטרנספורמר במקום ה-LSTM
    model = TimeSeriesTransformer(
        input_dim=X_train.shape[2],
        d_model=D_MODEL,
        nhead=N_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    criterion = DirectionalLogCoshLoss(directional_penalty=10.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    early_stopping = EarlyStopping(patience=7, verbose=True)

    print("Starting training (Transformer Architecture)...")
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.view(-1), y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs.view(-1), y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # --- הערכה וויזואליזציה ---
    print("\nEvaluating Transformer on Test Set...")
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_transformer_model.pth')))
    model.eval()

    predictions = []
    with torch.no_grad():
        for X_batch, in test_loader:
            outputs = model(X_batch)
            predictions.extend(outputs.view(-1).cpu().numpy())

    predictions = np.array(predictions) / SCALE_FACTOR
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    correct_direction = np.sign(predictions) == np.sign(y_test)
    accuracy = np.mean(correct_direction) * 100

    print(f"\n======== RESULTS ========")
    print(f"Model: Transformer (Attention)")
    print(f"RMSE: {rmse:.6f}")
    print(f"Directional Accuracy: {accuracy:.2f}%")
    print(f"=========================\n")

    # --- ויזואליזציה נרות (Candles) ---
    print("Generating Candle Visualization...")
    csv_path = os.path.join(BASE_DIR, 'data', 'BTCUSDT_5m_data.csv')
    df_full = pd.read_csv(csv_path)
    df_full['open_time'] = pd.to_datetime(df_full['open_time'])
    df_full.set_index('open_time', inplace=True)

    SEQ_LENGTH = 120
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1

    n = len(df_full)
    val_end = int(n * (TRAIN_SPLIT + VAL_SPLIT))
    test_start_index = val_end + SEQ_LENGTH

    df_test_candles = df_full.iloc[test_start_index : test_start_index + len(predictions)].copy()
    previous_closes = df_full.iloc[test_start_index-1 : test_start_index + len(predictions)-1]['close'].values

    predicted_prices = previous_closes * np.exp(predictions)
    df_test_candles['Predicted_Close'] = predicted_prices

    ZOOM_SAMPLES = 150
    df_plot = df_test_candles.head(ZOOM_SAMPLES)

    prediction_plot = mpf.make_addplot(
        df_plot['Predicted_Close'], type='line', color='blue', width=2.0, linestyle='-', panel=0
    )

    title_text = f'Transformer BTC/USDT\nRMSE: {rmse:.5f} | Dir Acc: {accuracy:.2f}%'
    mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
    s  = mpf.make_mpf_style(marketcolors=mc)

    mpf.plot(
        df_plot, type='candle', style=s, addplot=prediction_plot, volume=True,
        title=title_text, ylabel='Price (USDT)', figsize=(14, 8), tight_layout=True
    )

    # --- ויזואליזציה Loss מול מציאות ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Transformer Training Process')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(14, 7))
    plt.plot(y_test[:150], label='Actual Log Return', color='blue', alpha=0.7)
    plt.plot(predictions[:150], label='Transformer Prediction', color='red', linewidth=1.5)
    plt.title(f'Transformer Prediction Performance\nRMSE: {rmse:.5f} | Accuracy: {accuracy:.2f}%')
    plt.xlabel('Time Steps (5m intervals)')
    plt.ylabel('Log Return Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    train()
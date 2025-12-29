import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# --- הגדרות ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'processed_data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# היפר-פרמטרים (ניתן לשינוי לפי דרישות SRS)
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.5

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
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
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_lstm_model.pth'))
        self.best_loss = val_loss

def load_data():
    print("Loading data...")
    # טעינת קבצי ה-NumPy
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
    y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
    return X_train, y_train, X_val, y_val, X_test, y_test

def train():
    # בדיקת זמינות GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. טעינת הנתונים
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    print(f"Data loaded! Train shape: {X_train.shape}")

    # המרת ל-Tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    # יצירת DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. בניית המודל
    input_size = X_train.shape[2]
    model = LSTMModel(input_size=input_size).to(device)
    print(model)

    # 3. הגדרת Loss ו-Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. הכנת Early Stopping
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    early_stopping = EarlyStopping(patience=5, verbose=True)

    # 5. ביצוע האימון (Training Loop)
    print("Starting training...")
    train_losses = []
    val_losses = []

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

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Early Stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # 6. הצגת גרף הלמידה (Loss)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training Process (Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.show()

    # 7. הערכה סופית על ה-Test Set
    print("Evaluating on Test Set...")
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_lstm_model.pth')))
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            predictions.extend(outputs.squeeze().cpu().numpy())
            actuals.extend(y_batch.cpu().numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # חישוב מדדים
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    print(f"Final Test RMSE: {rmse:.4f}")
    print(f"Final Test R^2: {r2:.4f}")

    # 8. Enhanced Visualization
    plt.figure(figsize=(12, 6))

    # If you have your scaler, use:
    # actuals_rescaled = scaler.inverse_transform(actuals.reshape(-1, 1))
    # predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1))

    plt.plot(actuals[:100], label='Actual', color='#1f77b4', linewidth=2)
    plt.plot(predictions[:100], label='Predicted', color='#ff7f0e', linestyle='--', linewidth=2)

    # Add a fill between to show the error/gap
    plt.fill_between(range(100), actuals[:100], predictions[:100], color='gray', alpha=0.2)

    plt.title(f'LSTM Test Performance (RMSE: {rmse:.4f})', fontsize=14)
    plt.xlabel('Time Steps (Test Set)', fontsize=12)
    plt.ylabel('Price (Normalized)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import joblib # לא בשימוש כרגע כי אנחנו עובדים עם לוג ללא סקיילר

# --- הגדרות ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'processed_data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# היפר-פרמטרים
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2 # הורדתי ל-0.2 (0.5 זה קצת אגרסיבי מדי ל-Time Series ועשוי להקשות על הלמידה)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # batch_first=True: הקלט מגיע בצורה (batch, seq, features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # שכבת Dropout נוספת אחרי ה-LSTM ולפני השכבה הלינארית
        self.dropout = nn.Dropout(dropout)
        
        # שכבת יציאה לחיזוי ערך יחיד
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # אתחול מצבים נסתרים (Hidden States)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # הרצת ה-LSTM
        # out מכיל את הפלט של כל שלבי הזמן
        out, _ = self.lstm(x, (h0, c0))
        
        # לוקחים רק את הפלט של הצעד האחרון (Many-to-One)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class EarlyStopping:
    """מחלקה לעצירת האימון כשאין שיפור ב-Validation"""
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
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
    y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
    return X_train, y_train, X_val, y_val, X_test, y_test

def train():
    # בדיקת חומרה (GPU vs CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. טעינת הנתונים
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    print(f"Data loaded! Train shape: {X_train.shape}")

    # המרה ל-Tensors של PyTorch
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
    model = LSTMModel(input_size=input_size, hidden_size=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
    # print(model) # אפשר להדפיס כדי לראות את המבנה

    # 3. הגדרת Loss ו-Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. הכנת תיקייה לשמירה
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    early_stopping = EarlyStopping(patience=5, verbose=True)

    # 5. לולאת האימון
    print("Starting training...")
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        # --- שלב האימון ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()                   # איפוס גרדיאנטים
            outputs = model(X_batch)                # Forward
            loss = criterion(outputs.squeeze(), y_batch) # חישוב שגיאה (squeeze מוריד מימדים מיותרים)
            loss.backward()                         # Backward
            optimizer.step()                        # עדכון משקולות
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # --- שלב הולידציה ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        # בדיקת Early Stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # 6. הצגת גרף הלמידה
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training Process (Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.show()

    # 7. הערכה סופית על ה-Test Set
    print("\nEvaluating on Test Set...")
    # טעינת המודל הטוב ביותר שנשמר
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
    r2 = r2_score(actuals, predictions) # הערה: ב-Log Returns ה-R2 יהיה נמוך מאוד, וזה תקין!
    
    print(f"Final Test RMSE: {rmse:.5f}")
    print(f"Final Test R^2: {r2:.5f}")

    # 8. ויזואליזציה (זום על 100 דוגמאות ראשונות)
    plt.figure(figsize=(12, 6))
    plt.plot(actuals[:150], label='Actual Log Return', color='blue')
    plt.plot(predictions[:150], label='Predicted Log Return', color='red', linestyle='--')
    plt.title(f'LSTM Prediction vs Actual (Log Returns)\nRMSE: {rmse:.5f}')
    plt.xlabel('Time Steps')
    plt.ylabel('Log Return Value')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train()
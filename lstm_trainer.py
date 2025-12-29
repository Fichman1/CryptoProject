import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

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
DROPOUT = 0.2

# בדיקה אם יש GPU זמין (מאיץ משמעותית את האימון)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. הגדרת המודל (PyTorch Style) ---
class CryptoLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.2):
        super(CryptoLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # שכבת LSTM
        # batch_first=True אומר שהקלט מגיע בצורה (batch, seq, features)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout)
        
        # שכבת Linear סופית לחיזוי המחיר
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # אתחול המצבים הנסתרים (Hidden States) לאפס
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        
        # הרצת ה-LSTM
        # out מכיל את הפלט של כל שלבי הזמן
        out, _ = self.lstm(x, (h0, c0))
        
        # אנחנו לוקחים רק את הפלט של הצעד האחרון (החיזוי הסופי)
        out = out[:, -1, :]
        
        # העברה בשכבה הלינארית
        out = self.fc(out)
        return out

# --- 2. טעינת נתונים והכנת DataLoaders ---
def load_data():
    print("Loading data...")
    # טעינת קבצי ה-NumPy
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
    y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

    # המרה ל-Tensors של PyTorch
    train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    val_data = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
    test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    # יצירת DataLoaders (מנהלים את ה-Batches)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    
    return train_loader, val_loader, test_loader, X_train.shape[2]

# --- 3. לולאת האימון (Training Loop) ---
def train_model():
    train_loader, val_loader, test_loader, input_dim = load_data()
    
    model = CryptoLSTM(input_dim=input_dim, hidden_dim=HIDDEN_DIM, 
                       num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
    
    criterion = nn.MSELoss() # פונקציית הפסד
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # אופטימייזר

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    print("Starting training...")
    for epoch in range(EPOCHS):
        # --- שלב האימון ---
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()       # איפוס גרדיאנטים
            outputs = model(X_batch)    # Forward pass
            loss = criterion(outputs, y_batch.unsqueeze(1)) # חישוב שגיאה
            loss.backward()             # Backward pass
            optimizer.step()            # עדכון משקולות
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- שלב הולידציה ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad(): # אין צורך לחשב גרדיאנטים בולידציה
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.unsqueeze(1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

        # שמירת המודל הטוב ביותר (Checkpoint)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_lstm_model.pth'))
            # print("  Saved best model!")

    # --- 4. ויזואליזציה של תהליך הלמידה ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training Process')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.show()

    # --- 5. הערכה סופית על ה-Test Set ---
    print("\nEvaluating on Test Set...")
    # טעינת המשקולות הכי טובות
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_lstm_model.pth')))
    model.eval()
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.numpy())
            
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # חישוב מדדים
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    
    print(f"Final Test RMSE: {rmse:.4f}")
    print(f"Final Test R^2: {r2:.4f}")
    
    # גרף השוואה
    plt.figure(figsize=(12, 6))
    plt.plot(actuals[:100], label='Actual', color='blue')
    plt.plot(predictions[:100], label='Predicted', color='red', linestyle='--')
    plt.title('Prediction vs Actual (First 100 Test Samples)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_model()
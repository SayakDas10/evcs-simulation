import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================
# Model Definition
# ============================
class SmallNN(nn.Module):
    def __init__(self):
        super(SmallNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)


# ============================
# Training Function
# ============================
def train_model(model, optimizer, criterion, X_train, y_train, epochs=200, ckpt_dir='checkpoints'):
    os.makedirs(ckpt_dir, exist_ok=True)
    losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}]  Loss: {loss.item():.6f}")

        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, ckpt_path)
            print(f"✅ Checkpoint saved: {ckpt_path}")

    return losses


# ============================
# Plotting Function
# ============================
def plot_results(y_true, y_pred, losses=None, save_dir='plots', suffix=''):
    os.makedirs(save_dir, exist_ok=True)

    # Actual vs Predicted Scatter
    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, y_pred, color='blue', alpha=0.6)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             color='red', linestyle='--')
    plt.title('Actual vs Predicted MCP')
    plt.xlabel('Actual MCP (Rs/MWh)')
    plt.ylabel('Predicted MCP (Rs/MWh)')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'predictions_vs_actual{suffix}.png'))
    plt.close()

    # Loss Curve
    if losses is not None:
        plt.figure(figsize=(8, 4))
        plt.plot(losses)
        plt.title('Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'training_loss_curve{suffix}.png'))
        plt.close()

    print(f"📊 Plots saved to: {save_dir}/")


# ============================
# Testing Function (with metrics)
# ============================
def test_with_checkpoint(model_class, checkpoint_path, X_test, y_test, scaler_y, save_dir='plots'):
    # Initialize model and load checkpoint
    model = model_class()
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"🔹 Loaded checkpoint: {checkpoint_path} (epoch {checkpoint['epoch']})")

    # Predictions
    with torch.no_grad():
        preds = model(X_test).numpy()
        preds_orig = scaler_y.inverse_transform(preds)
        y_test_orig = scaler_y.inverse_transform(y_test.numpy())

    # Compute metrics
    mae = mean_absolute_error(y_test_orig, preds_orig)
    rmse = np.sqrt(mean_squared_error(y_test_orig, preds_orig))
    r2 = r2_score(y_test_orig, preds_orig)

    print("\n📈 Evaluation Metrics:")
    print(f"MAE  = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"R²   = {r2:.4f}")

    # Plot
    suffix = f"_epoch_{checkpoint['epoch']}"
    plot_results(y_test_orig, preds_orig, save_dir=save_dir, suffix=suffix)

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ============================
# Main Script
# ============================
if __name__ == "__main__":
    # === Load Excel Data ===
    df = pd.read_excel('RTM_Market Snapshot.xlsx')
    TRAIN = False
    CHECKPOINT = "checkpoints/checkpoint_epoch_200.pth"
    

    X = df[['Purchase Bid (MW)', 'Sell Bid (MW)', 'MCV (MW)', 'Final Scheduled Volume (MW)']].values
    y = df['MCP (Rs/Mwh)'].values.reshape(-1, 1)

    # === Train-Test Split ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # === Normalize ===
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)

    # === Convert to Tensors ===
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # === Initialize Model, Loss, Optimizer ===
    model = SmallNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # === Train Model ===
    if TRAIN:
        losses = train_model(model, optimizer, criterion, X_train, y_train, epochs=200)

    # === Evaluate from Checkpoint ===
    if CHECKPOINT:
        last_ckpt = CHECKPOINT
        if os.path.exists(last_ckpt):
            metrics = test_with_checkpoint(SmallNN, last_ckpt, X_test, y_test, scaler_y, save_dir='plots')
            print(f"\n✅ Final Metrics: {metrics}")
        else:
            print("⚠️ No checkpoint found to test.")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# === Step 1: Read Excel File ===
# Replace 'data.xlsx' with your file path
df = pd.read_excel('dynamic_price/RTM_Market Snapshot.xlsx')

# Ensure the column names match your Excel sheet
X = df[['Final Scheduled Volume (MW)']].values  # Feature (must be 2D for sklearn)
y = df['MCP (Rs/Mwh)'].values           # Target

# print(X[:10,0])
# exit()

# === Step 2: Split into Train/Test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Step 3: Train Linear Regression Model ===
model = LinearRegression()
model.fit(X_train, y_train)

# === Step 4: Predict on Test Data ===
y_pred = model.predict(X_test)

# === Step 5: Print Parameters and Accuracy ===
print(f"Weight (w): {model.coef_[0]:.4f}")
print(f"Bias (b): {model.intercept_:.4f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# === Step 6: Plot Test Data and Regression Line ===
plt.figure(figsize=(8,5))
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Fitted Line')
plt.title('Linear Regression: Power Demand vs Price')
plt.xlabel('Power Demand')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

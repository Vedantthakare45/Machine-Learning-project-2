
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Generate a housing dataset
np.random.seed(42)

size = np.random.randint(500, 2500, 100)
age = np.random.randint(1, 30, 100)

# Price is measured in lakhs
price = (
    10
    + 0.05 * size
    - 0.5 * age
    + np.random.normal(0, 5, 100)
)

# Create a DataFrame
data = pd.DataFrame({
    "Size": size,
    "Age": age,
    "Price": price
})

print("Housing Dataset:")
print(data.head())

# Step 2: Separate features and target
X = data[["Size", "Age"]]
y = data["Price"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Predict house prices
y_pred = model.predict(X_test)

print("\nActual Prices:")
print(y_test.head().values)

print("\nPredicted Prices:")
print(y_pred[:5])

# Step 5: Calculate RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nMean Squared Error:", mse)
print("Root Mean Squared Error:", rmse, "lakhs")

# Predict the price of a new house
new_house = pd.DataFrame({
    "Size": [1500],
    "Age": [5]
})

predicted_price = model.predict(new_house)

print("\nPredicted price for new house:",
      round(predicted_price[0], 2), "lakhs")

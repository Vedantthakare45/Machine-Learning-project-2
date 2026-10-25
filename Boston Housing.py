import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load Boston Housing dataset
url = "https://lib.stat.cmu.edu/datasets/boston"

raw_df = pd.read_csv(url, sep=r"\s+", skiprows=22, header=None)

data = np.hstack([
    raw_df.values[::2, :],
    raw_df.values[1::2, :2]
])

target = raw_df.values[1::2, 2]

X = pd.DataFrame(data)
y = target

print("Dataset shape:", X.shape)

# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train both models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42)
}

results = {"Model": [], "MAE": [], "RMSE": []}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    results["Model"].append(name)
    results["MAE"].append(mae)
    results["RMSE"].append(rmse)

# 4. Display evaluation results
results_df = pd.DataFrame(results)
print("\nRegression Evaluation Results:")
print(results_df.round(3))

# 5. Plot bar chart
results_df.set_index("Model")[["MAE", "RMSE"]].plot(
    kind="bar", figsize=(8, 5)
)

plt.title("Linear Regression vs Decision Tree Regressor")
plt.ylabel("Error")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
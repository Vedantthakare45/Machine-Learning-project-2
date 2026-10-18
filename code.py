# Step 1: Import the required libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn.datasets import load_iris

# Step 2: Load the Iris dataset
iris = load_iris()

# Step 3: Create a Pandas DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Step 4: Display the first five rows
print("First 5 rows of the Iris dataset:")
print(df.head())

# Step 5: Display descriptive statistics
print("\nBasic Statistics of the Iris dataset:")
print(df.describe())
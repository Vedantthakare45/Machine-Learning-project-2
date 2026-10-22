# Step 1: Import required libraries
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 2: Load the Iris dataset
iris = load_iris()

# Step 3: Separate features and target labels
X = iris.data
y = iris.target

# Display flower species
print("Flower species:", iris.target_names)

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 5: Create the Logistic Regression classifier
model = LogisticRegression(max_iter=200)

# Step 6: Train the model
model.fit(X_train, y_train)

# Step 7: Predict flower species
y_pred = model.predict(X_test)

# Step 8: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")

# Step 9: Display confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)
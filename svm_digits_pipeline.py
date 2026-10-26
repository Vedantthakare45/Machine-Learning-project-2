# 1. Import required libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 2. Load the Digits dataset
digits = load_digits()

X = digits.data
y = digits.target

print("Dataset Shape:", X.shape)
print("Number of Classes:", len(np.unique(y)))

# 3. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# 4. Create a Pipeline with StandardScaler and SVC
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC())
])

# 5. Define the hyperparameter grid
param_grid = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": [0.001, 0.01, 0.1, 1]
}

# 6. Perform GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

print("\nSearching for the best hyperparameters...")
grid_search.fit(X_train, y_train)

# 7. Display the best hyperparameters
print("\nBest Hyperparameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Accuracy:")
print(round(grid_search.best_score_, 4))

# 8. Evaluate the best model on test data
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", round(accuracy, 4))
print("Test Accuracy Percentage:", round(accuracy * 100, 2), "%")

# 9. Display the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 10. Create and display the confusion matrix
cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=digits.target_names
)

display.plot(cmap="Blues")
plt.title("SVM Digits Classification - Confusion Matrix")
plt.show()

# 11. Display an example prediction
sample = X_test[0].reshape(1, -1)
prediction = best_model.predict(sample)[0]

print("\nExample Prediction:")
print("Actual Digit:", y_test[0])
print("Predicted Digit:", prediction)

# 12. Display the handwritten digit image
plt.imshow(X_test[0].reshape(8, 8), cmap="gray")
plt.title(f"Actual: {y_test[0]}, Predicted: {prediction}")
plt.axis("off")
plt.show()
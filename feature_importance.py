
# ==========================================
# Titanic Feature Importance Dashboard
# Random Forest + Plotly
# ==========================================

import pandas as pd
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# TASK 1: Load Titanic Dataset
# ==========================================

# Load Titanic dataset
df = sns.load_dataset("titanic")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)


# Select input features
features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

# Select target column
X = df[features]
y = df["survived"]


# Separate numerical and categorical features
numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]


# Preprocessing numerical features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])


# Preprocessing categorical features
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


# ==========================================
# TASK 2: Train Random Forest Classifier
# ==========================================

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create Random Forest model
model = Pipeline([
    ("preprocessor", preprocessor),

    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])


# Train model
model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")


# Make predictions
y_pred = model.predict(X_test)


# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==========================================
# TASK 2: Extract Feature Importance
# ==========================================

# Get trained preprocessing object
trained_preprocessor = model.named_steps["preprocessor"]


# Get encoded categorical feature names
encoder = (
    trained_preprocessor
    .named_transformers_["cat"]
    .named_steps["encoder"]
)


encoded_features = encoder.get_feature_names_out(
    categorical_features
)


# Combine numerical and encoded feature names
all_feature_names = (
    numeric_features + list(encoded_features)
)


# Extract feature importance from Random Forest
importance_values = (
    model.named_steps["classifier"].feature_importances_
)


# Create DataFrame
importance_df = pd.DataFrame({
    "Feature": all_feature_names,
    "Importance": importance_values
})


# Sort features by importance
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


print("\nFeature Importance:")
print(importance_df.to_string(index=False))


# ==========================================
# TASK 3: Create Interactive Plotly Chart
# ==========================================

# Create horizontal bar chart
fig = px.bar(
    importance_df.sort_values(
        by="Importance",
        ascending=True
    ),

    x="Importance",

    y="Feature",

    orientation="h",

    title="Titanic Random Forest Feature Importance",

    labels={
        "Importance": "Feature Importance Score",
        "Feature": "Feature Name"
    },

    hover_data={
        "Importance": ":.4f"
    }
)


# Customize chart
fig.update_layout(
    template="plotly_white",

    height=600,

    xaxis_title="Importance Score",

    yaxis_title="Features"
)


# Display interactive chart
fig.show()


# Save chart as HTML
fig.write_html(
    "titanic_feature_importance.html"
)


print("\nInteractive dashboard created successfully!")

print("Chart saved as titanic_feature_importance.html")

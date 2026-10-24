import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# 1. Load the Adult dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week",
    "native-country", "income"
]

df = pd.read_csv(
    url,
    names=columns,
    skipinitialspace=True,
    na_values="?"
)

print("Original Dataset:")
print(df.head())

# 2. Separate features and target
X = df.drop("income", axis=1)
y = df["income"]

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(include="number").columns
categorical_features = X.select_dtypes(exclude="number").columns

# Numerical preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

# Create ColumnTransformer
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 3. Fit and transform the features
processed_data = preprocessor.fit_transform(X)

# Get the transformed column names
feature_names = preprocessor.get_feature_names_out()

# Convert the processed data into a DataFrame
processed_df = pd.DataFrame(
    processed_data,
    columns=feature_names,
    index=df.index
)

# Add the original income target back
processed_df["income"] = y

# 4. Export the processed DataFrame to CSV
processed_df.to_csv("adult_processed.csv", index=False)

print("\nProcessed Dataset:")
print(processed_df.head())

print("\nProcessed dataset shape:", processed_df.shape)
print("\nCSV file exported successfully!")
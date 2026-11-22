# ============================================================
# DIAMOND PRICE PREDICTION USING KNN REGRESSION
# DATA PREPROCESSING
# ============================================================

# 1. Import Libraries

import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline


# 2. Define Project Directory

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "diamonds.csv"


# 3. Load Dataset

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET LOADED ==========\n")

print("Dataset Shape:", df.shape)


# 4. Check Missing Values

print("\n========== MISSING VALUES ==========\n")

print(df.isnull().sum())


# 5. Check Duplicate Records

print("\n========== DUPLICATE RECORDS ==========\n")

print("Total Duplicates:", df.duplicated().sum())

# We retain duplicate rows because identical diamond
# characteristics and prices can represent valid records.


# 6. Separate Independent and Dependent Variables

X = df.drop("price", axis=1)

y = df["price"]

print("\n========== FEATURE SEPARATION ==========\n")

print("Independent Variables Shape:", X.shape)

print("Dependent Variable Shape:", y.shape)


# 7. Identify Numerical and Categorical Features

numerical_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()

print("\nNumerical Features:")

print(numerical_features)

print("\nCategorical Features:")

print(categorical_features)


# 8. Split Dataset into Training and Testing Sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print("\n========== TRAIN TEST SPLIT ==========\n")

print("Training Features:", X_train.shape)

print("Testing Features:", X_test.shape)

print("Training Target:", y_train.shape)

print("Testing Target:", y_test.shape)


# 9. Numerical Preprocessing Pipeline

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# 10. Categorical Preprocessing Pipeline

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# 11. Combine Numerical and Categorical Pipelines

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)


# 12. Fit and Transform Training Data

X_train_processed = preprocessor.fit_transform(X_train)


# 13. Transform Testing Data

X_test_processed = preprocessor.transform(X_test)


# 14. Display Processed Data Information

print("\n========== PREPROCESSING RESULTS ==========\n")

print("Original Training Shape:", X_train.shape)

print("Original Testing Shape:", X_test.shape)

print("Processed Training Shape:", X_train_processed.shape)

print("Processed Testing Shape:", X_test_processed.shape)


# 15. Verify Processed Data

print("\n========== PREPROCESSING VERIFICATION ==========\n")

print("Training Data Contains Missing Values:",
      np.isnan(X_train_processed).any())

print("Testing Data Contains Missing Values:",
      np.isnan(X_test_processed).any())

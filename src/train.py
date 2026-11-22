# ============================================================
# DIAMOND PRICE PREDICTION USING KNN REGRESSION
# MODEL TRAINING AND EVALUATION
# ============================================================


# 1. Import Required Libraries

import numpy as np
import pandas as pd
import pickle

from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# 2. Define Project Directory

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "diamonds.csv"

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# 3. Load Dataset

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET LOADED ==========\n")

print("Dataset Shape:", df.shape)


# 4. Separate Independent and Dependent Variables

X = df.drop("price", axis=1)

y = df["price"]


# 5. Identify Numerical and Categorical Features

numerical_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()


print("\nNumerical Features:", numerical_features)

print("\nCategorical Features:", categorical_features)


# 6. Split Dataset into Training and Testing Sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)


print("\n========== TRAIN TEST SPLIT ==========\n")

print("X_train:", X_train.shape)

print("X_test:", X_test.shape)

print("y_train:", y_train.shape)

print("y_test:", y_test.shape)


# 7. Numerical Preprocessing Pipeline

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# 8. Categorical Preprocessing Pipeline

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# 9. Combine Both Pipelines

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)


# 10. Create Complete ML Pipeline

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", KNeighborsRegressor())
    ]
)


# 11. Hyperparameter Tuning

print("\n========== HYPERPARAMETER TUNING ==========\n")

# Use a representative subset of training data
# to make hyperparameter tuning faster.

X_tune, _, y_tune, _ = train_test_split(
    X_train,
    y_train,
    test_size=0.80,
    random_state=42
)


# Different K values to experiment with

param_grid = {
    "model__n_neighbors": [3, 5, 7, 9, 11]
}


# Grid Search with Cross Validation

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1
)


grid_search.fit(X_tune, y_tune)


# 12. Find Best K Value

best_k = grid_search.best_params_["model__n_neighbors"]

print("\nBest K Value:", best_k)

print("Best Cross Validation MAE:", -grid_search.best_score_)


# 13. Train Final Model Using Complete Training Data

print("\n========== FINAL MODEL TRAINING ==========\n")

best_model = grid_search.best_estimator_

best_model.fit(X_train, y_train)

print("KNN Model Trained Successfully!")


# 14. Make Predictions

y_pred = best_model.predict(X_test)


# 15. Model Evaluation

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print("\n========== MODEL EVALUATION ==========\n")

print("Mean Absolute Error (MAE):", round(mae, 2))

print("Mean Squared Error (MSE):", round(mse, 2))

print("Root Mean Squared Error (RMSE):", round(rmse, 2))

print("R2 Score:", round(r2, 4))


# 16. Save Complete ML Pipeline Using Pickle

MODEL_PATH = MODEL_DIR / "knn_diamond_pipeline.pkl"

with open(MODEL_PATH, "wb") as file:

    pickle.dump(best_model, file)


print("\n========== MODEL SAVING ==========\n")

print("Model Saved Successfully!")

print("Model Location:", MODEL_PATH)


# 17. Display Actual vs Predicted Prices

results = pd.DataFrame({

    "Actual Price": y_test.values[:10],

    "Predicted Price": y_pred[:10]

})


print("\n========== ACTUAL VS PREDICTED ==========\n")

print(results)

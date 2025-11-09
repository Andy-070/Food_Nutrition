"""
STEP 3 — Ridge Regression (Supervised Learning)
Builds a model to predict a Nutrition Score using cleaned + clustered data
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

DATA_PATH = os.path.join("data", "ifct_clustered.csv")
MODEL_PATH = os.path.join("models", "ridge.pkl")

def compute_score(row):
    """Rule-based nutrition score generator"""
    score = 100
    # Balanced nutrition rules
    if row["Protein (g)"] < 5:  score -= 10
    if row["Fiber (g)"] < 2:    score -= 10
    if row["Fat (g)"] > 20:     score -= 10
    if row["Sugars (g)"] > 15:  score -= 10
    if row["Sodium (mg)"] > 400:score -= 5
    if row["Vitamin C (mg)"] < 10: score -= 5
    if row["Iron (mg)"] < 1:    score -= 5
    if row["Calcium (mg)"] < 50:score -= 5
    return max(min(score, 100), 0)

def train_ridge():
    # Load clustered data
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded clustered dataset: {df.shape[0]} rows")

    # Compute rule-based nutrition score
    df["Nutrition_Score"] = df.apply(compute_score, axis=1)

    # Features: scaled nutrients + cluster
    features = ['Energy (kcal)','Protein (g)','Fat (g)','CHO (g)','Fiber (g)',
                'Sugars (g)','Sodium (mg)','Calcium (mg)','Iron (mg)','Vitamin C (mg)','Cluster']
    X = df[features]
    y = df["Nutrition_Score"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Ridge regression
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nRidge Regression Performance:")
    print(f"MAE: {mae:.3f}")
    print(f"R²:  {r2:.3f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(ridge, MODEL_PATH)
    print(f"Saved Ridge model → {MODEL_PATH}")

    # Show a few predictions
    preview = pd.DataFrame({"Actual": y_test[:10].values, "Predicted": y_pred[:10]})
    print("\n Sample Predictions:")
    print(preview)

if __name__ == "__main__":
    train_ridge()

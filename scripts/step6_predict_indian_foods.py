"""
STEP 6 — Predict Nutrition Score & Feedback for Cooked Indian Foods
Uses existing Ridge + KMeans models trained on IFCT data.
"""

import os
import pandas as pd
import numpy as np
import joblib

# === File paths ===
DATA_PATH = os.path.join("data", "indian_foods_cleaned.csv")
KMEANS_PATH = os.path.join("models", "kmeans.pkl")
RIDGE_PATH = os.path.join("models", "ridge.pkl")
OUTPUT_PATH = os.path.join("data", "indian_foods_scored.csv")

# === Load assets ===
print("🔄 Loading models and data ...")
df = pd.read_csv(DATA_PATH)
kmeans = joblib.load(KMEANS_PATH)
ridge = joblib.load(RIDGE_PATH)

# === Feature columns (same as before) ===
features = ['Energy (kcal)', 'Protein (g)', 'Fat (g)', 'CHO (g)', 'Fiber (g)',
            'Sugars (g)', 'Sodium (mg)', 'Calcium (mg)', 'Iron (mg)', 'Vitamin C (mg)']

# --- Generate clusters ---
print("🧠 Assigning clusters using KMeans ...")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
df["Cluster"] = kmeans.predict(X_scaled)

# --- Predict nutrition scores using Ridge ---
print("📈 Predicting Nutrition Scores ...")
X_with_cluster = np.hstack([X_scaled, df["Cluster"].values.reshape(-1,1)])
df["Predicted_Score"] = ridge.predict(X_with_cluster).clip(0,100)

# --- Generate feedback text ---
def generate_feedback(row):
    tips = []
    if row["Protein (g)"] < 5:
        tips.append("Add more protein sources like dal, paneer or lentils.")
    if row["Fiber (g)"] < 2:
        tips.append("Include fiber-rich veggies or whole grains.")
    if row["Fat (g)"] > 20:
        tips.append("Limit oily or fried food.")
    if row["Sugars (g)"] > 10:
        tips.append("Cut down on sugary items or desserts.")
    if row["Sodium (mg)"] > 400:
        tips.append("Reduce salt and processed foods.")
    if row["Vitamin C (mg)"] < 10:
        tips.append("Add citrus fruits, amla or bell peppers.")
    if not tips:
        tips.append("Well-balanced dish 👍.")
    return " | ".join(tips)

df["Feedback"] = df.apply(generate_feedback, axis=1)

# --- Save scored dataset ---
os.makedirs("data", exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Saved results to: {OUTPUT_PATH}")

# --- Show sample output ---
print("\n🔹 Sample predictions:")
print(df[["Food Item","Predicted_Score","Cluster","Feedback"]].head(10))

"""
STEP 4 — Feedback Generator & Interactive Demo
Lets a user query any Indian food from the IFCT dataset
and receive:
    - Cluster group
    - Predicted Nutrition Score
    - Personalized dietary feedback
"""

import os
import pandas as pd
import numpy as np
import joblib

# File paths
DATA_PATH = os.path.join("data", "ifct_clustered.csv")
RIDGE_PATH = os.path.join("models", "ridge.pkl")
KMEANS_PATH = os.path.join("models", "kmeans.pkl")

# Load assets
df = pd.read_csv(DATA_PATH)
ridge = joblib.load(RIDGE_PATH)
kmeans = joblib.load(KMEANS_PATH)

# Nutrient columns used for prediction
features = ['Energy (kcal)', 'Protein (g)', 'Fat (g)', 'CHO (g)', 'Fiber (g)',
            'Sugars (g)', 'Sodium (mg)', 'Calcium (mg)', 'Iron (mg)', 'Vitamin C (mg)']

# --- Feedback generator function ---
def generate_feedback(row):
    tips = []
    if row["Protein (g)"] < 5:
        tips.append("Try increasing protein intake (dal, paneer, lentils).")
    if row["Fiber (g)"] < 2:
        tips.append("Add fiber-rich foods like vegetables, oats, or millets.")
    if row["Fat (g)"] > 20:
        tips.append("Reduce fried or oily foods to lower fat content.")
    if row["Sugars (g)"] > 10:
        tips.append("Cut down on sugar-heavy items and sweets.")
    if row["Sodium (mg)"] > 400:
        tips.append("Too much salt — avoid pickles or packaged snacks.")
    if row["Vitamin C (mg)"] < 10:
        tips.append("Add citrus fruits, guava, or amla for Vitamin C.")
    if not tips:
        tips.append("This food looks nutritionally balanced 👍.")
    return tips


# --- Function to predict nutrition score for a food ---
def predict_food(food_name):
    # Case-insensitive match
    matches = df[df['Food Item'].str.contains(food_name, case=False, na=False)]
    if matches.empty:
        print(f"❌ No match found for '{food_name}' in IFCT dataset.")
        return

    food = matches.iloc[0]  # take first match
    X = food[features].values.reshape(1, -1)
    cluster = int(food["Cluster"])
    X_input = np.append(X, cluster).reshape(1, -1)
    predicted_score = ridge.predict(X_input)[0]

    print(f"\n🍴 Food: {food['Food Item']}")
    print(f"📊 Cluster Group: {cluster}")
    print(f"⭐ Predicted Nutrition Score: {predicted_score:.1f}/100")

    print("\n💡 Feedback:")
    for tip in generate_feedback(food):
        print(" -", tip)

# --- Demo Interface ---
if __name__ == "__main__":
    print("=== Indian Nutrition Feedback System ===")
    while True:
        query = input("\nEnter a food name (or 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("Goodbye! 👋")
            break
        predict_food(query)

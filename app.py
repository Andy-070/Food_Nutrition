"""
Indian Nutrition AI — Streamlit Web App
Hybrid ML model combining K-Means (unsupervised) + Ridge (supervised)
Built from IFCT + Indian Food Nutrition datasets
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# ------------------------------
# Paths
# ------------------------------
DATA_PATH = "data/indian_foods_cleaned.csv"
KMEANS_PATH = "models/kmeans.pkl"
RIDGE_PATH = "models/ridge.pkl"

# ------------------------------
# Load assets
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_resource
def load_models():
    kmeans = joblib.load(KMEANS_PATH)
    ridge = joblib.load(RIDGE_PATH)
    return kmeans, ridge

df = load_data()
kmeans, ridge = load_models()

features = ['Energy (kcal)', 'Protein (g)', 'Fat (g)', 'CHO (g)', 'Fiber (g)',
            'Sugars (g)', 'Sodium (mg)', 'Calcium (mg)', 'Iron (mg)', 'Vitamin C (mg)']

# ------------------------------
# Utility: Generate feedback
# ------------------------------
def generate_feedback(row):
    tips = []
    if row["Protein (g)"] < 5:
        tips.append("Add more protein (dal, paneer, lentils).")
    if row["Fiber (g)"] < 2:
        tips.append("Include fiber-rich veggies or whole grains.")
    if row["Fat (g)"] > 20:
        tips.append("Limit oily or fried foods.")
    if row["Sugars (g)"] > 10:
        tips.append("Cut down on sweets or sugary drinks.")
    if row["Sodium (mg)"] > 400:
        tips.append("Reduce salt or processed snacks.")
    if row["Vitamin C (mg)"] < 10:
        tips.append("Add citrus fruits, amla, or bell peppers.")
    if not tips:
        tips.append("Looks well-balanced 👍.")
    return tips

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Indian Nutrition AI", page_icon="🥗", layout="centered")

st.title("🇮🇳 Indian Nutrition Analyzer")
st.caption("Hybrid ML System (K-Means + Ridge Regression)")

# user_input = st.text_input(" Enter a dish name (e.g., Palak Paneer, Aam Panna, Roti):")

# if user_input:
#     matches = df[df["Food Item"].str.contains(user_input, case=False, na=False)]
#     if matches.empty:
#         st.error(f"No match found for '{user_input}' in dataset.")
#     else:
#         food = matches.iloc[0]
#         st.subheader(food["Food Item"])

#         # Compute cluster + score
#         X = food[features].values.reshape(1, -1)
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(df[features])
#         cluster = kmeans.predict(scaler.transform(X))[0]
#         X_input = np.append(X, cluster).reshape(1, -1)
#         score = ridge.predict(X_input)[0]

#         st.metric(label="⭐ Nutrition Score", value=f"{score:.1f}/100")
#         st.write(f"**Cluster Group:** {cluster}")

#         # Show nutrients table
#         st.write("### 🍽️ Nutritional Breakdown (per 100g or serving)")
#         st.dataframe(food[features].to_frame().T.style.format("{:.2f}"))

#         # Feedback tips
#         st.write("### 💡 Feedback")
#         for tip in generate_feedback(food):
#             st.markdown(f"- {tip}")

# from rapidfuzz import process

# def fuzzy_match(query, choices, limit=3, threshold=70):
#     """Find best matching food items using fuzzy similarity"""
#     results = process.extract(query, choices, limit=limit, score_cutoff=threshold)
#     return results

# user_input = st.text_input("🔍 Enter a dish name (e.g., Palak Paneer, Aam Panna, Roti):")

# if user_input:
#     choices = df["Food Item"].tolist()
#     matches = fuzzy_match(user_input, choices)

#     if not matches:
#         st.error(f"No close match found for '{user_input}'. Try a simpler name.")
#     else:
#         best_match = matches[0][0]
#         match_score = matches[0][1]
#         st.info(f"🔎 Best match: **{best_match}** (Similarity: {match_score:.1f}%)")

#         food = df[df["Food Item"] == best_match].iloc[0]
#         st.subheader(food["Food Item"])

#         # Compute cluster + score
#         X = food[features].values.reshape(1, -1)
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(df[features])
#         cluster = kmeans.predict(scaler.transform(X))[0]
#         X_input = np.append(X, cluster).reshape(1, -1)
#         score = ridge.predict(X_input)[0]

#         st.metric(label="⭐ Nutrition Score", value=f"{score:.1f}/100")
#         st.write(f"**Cluster Group:** {cluster}")

#         # Show nutrients
#         st.write("### 🍽️ Nutritional Breakdown (per 100g or serving)")
#         st.dataframe(food[features].to_frame().T.style.format("{:.2f}"))

#         # Feedback tips
#         st.write("### 💡 Feedback")
#         for tip in generate_feedback(food):
#             st.markdown(f"- {tip}")

from rapidfuzz import process
import re

def fuzzy_match(query, choices, limit=1, threshold=65):
    """Return the best match above a similarity threshold"""
    results = process.extract(query, choices, limit=limit, score_cutoff=threshold)
    return results[0][0] if results else None

def parse_meal_input(meal_text):
    """
    Parse user input like '2 rotis, dal, 1 glass aam panna'
    Returns a list of (food_name, quantity)
    """
    meal_parts = re.split(r',|and', meal_text.lower())
    parsed = []
    for part in meal_parts:
        part = part.strip()
        match = re.match(r'(\d+)?\s*(.*)', part)
        qty = int(match.group(1)) if match.group(1) else 1
        food_name = match.group(2).strip()
        parsed.append((food_name, qty))
    return parsed


user_input = st.text_input("🍽️ Enter your full meal (e.g., '2 rotis, dal, and aam panna'):").strip()

if user_input:
    parsed_meal = parse_meal_input(user_input)
    meal_summary = []
    total_nutrients = {col: 0 for col in features}

    st.subheader("🍛 Meal Breakdown")

    for item, qty in parsed_meal:
        best_match = fuzzy_match(item, df["Food Item"].tolist())
        if not best_match:
            st.warning(f"❌ Could not find a match for '{item}'")
            continue

        food = df[df["Food Item"] == best_match].iloc[0]
        st.write(f"**{qty} × {best_match}**")

        # Scale nutrients by quantity
        scaled = food[features] * qty
        for col in features:
            total_nutrients[col] += scaled[col]

        # Compute per-item score
        X = food[features].values.reshape(1, -1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features])
        cluster = kmeans.predict(scaler.transform(X))[0]
        X_input = np.append(X, cluster).reshape(1, -1)
        score = ridge.predict(X_input)[0]

        st.metric(label=f"{best_match}", value=f"{score:.1f}/100")
        st.caption(f"Cluster Group: {cluster}")

        # Store summary
        meal_summary.append({"Food": best_match, "Qty": qty, "Score": score})

    # --- Show total nutrients ---
    st.write("### ⚖️ Total Meal Nutrition (approx.)")
    st.dataframe(pd.DataFrame([total_nutrients]).style.format("{:.2f}"))

    # --- Compute overall meal score ---
    if meal_summary:
        avg_score = np.mean([m["Score"] for m in meal_summary])
        st.subheader(f"⭐ Overall Meal Score: {avg_score:.1f}/100")

        # --- Generate combined feedback ---
        st.write("### 💡 Meal Feedback")
        meal_feedback = []
        for food_name, qty in parsed_meal:
            best_match = fuzzy_match(food_name, df["Food Item"].tolist())
            if best_match:
                food = df[df["Food Item"] == best_match].iloc[0]
                meal_feedback += generate_feedback(food)

        for tip in sorted(set(meal_feedback)):
            st.markdown(f"- {tip}")


# ------------------------------
# Footer
# ------------------------------
st.divider()
st.caption("Developed with ❤️ using IFCT + Kaggle Indian Food Dataset + Scikit-Learn")

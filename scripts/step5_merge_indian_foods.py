"""
STEP 5 — Clean and Align Indian Cooked Food Dataset
Dataset source: batthulavinay/indian-food-nutrition (Kaggle)
"""

import os
import pandas as pd

RAW_PATH = os.path.join("data", "indian_food_nutrition.csv")
CLEAN_PATH = os.path.join("data", "indian_foods_cleaned.csv")

# 1️⃣ Load dataset
df = pd.read_csv(RAW_PATH)
print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# 2️⃣ Rename columns to match IFCT format
rename_map = {
    'Dish Name': 'Food Item',
    'Calories (kcal)': 'Energy (kcal)',
    'Protein (g)': 'Protein (g)',
    'Fats (g)': 'Fat (g)',
    'Carbohydrates (g)': 'CHO (g)',
    'Fibre (g)': 'Fiber (g)',
    'Free Sugar (g)': 'Sugars (g)',
    'Sodium (mg)': 'Sodium (mg)',
    'Calcium (mg)': 'Calcium (mg)',
    'Iron (mg)': 'Iron (mg)',
    'Vitamin C (mg)': 'Vitamin C (mg)'
}

df = df.rename(columns=rename_map)

# 3️⃣ Ensure all required columns exist
required_cols = ['Food Item','Energy (kcal)','Protein (g)','Fat (g)',
                 'CHO (g)','Fiber (g)','Sugars (g)','Sodium (mg)',
                 'Calcium (mg)','Iron (mg)','Vitamin C (mg)']

for col in required_cols:
    if col not in df.columns:
        df[col] = 0.0  # fill missing columns

# 4️⃣ Clean numeric columns
for c in required_cols:
    if c != 'Food Item':
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

# 5️⃣ Clean text column
df['Food Item'] = df['Food Item'].astype(str).str.strip()

# 6️⃣ Save cleaned dataset
os.makedirs("data", exist_ok=True)
df.to_csv(CLEAN_PATH, index=False)
print(f"✅ Cleaned and saved as: {CLEAN_PATH}")

# Show small preview
print("\n🔹 Sample:")
print(df.head(10))

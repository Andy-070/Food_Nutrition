"""
STEP 1 — Clean IFCT 2017 Dataset
Extract main nutrients & standardize column names
"""

import pandas as pd
import os

RAW_PATH = os.path.join("data", "ifct2017.csv")
CLEAN_PATH = os.path.join("data", "ifct_cleaned.csv")

def clean_ifct():
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Drop all “_e” columns (estimation/error values)
    df = df[[c for c in df.columns if not c.endswith('_e')]]

    # Select only the key nutrient columns
    keep_cols = {
        "name": "Food Item",
        "enerc": "Energy (kcal)",
        "protcnt": "Protein (g)",
        "fatce": "Fat (g)",
        "choavldf": "CHO (g)",
        "fibtg": "Fiber (g)",
        "fsugar": "Sugars (g)",
        "na": "Sodium (mg)",
        "ca": "Calcium (mg)",
        "fe": "Iron (mg)",
        "vitc": "Vitamin C (mg)",
    }

    missing = [col for col in keep_cols.keys() if col not in df.columns]
    if missing:
        print("⚠️ Missing columns:", missing)

    # Extract and rename
    df_clean = df[list(keep_cols.keys())].rename(columns=keep_cols)

    # Fill missing numeric values
    for col in df_clean.columns:
        if col != "Food Item":
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0.0)

    # Save cleaned dataset
    os.makedirs("data", exist_ok=True)
    df_clean.to_csv(CLEAN_PATH, index=False)
    print(f"✅ Cleaned dataset saved to: {CLEAN_PATH}")
    print(df_clean.head(10))

if __name__ == "__main__":
    clean_ifct()

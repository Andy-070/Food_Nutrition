"""
STEP 2 — Feature Scaling + K-Means Clustering (Unsupervised Learning)
Purpose:
    - Load cleaned IFCT dataset
    - Standardize features
    - Find optimal number of clusters (Elbow Method)
    - Run K-Means and save cluster labels
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = os.path.join("data", "ifct_cleaned.csv")
CLUSTERED_PATH = os.path.join("data", "ifct_clustered.csv")
MODEL_PATH = os.path.join("models", "kmeans.pkl")

def run_kmeans():
    # Load cleaned dataset
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded cleaned dataset: {df.shape[0]} rows")

    # Select numeric features
    features = ['Energy (kcal)', 'Protein (g)', 'Fat (g)', 'CHO (g)', 'Fiber (g)',
                'Sugars (g)', 'Sodium (mg)', 'Calcium (mg)', 'Iron (mg)', 'Vitamin C (mg)']
    X = df[features]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determine optimal k using Elbow Method
    inertia = []
    k_values = range(2, 7)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(6,4))
    plt.plot(k_values, inertia, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.tight_layout()
    plt.savefig("data/elbow_plot.png")
    print("📈 Elbow plot saved as data/elbow_plot.png")

    # Choose cluster count (4 works well for food types)
    k_opt = 4
    kmeans_final = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
    clusters = kmeans_final.fit_predict(X_scaled)

    df["Cluster"] = clusters

    # Analyze cluster means
    cluster_summary = df.groupby("Cluster")[features].mean().round(2)
    print("\n🔹 Cluster Summary (avg nutrient values):")
    print(cluster_summary)

    # Save results
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    import joblib
    joblib.dump(kmeans_final, MODEL_PATH)
    df.to_csv(CLUSTERED_PATH, index=False)
    print(f"\n✅ Saved clustered dataset → {CLUSTERED_PATH}")
    print(f"✅ Saved K-Means model → {MODEL_PATH}")

    # Visualize clusters
    plt.figure(figsize=(7,5))
    sns.scatterplot(data=df, x="Protein (g)", y="CHO (g)", hue="Cluster", palette="Set2")
    plt.title("Food Clusters (Protein vs Carbs)")
    plt.tight_layout()
    plt.savefig("data/cluster_plot.png")
    print("📊 Cluster visualization saved as data/cluster_plot.png")

if __name__ == "__main__":
    run_kmeans()

# 🇮🇳 Indian Nutrition Analyzer  
### *Hybrid Machine Learning System using K-Means & Ridge Regression*

---

## 🧠 Project Overview
This project builds an AI-based *Indian Nutrition Analyzer* that predicts a *Nutrition Score (0–100)* for any Indian dish and provides *personalized dietary feedback*.  
The system combines *unsupervised (K-Means clustering)* and *supervised (Ridge Regression)* learning techniques to analyze both *raw ingredients* (from IFCT 2017) and *cooked foods* (from the Kaggle Indian Food Nutrition dataset).  
A *Streamlit app* serves as the final interface for real-time nutrition insights.

---

## 👨‍💻 Team Members and Work Split

### 🧩 *Group 1 – Data Engineering & K-Means Model*
*Members:*  
- *Rohit More (2023bit056)*  
- *Prasad Jadhav (2023bit052)*  

*Responsibilities:*  
- Collected and cleaned *IFCT 2017* and *Indian Food Nutrition* datasets.  
- Standardized nutrient column names (Energy, Protein, Fat, Carbs, Fiber, etc.).  
- Handled missing values, removed duplicates, and normalized numerical values.  
- Performed *exploratory data analysis (EDA)* to understand nutrient distributions.  
- Applied *feature scaling* using StandardScaler.  
- Implemented *K-Means Clustering* to group foods into categories such as:
  - High Protein
  - High Carbohydrate
  - High Fat
  - Balanced Foods  
- Determined optimal clusters using the *Elbow Method* and visualized results.  
- Saved models (kmeans.pkl, scaler.pkl) for integration with the main pipeline.

*Scripts Developed:*  
- step1_clean_ifct.py – Cleaning and standardization of IFCT data.  
- step2_clustering.py – Feature scaling, K-Means model training, visualization.  
- step5_merge_indian_foods.py – Integration of cooked Indian food dataset.

---

### 🧩 *Group 2 – Ridge Regression & Streamlit Application*
*Members:*  
- *Anurag Bhavthankar (2024bitXXX)*  
- *Priti Sanghai (2023bit153)*  

*Responsibilities:*  
- Developed a *rule-based scoring function* to assign nutrition scores (0–100).  
- Trained *Ridge Regression* on clustered IFCT data to predict Nutrition Scores.  
- Evaluated model performance using *MAE* and *R²* metrics.  
- Built a *Streamlit web application* to allow users to:
  - Input food/dish names or full meals.  
  - View predicted Nutrition Score and cluster classification.  
  - Get personalized feedback on improving diet balance.  
- Integrated *fuzzy matching* (using RapidFuzz) for flexible user search.  
- Implemented *multi-dish analyzer* (e.g., “2 rotis, dal, and aam panna”) with total meal nutrition summary.  
- Deployed final hybrid model combining Ridge + K-Means for real-time predictions.

*Scripts Developed:*  
- step3_supervised_model.py – Ridge Regression training and evaluation.  
- step6_predict_indian_foods.py – Scoring cooked foods and feedback logic.  
- app.py – Streamlit web application (frontend + backend integration).

---

## 🧩 Collaborative Work (All Members)
- Joint discussion on project architecture and ML pipeline design.  
- Testing and debugging of each stage (data → model → Streamlit).  
- Preparing project documentation and presentation slides.  
- Conducting performance analysis and feature evaluation.  

---

## 📂 Repository Structure

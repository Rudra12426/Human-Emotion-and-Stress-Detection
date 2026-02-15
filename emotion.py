print("NOTE: If you face any error while running this script, please run it using Jupyter Notebook environment.")
# =========================================================
# ADVANCED HUMAN STRESS DETECTION USING MACHINE LEARNING
# =========================================================

import pandas as pd
import numpy as np
# =========================================================
# ADVANCED HUMAN STRESS DETECTION USING MACHINE LEARNING
# WITH CSV LOGGING FOR DETECTED STRESS
# =========================================================

# STEP 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import csv
from datetime import datetime
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# =========================================================
# STEP 2: Load Dataset
# =========================================================
print("Loading dataset...")

df = pd.read_csv(
    r"C:\Users\Victus\OneDrive\Desktop\ML Project\Human Emotion and stress Detection\Stress Indicators Dataset for Mental Health Classification.csv"
)

print("\nDataset loaded successfully!")
print("Shape:", df.shape)

# =========================================================
# STEP 3: Data Inspection
# =========================================================
print("\nChecking missing values...")
print(df.isnull().sum())

# =========================================================
# STEP 4: Data Cleaning
# =========================================================
df.fillna(df.mean(numeric_only=True), inplace=True)

# =========================================================
# STEP 5: Feature & Target Separation
# =========================================================
X = df.drop("stress_type", axis=1)
y = df["stress_type"]

# =========================================================
# STEP 6: Feature Scaling
# =========================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================================================
# STEP 7: Train-Test Split
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================================
# STEP 8: Hyperparameter Tuning
# =========================================================
print("\nPerforming hyperparameter tuning...")

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("Best Parameters Found:")
print(grid_search.best_params_)

# =========================================================
# STEP 9: Model Evaluation
# =========================================================
print("\nEvaluating model...")

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Final Model Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================================================
# STEP 10: Confusion Matrix
# =========================================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Stress Detection")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================================================
# STEP 11: Feature Importance (Explainable AI)
# =========================================================
importances = best_model.feature_importances_
feature_names = X.columns

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
plt.title("Feature Importance (Explainable AI)")
plt.show()

# =========================================================
# STEP 12: Save Model & Scaler
# =========================================================
joblib.dump(best_model, "stress_detection_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved successfully!")

# =========================================================
# STEP 13: Real User Prediction (DEMO) + CSV LOGGING
# =========================================================
print("\nPredicting stress for a new user...")

# Example input: using mean values of features as dummy input
new_user_data = np.array([X.mean().values])
new_user_scaled = scaler.transform(new_user_data)

prediction = best_model.predict(new_user_scaled)[0]
probabilities = best_model.predict_proba(new_user_scaled)[0]

# Map class labels
stress_labels = {0: "No Stress", 1: "Moderate Stress", 2: "High Stress"}
predicted_label = stress_labels[prediction]
confidence = max(probabilities)

# ------------------- CSV Logging Section -------------------
CSV_FILE = "detected_stress_logs.csv"

# Save ONLY if stress detected (Moderate or High)
if prediction in [1, 2]:
    file_exists = os.path.isfile(CSV_FILE)

    with open(CSV_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write header only once
        if not file_exists:
            writer.writerow([
                "Timestamp",
                "Predicted_Stress_Level",
                "Confidence"
            ])

        # Write new stress record
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            predicted_label,
            round(confidence, 2)
        ])

    print(f"⚠️ Stress detected! Prediction saved to {CSV_FILE}")
else:
    print("✅ No stress detected. Nothing saved.")

# =========================================================
# END OF ADVANCED PROJECT
# =========================================================

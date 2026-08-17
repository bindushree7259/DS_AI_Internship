import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("data.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nColumn Names:")
print(data.columns)


# ==========================================
# 2. REMOVE UNNECESSARY COLUMNS
# ==========================================

# Remove ID column
# Remove empty column if it exists

data = data.drop(
    columns=["id", "Unnamed: 32"],
    errors="ignore"
)


# ==========================================
# 3. CONVERT DIAGNOSIS TO NUMBERS
# ==========================================

# M = Malignant = 1
# B = Benign = 0

data["diagnosis"] = data["diagnosis"].map({
    "M": 1,
    "B": 0
})


# ==========================================
# 4. SEPARATE FEATURES AND TARGET
# ==========================================

# X = Input features
X = data.drop("diagnosis", axis=1)

# y = Target
y = data["diagnosis"]


print("\nInput Features:")
print(X.head())

print("\nTarget:")
print(y.head())


# ==========================================
# 5. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==========================================
# 6. LOGISTIC REGRESSION
# ==========================================

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])


# Train model
logistic_model.fit(X_train, y_train)


# Make predictions
y_pred = logistic_model.predict(X_test)


# ==========================================
# 7. LOGISTIC REGRESSION RESULTS
# ==========================================

print("\n================================")
print("LOGISTIC REGRESSION")
print("================================")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Benign", "Malignant"]
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==========================================
# 8. RANDOM FOREST
# ==========================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train Random Forest
rf_model.fit(X_train, y_train)


# Make predictions
rf_pred = rf_model.predict(X_test)


# ==========================================
# 9. RANDOM FOREST RESULTS
# ==========================================

print("\n================================")
print("RANDOM FOREST")
print("================================")

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print("Accuracy:", rf_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1 Score:", rf_f1)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        rf_pred,
        target_names=["Benign", "Malignant"]
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))


# ==========================================
# 10. MODEL COMPARISON
# ==========================================

print("\n================================")
print("MODEL COMPARISON")
print("================================")

print("Logistic Regression Accuracy:", accuracy)
print("Random Forest Accuracy:", rf_accuracy)


if accuracy > rf_accuracy:
    print("\nBest Model: Logistic Regression")

elif rf_accuracy > accuracy:
    print("\nBest Model: Random Forest")

else:
    print("\nBoth models have the same accuracy.")


# ==========================================
# 11. PREDICT ONE TEST SAMPLE
# ==========================================

sample = X_test.iloc[[0]]

prediction = logistic_model.predict(sample)

if prediction[0] == 1:
    print("\nSample Prediction: Malignant")
else:
    print("\nSample Prediction: Benign")
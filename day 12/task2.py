# ============================================================
# LOGISTIC REGRESSION - LOAN DEFAULT PREDICTION
# ============================================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ============================================================
# 1. CREATE DATASET
# ============================================================

data = {
    "Income": [
        25000, 30000, 35000, 40000, 45000,
        50000, 55000, 60000, 65000, 70000,
        28000, 32000, 38000, 48000, 52000
    ],

    "Credit_Score": [
        550, 580, 600, 620, 650,
        680, 700, 720, 750, 780,
        570, 590, 630, 670, 710
    ],

    "Loan_Amount": [
        300000, 280000, 250000, 220000, 200000,
        180000, 150000, 140000, 120000, 100000,
        320000, 270000, 240000, 190000, 160000
    ],

    "Employment_Status": [
        0, 0, 0, 1, 1,
        1, 1, 1, 1, 1,
        0, 0, 1, 1, 1
    ],

    "Previous_Payment_History": [
        0, 0, 0, 1, 1,
        1, 1, 1, 1, 1,
        0, 0, 1, 1, 1
    ],

    "Default": [
        1, 1, 1, 0, 0,
        0, 0, 0, 0, 0,
        1, 1, 0, 0, 0
    ]
}

df = pd.DataFrame(data)


# ============================================================
# 2. DISPLAY DATASET
# ============================================================

print("=" * 60)
print("LOAN DEFAULT DATASET")
print("=" * 60)

print(df)


# ============================================================
# 3. INPUT AND TARGET
# ============================================================

X = df[
    [
        "Income",
        "Credit_Score",
        "Loan_Amount",
        "Employment_Status",
        "Previous_Payment_History"
    ]
]

y = df["Default"]


# ============================================================
# 4. SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# ============================================================
# 5. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# 6. CREATE LOGISTIC REGRESSION MODEL
# ============================================================

model = LogisticRegression()

model.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# 7. PREDICTION
# ============================================================

y_pred = model.predict(
    X_test_scaled
)


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# ============================================================
# 9. CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\n" + "=" * 60)
print("CLASSIFICATION METRICS")
print("=" * 60)

print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1-Score :", round(f1 * 100, 2), "%")


# ============================================================
# 10. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Not Default",
            "Default"
        ],
        zero_division=0
    )
)


# ============================================================
# 11. PREDICT A NEW CUSTOMER
# ============================================================

new_customer = pd.DataFrame({
    "Income": [35000],
    "Credit_Score": [590],
    "Loan_Amount": [280000],
    "Employment_Status": [0],
    "Previous_Payment_History": [0]
})

new_customer_scaled = scaler.transform(
    new_customer
)

prediction = model.predict(
    new_customer_scaled
)

print("\n" + "=" * 60)
print("NEW CUSTOMER PREDICTION")
print("=" * 60)

if prediction[0] == 1:
    print("Prediction: Customer may DEFAULT on the loan.")
else:
    print("Prediction: Customer may NOT DEFAULT on the loan.")
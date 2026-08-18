# ============================================================
# ML TASK:
# Real-world problem using Supervised and Unsupervised Learning
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# ============================================================
# 1. REAL-WORLD PROBLEM
# ============================================================

print("=" * 60)
print("REAL-WORLD ML PROBLEM: CROP SUITABILITY")
print("=" * 60)

print("""
Scenario:
Farmers need to decide which crop is suitable for their land.

The decision can depend on:
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall
""")


# ============================================================
# 2. LOAD DATASET
# ============================================================

base_path = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(
    base_path,
    "Crop_recommendation.csv"
)

data = pd.read_csv(csv_path)

print("\nDataset loaded successfully!")
print("\nFirst 5 rows:")
print(data.head())


# ============================================================
# 3. DATA INFORMATION
# ============================================================

print("\nDataset shape:")
print(data.shape)

print("\nMissing values:")
print(data.isnull().sum())


# ============================================================
# 4. SUPERVISED LEARNING
# ============================================================

print("\n" + "=" * 60)
print("SUPERVISED LEARNING")
print("=" * 60)

print("""
In supervised learning, the model learns from labelled data.

Input:
N, P, K, temperature, humidity, pH and rainfall

Target:
Crop name
""")


# Input features
X = data[
    [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ]
]


# Target variable
y = data["label"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 5. CLASSIFICATION
# ============================================================

print("\nClassification model: Random Forest")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nClassification Accuracy:")
print(round(accuracy * 100, 2), "%")


print("""
Why classification?

The target is a category such as:

Rice
Wheat
Maize
Cotton
...

Therefore, crop recommendation is a CLASSIFICATION problem.

Regression would not be suitable for predicting the crop name.
""")


# ============================================================
# 6. REGRESSION JUSTIFICATION
# ============================================================

print("\n" + "=" * 60)
print("REGRESSION VS CLASSIFICATION")
print("=" * 60)

print("""
CLASSIFICATION:
Used when the output is a category.

Example:
Input soil/environment conditions
        ↓
Recommended crop = Rice

REGRESSION:
Used when the output is a continuous numerical value.

Example:
Input soil/environment conditions
        ↓
Predicted crop yield = 4.5 tons/hectare

Therefore:

Crop recommendation → Classification
Crop yield prediction → Regression
""")


# ============================================================
# 7. UNSUPERVISED LEARNING
# ============================================================

print("\n" + "=" * 60)
print("UNSUPERVISED LEARNING")
print("=" * 60)

print("""
In unsupervised learning, the model does not use the crop label.

The objective is to discover natural groups of farms
with similar soil and environmental conditions.
""")


# Select only input features
cluster_data = data[
    [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ]
]


# Scale the data
scaler = StandardScaler()

scaled_data = scaler.fit_transform(cluster_data)


# K-Means clustering
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(scaled_data)


# Add cluster information
data["cluster"] = clusters


print("\nCluster assigned to each farm:")
print(data[["label", "cluster"]].head(10))


print("""
What does this mean?

K-Means does NOT predict the crop.

Instead, it discovers groups of farms having
similar soil and environmental characteristics.

Therefore:

Supervised learning → Predict crop
Unsupervised learning → Discover farm groups
""")


# ============================================================
# 8. DATA LEAKAGE
# ============================================================

print("\n" + "=" * 60)
print("DATA LEAKAGE")
print("=" * 60)

print("""
Realistic data leakage example:

Suppose we want to recommend a crop BEFORE planting.

The model should only use information available before planting.

If we accidentally include:

- Final harvest yield
- Future rainfall
- Harvest result
- Post-harvest soil measurements

the model receives information that would not be available
when the actual prediction is made.

This is called DATA LEAKAGE.

Data leakage can make the model appear extremely accurate
during testing but perform poorly in the real world.

Prevention:
Use only information available at prediction time.
""")


# ============================================================
# 9. OVERFITTING
# ============================================================

print("\n" + "=" * 60)
print("OVERFITTING")
print("=" * 60)

print("""
Overfitting occurs when a model learns the training data
too closely, including noise and unnecessary patterns.

We demonstrate this using a very deep Decision Tree.
""")


overfit_model = DecisionTreeClassifier(
    max_depth=None,
    random_state=42
)

overfit_model.fit(X_train, y_train)

train_pred = overfit_model.predict(X_train)

test_pred = overfit_model.predict(X_test)

train_accuracy = accuracy_score(
    y_train,
    train_pred
)

test_accuracy = accuracy_score(
    y_test,
    test_pred
)


print("Training accuracy:",
      round(train_accuracy * 100, 2), "%")

print("Testing accuracy:",
      round(test_accuracy * 100, 2), "%")


print("""
Effect:

If training accuracy is much higher than testing accuracy,
the model may be overfitting.

Such a model may perform poorly on new farms.
""")


# ============================================================
# 10. UNDERFITTING
# ============================================================

print("\n" + "=" * 60)
print("UNDERFITTING")
print("=" * 60)

print("""
Underfitting occurs when a model is too simple to learn
the important patterns in the data.

We demonstrate this using a Decision Tree with depth = 1.
""")


underfit_model = DecisionTreeClassifier(
    max_depth=1,
    random_state=42
)

underfit_model.fit(X_train, y_train)

under_train_pred = underfit_model.predict(X_train)

under_test_pred = underfit_model.predict(X_test)


under_train_accuracy = accuracy_score(
    y_train,
    under_train_pred
)

under_test_accuracy = accuracy_score(
    y_test,
    under_test_pred
)


print("Training accuracy:",
      round(under_train_accuracy * 100, 2), "%")

print("Testing accuracy:",
      round(under_test_accuracy * 100, 2), "%")


print("""
Effect:

If both training and testing accuracy are low,
the model may be underfitting.

Such a model has not learned enough from the data.
""")


# ============================================================
# 11. FINAL ANSWER SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL CONCLUSION")
print("=" * 60)

print("""
Real-world problem:
Crop suitability recommendation for farmers.

Supervised learning:
Used to predict the suitable crop using labelled data.

Classification:
Chosen because the output is a crop category.

Regression:
Would be used if predicting crop yield, which is a
continuous numerical value.

Unsupervised learning:
K-Means clustering can group farms with similar soil
and environmental characteristics.

Data leakage:
Using future or post-harvest information during training
can make the model unrealistically accurate.

Overfitting:
A complex model can memorize training data and perform
poorly on unseen farms.

Underfitting:
An overly simple model cannot learn important patterns.

A good ML model should generalize well to new data.
""")
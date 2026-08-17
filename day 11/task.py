# HOUSEHOLD ELECTRICITY CONSUMPTION PREDICTION
# Supervised Learning - Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# 1. CREATE DATASET
# --------------------------------------------------

data = {
    "Temperature": [20, 22, 24, 26, 28, 30, 32, 25, 27, 29,
                    21, 23, 31, 33, 26, 28, 30, 24, 22, 32],

    "Appliances": [2, 3, 3, 4, 5, 6, 7, 4, 5, 6,
                   2, 3, 7, 8, 4, 5, 6, 3, 2, 7],

    "TimeOfDay": [6, 7, 8, 10, 12, 14, 16, 18, 19, 20,
                  6, 8, 17, 19, 11, 13, 15, 9, 7, 18],

    "PreviousUsage": [2.1, 2.5, 2.8, 3.2, 4.0, 4.5, 5.2, 4.1, 4.8, 5.0,
                      2.0, 2.7, 5.5, 6.0, 3.5, 4.2, 4.7, 2.9, 2.4, 5.3],

    "Consumption": [2.5, 2.9, 3.2, 3.8, 4.7, 5.4, 6.3, 5.0, 5.7, 6.1,
                    2.4, 3.1, 6.6, 7.2, 4.1, 4.9, 5.6, 3.3, 2.8, 6.4]
}

df = pd.DataFrame(data)

print("DATASET")
print(df)

# --------------------------------------------------
# 2. FEATURES AND TARGET
# --------------------------------------------------

X = df[[
    "Temperature",
    "Appliances",
    "TimeOfDay",
    "PreviousUsage"
]]

y = df["Consumption"]

print("\nFEATURES")
print(X.head())

print("\nTARGET")
print(y.head())

# --------------------------------------------------
# 3. SPLIT DATA
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTRAINING DATA:", X_train.shape)
print("TESTING DATA:", X_test.shape)

# --------------------------------------------------
# 4. CREATE REGRESSION MODEL
# --------------------------------------------------

model = LinearRegression()

# --------------------------------------------------
# 5. TRAIN MODEL
# --------------------------------------------------

model.fit(X_train, y_train)

print("\nMODEL TRAINING COMPLETED")

# --------------------------------------------------
# 6. PREDICTION
# --------------------------------------------------

y_pred = model.predict(X_test)

print("\nACTUAL VALUES")
print(y_test.values)

print("\nPREDICTED VALUES")
print(y_pred)

# --------------------------------------------------
# 7. MODEL EVALUATION
# --------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print("Mean Absolute Error :", mae)
print("Mean Squared Error  :", mse)
print("R2 Score            :", r2)

# --------------------------------------------------
# 8. PREDICT NEW HOUSEHOLD CONSUMPTION
# --------------------------------------------------

new_transaction = [[30, 6, 19, 5.0]]

prediction = model.predict(new_transaction)

print("\n" + "=" * 50)
print("NEW HOUSEHOLD PREDICTION")
print("=" * 50)

print("Temperature       :", new_transaction[0][0])
print("Appliances Used   :", new_transaction[0][1])
print("Time of Day       :", new_transaction[0][2])
print("Previous Usage    :", new_transaction[0][3])

print("\nPredicted Electricity Consumption:",
      round(prediction[0], 2), "kWh")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("electricity_data_800.csv")

print("DATASET")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

# 2. Features and Target
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

# 3. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# 4. Create Linear Regression model
model = LinearRegression()

# 5. Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")

# 6. Make predictions
y_pred = model.predict(X_test)

print("\nACTUAL VALUES")
print(y_test.head(10).values)

print("\nPREDICTED VALUES")
print(y_pred[:10])

# 7. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("Mean Absolute Error:", round(mae, 3))
print("Mean Squared Error:", round(mse, 3))
print("R2 Score:", round(r2, 3))

# 8. Predict new electricity consumption
new_data = pd.DataFrame({
    "Temperature": [30],
    "Appliances": [6],
    "TimeOfDay": [19],
    "PreviousUsage": [5.0]
})

prediction = model.predict(new_data)

print("\n==============================")
print("NEW PREDICTION")
print("==============================")

print("Temperature:", 30, "°C")
print("Appliances:", 6)
print("Time of Day:", 19)
print("Previous Usage:", 5.0, "kWh")

print(
    "Predicted Electricity Consumption:",
    round(prediction[0], 2),
    "kWh"
)
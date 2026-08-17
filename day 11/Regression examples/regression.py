import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the dataset
data = pd.read_csv("Housing.csv")

# 2. Display the dataset
print("First 5 rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nColumn names:")
print(data.columns)

# 3. Convert categorical columns into numbers
data = pd.get_dummies(data, drop_first=True)

print("\nDataset after encoding:")
print(data.head())

# 4. Separate input features (X) and target (y)

# X = Features used to predict price
X = data.drop("price", axis=1)

# y = Price that we want to predict
y = data["price"]

# 5. Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# 6. Create Linear Regression model
model = LinearRegression()

# 7. Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")

# 8. Predict house prices
y_pred = model.predict(X_test)

# 9. Display actual and predicted prices
print("\nActual prices:")
print(y_test.head().values)

print("\nPredicted prices:")
print(y_pred[:5])

# 10. Evaluate the model

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# 11. Predict the price of one new house
# Values:
# area = 5000
# bedrooms = 3
# bathrooms = 2
# stories = 2
# mainroad = yes
# guestroom = no
# basement = no
# hotwaterheating = no
# airconditioning = yes
# parking = 2
# prefarea = yes
# furnishingstatus = semi-furnished

new_house = pd.DataFrame({
    "area": [5000],
    "bedrooms": [3],
    "bathrooms": [2],
    "stories": [2],
    "mainroad_yes": [1],
    "guestroom_yes": [0],
    "basement_yes": [0],
    "hotwaterheating_yes": [0],
    "airconditioning_yes": [1],
    "parking": [2],
    "prefarea_yes": [1],
    "furnishingstatus_semi-furnished": [1],
    "furnishingstatus_unfurnished": [0]
})

# Make sure new house has the same columns as training data
new_house = new_house.reindex(columns=X.columns, fill_value=0)

# Predict
new_price = model.predict(new_house)

print("\nNew House Details:")
print(new_house)

print("\nPredicted House Price:")
print(new_price[0])
import pandas as pd

# Create a Series with different letter cases and missing values
names = pd.Series(["Bindu", "RAHUL", None, "Anu", "KIRAN", None])

print("Original Names:")
print(names)

# Detect missing values
print("\nMissing Values:")
print(names.isna())

# Fill missing values
names = names.fillna("Unknown")

print("\nAfter Filling Missing Values:")
print(names)

# Convert names to lowercase
names = names.str.lower()

print("\nNames in Lowercase:")
print(names)

# Filter names containing the letter 'a'
filtered_names = names[names.str.contains("a")]

print("\nNames containing letter 'a':")
print(filtered_names)
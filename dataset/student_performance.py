import pandas as pd
from pathlib import Path

# ==========================================
# 1. LOAD DATASET
# ==========================================

file_path = Path(__file__).parent / "student_performance.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("ORIGINAL DATASET")
print("=" * 60)

print(df)


# ==========================================
# 2. ORIGINAL SHAPE
# ==========================================

print("\n" + "=" * 60)
print("ORIGINAL DATASET SHAPE")
print("=" * 60)

print("Shape:", df.shape)


# ==========================================
# 3. IDENTIFY MISSING VALUES
# ==========================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())

print("\nTotal Missing Values:",
      df.isnull().sum().sum())


# ==========================================
# 4. DETECT DUPLICATES
# ==========================================

duplicate_columns = [
    "Name",
    "Age",
    "Gender",
    "Maths",
    "Science",
    "English",
    "Attendance",
    "Grade"
]

duplicate_count = df.duplicated(
    subset=duplicate_columns
).sum()

print("\n" + "=" * 60)
print("DUPLICATES")
print("=" * 60)

print("Duplicate Records:", duplicate_count)

print("\nDuplicate Rows:")

print(
    df[
        df.duplicated(
            subset=duplicate_columns,
            keep=False
        )
    ]
)


# ==========================================
# 5. REMOVE DUPLICATES
# ==========================================

df = df.drop_duplicates(
    subset=duplicate_columns
)

print("\nShape After Removing Duplicates:",
      df.shape)


# ==========================================
# 6. HANDLE MISSING VALUES
# ==========================================

# Age
df["Age"] = (
    df["Age"]
    .fillna(df["Age"].mean())
    .round()
    .astype(int)
)

# Maths
df["Maths"] = (
    df["Maths"]
    .fillna(df["Maths"].mean())
    .round()
    .astype(int)
)

# Science
df["Science"] = (
    df["Science"]
    .fillna(df["Science"].mean())
    .round()
    .astype(int)
)

# English
df["English"] = (
    df["English"]
    .fillna(df["English"].mean())
    .round()
    .astype(int)
)

# Attendance
df["Attendance"] = (
    df["Attendance"]
    .fillna(df["Attendance"].mean())
    .round()
    .astype(int)
)


# ==========================================
# 7. CHECK MISSING VALUES AGAIN
# ==========================================

print("\n" + "=" * 60)
print("MISSING VALUES AFTER CLEANING")
print("=" * 60)

print(df.isnull().sum())

print("\nTotal Missing Values:",
      df.isnull().sum().sum())


# ==========================================
# 8. CHECK DATA TYPES
# ==========================================

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)


# ==========================================
# 9. CLEANED DATASET SHAPE
# ==========================================

print("\n" + "=" * 60)
print("CLEANED DATASET SHAPE")
print("=" * 60)

print("Shape:", df.shape)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ==========================================
# 10. FINAL CLEANED DATASET
# ==========================================

print("\n" + "=" * 60)
print("CLEANED DATASET")
print("=" * 60)

print(df)


# ==========================================
# 11. SAVE CLEANED DATASET
# ==========================================

output_file = (
    Path(__file__).parent /
    "cleaned_student_performance.csv"
)

df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 60)
print("SUCCESS")
print("=" * 60)

print("Cleaned dataset saved successfully!")
print(df.describe())
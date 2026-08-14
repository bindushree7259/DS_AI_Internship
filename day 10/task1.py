# ============================================================
# COMPLETE EDA PROJECT
# Dataset: Student Performance
# ============================================================

# -----------------------------
# 1. IMPORT LIBRARIES
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_theme(style="whitegrid")


# -----------------------------
# 2. CREATE DATASET
# -----------------------------

data = {
    "Student_ID": [101,102,103,104,105,106,107,108,109,110,111,112,113,114,115],

    "Age": [
        18,19,18,20,19,18,21,20,19,18,22,20,19,21,18
    ],

    "Study_Hours": [
        2,4,3,6,5,2,8,7,4,3,9,6,5,8,2
    ],

    "Attendance": [
        75,85,80,92,88,70,95,90,82,78,97,91,86,94,68
    ],

    "Maths": [
        65,78,72,90,85,55,95,88,76,68,98,91,84,93,50
    ],

    "Science": [
        70,80,75,92,87,60,96,90,78,72,99,89,85,95,55
    ],

    "English": [
        68,82,74,88,90,62,94,86,80,75,96,92,87,91,58
    ],

    "Gender": [
        "Female","Male","Female","Male","Female",
        "Male","Female","Male","Female","Male",
        "Female","Male","Female","Male","Female"
    ],

    "Grade": [
        "B","B","B","A","A",
        "C","A","A","B","B",
        "A","A","A","A","C"
    ]
}

df = pd.DataFrame(data)


# ============================================================
# 3. BASIC DATA UNDERSTANDING
# ============================================================

print("=" * 60)
print("DATASET")
print("=" * 60)

print(df)


print("\n" + "=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)

print(df.head())


print("\n" + "=" * 60)
print("LAST 5 ROWS")
print("=" * 60)

print(df.tail())


print("\n" + "=" * 60)
print("DATASET SHAPE")
print("=" * 60)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)

print(df.columns.tolist())


print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)


print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

df.info()


# ============================================================
# 4. STATISTICAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(df.describe())


# ============================================================
# 5. MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing_values = df.isnull().sum()

print(missing_values)


print("\nTotal missing values:",
      df.isnull().sum().sum())


# Visualization of missing values

plt.figure(figsize=(8, 4))

sns.heatmap(
    df.isnull(),
    cbar=False
)

plt.title("Missing Value Heatmap")
plt.show()


# ============================================================
# 6. DUPLICATE VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE ANALYSIS")
print("=" * 60)

duplicates = df.duplicated().sum()

print("Number of duplicate rows:", duplicates)


# ============================================================
# 7. IDENTIFY NUMERICAL AND CATEGORICAL COLUMNS
# ============================================================

numeric_cols = [
    "Age",
    "Study_Hours",
    "Attendance",
    "Maths",
    "Science",
    "English"
]

categorical_cols = [
    "Gender",
    "Grade"
]


print("\n" + "=" * 60)
print("NUMERICAL COLUMNS")
print("=" * 60)

print(numeric_cols)


print("\n" + "=" * 60)
print("CATEGORICAL COLUMNS")
print("=" * 60)

print(categorical_cols)


# ============================================================
# 8. UNIVARIATE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("UNIVARIATE ANALYSIS")
print("=" * 60)


# Numerical distributions

for col in numeric_cols:

    print("\nAnalyzing:", col)

    print(df[col].describe())

    plt.figure(figsize=(7, 5))

    sns.histplot(
        data=df,
        x=col,
        kde=True
    )

    plt.title("Distribution of " + col)
    plt.xlabel(col)
    plt.ylabel("Frequency")

    plt.show()


# ============================================================
# 9. BOX PLOT ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("BOXPLOT ANALYSIS")
print("=" * 60)


for col in numeric_cols:

    plt.figure(figsize=(7, 4))

    sns.boxplot(
        x=df[col]
    )

    plt.title("Boxplot of " + col)
    plt.xlabel(col)

    plt.show()


# ============================================================
# 10. CATEGORICAL FREQUENCY ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("CATEGORICAL ANALYSIS")
print("=" * 60)


for col in categorical_cols:

    print("\nFrequency of", col)

    print(df[col].value_counts())


    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=df,
        x=col
    )

    plt.title("Distribution of " + col)
    plt.xlabel(col)
    plt.ylabel("Count")

    plt.show()


# ============================================================
# 11. SKEWNESS ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("SKEWNESS ANALYSIS")
print("=" * 60)

skewness = df[numeric_cols].skew()

print(skewness)


# Interpretation

print("\nSkewness Interpretation:")

for col in numeric_cols:

    value = skewness[col]

    if value > 1:
        result = "Highly positively skewed"

    elif value > 0.5:
        result = "Moderately positively skewed"

    elif value < -1:
        result = "Highly negatively skewed"

    elif value < -0.5:
        result = "Moderately negatively skewed"

    else:
        result = "Approximately symmetric"

    print(col, ":", result)


# ============================================================
# 12. CREATE AVERAGE MARKS
# ============================================================

df["Average_Marks"] = (
    df["Maths"] +
    df["Science"] +
    df["English"]
) / 3


print("\n" + "=" * 60)
print("AVERAGE MARKS")
print("=" * 60)

print(
    df[
        [
            "Student_ID",
            "Maths",
            "Science",
            "English",
            "Average_Marks"
        ]
    ]
)


# ============================================================
# 13. BIVARIATE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("BIVARIATE ANALYSIS")
print("=" * 60)


# --------------------------------
# Study Hours vs Maths
# --------------------------------

plt.figure(figsize=(7, 5))

sns.scatterplot(
    data=df,
    x="Study_Hours",
    y="Maths"
)

plt.title("Study Hours vs Maths Marks")
plt.xlabel("Study Hours")
plt.ylabel("Maths Marks")

plt.show()


# --------------------------------
# Study Hours vs Average Marks
# --------------------------------

plt.figure(figsize=(7, 5))

sns.scatterplot(
    data=df,
    x="Study_Hours",
    y="Average_Marks"
)

plt.title("Study Hours vs Average Marks")
plt.xlabel("Study Hours")
plt.ylabel("Average Marks")

plt.show()


# --------------------------------
# Attendance vs Average Marks
# --------------------------------

plt.figure(figsize=(7, 5))

sns.scatterplot(
    data=df,
    x="Attendance",
    y="Average_Marks"
)

plt.title("Attendance vs Average Marks")
plt.xlabel("Attendance")
plt.ylabel("Average Marks")

plt.show()


# --------------------------------
# Age vs Average Marks
# --------------------------------

plt.figure(figsize=(7, 5))

sns.scatterplot(
    data=df,
    x="Age",
    y="Average_Marks"
)

plt.title("Age vs Average Marks")
plt.xlabel("Age")
plt.ylabel("Average Marks")

plt.show()


# ============================================================
# 14. BIVARIATE CATEGORICAL ANALYSIS
# ============================================================


# Grade vs Average Marks

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Grade",
    y="Average_Marks"
)

plt.title("Grade vs Average Marks")

plt.show()


# Gender vs Average Marks

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Gender",
    y="Average_Marks"
)

plt.title("Gender vs Average Marks")

plt.show()


# ============================================================
# 15. DEPARTMENT-STYLE COMPARISON
# ============================================================
# Grade frequency

grade_counts = df["Grade"].value_counts()

print("\nGrade Distribution:")
print(grade_counts)


plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Grade",
    order=["A", "B", "C"]
)

plt.title("Grade Distribution")
plt.xlabel("Grade")
plt.ylabel("Number of Students")

plt.show()


# ============================================================
# 16. CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)


correlation_cols = [
    "Age",
    "Study_Hours",
    "Attendance",
    "Maths",
    "Science",
    "English",
    "Average_Marks"
]

correlation = df[correlation_cols].corr()

print(correlation)


# ============================================================
# 17. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()


# ============================================================
# 18. FIND STRONGEST CORRELATION
# ============================================================

corr_matrix = correlation.copy()

# Remove diagonal
np.fill_diagonal(corr_matrix.values, np.nan)

# Find strongest correlation
strongest_pair = (
    corr_matrix
    .abs()
    .stack()
    .idxmax()
)

strongest_value = corr_matrix.loc[strongest_pair]

print("\n" + "=" * 60)
print("STRONGEST CORRELATION")
print("=" * 60)

print("Variables:", strongest_pair)
print("Correlation:", strongest_value)


# ============================================================
# 19. OUTLIER DETECTION USING IQR
# ============================================================

print("\n" + "=" * 60)
print("OUTLIER ANALYSIS")
print("=" * 60)


outlier_summary = {}


for col in numeric_cols:

    Q1 = df[col].quantile(0.25)

    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR

    upper_limit = Q3 + 1.5 * IQR


    outliers = df[
        (df[col] < lower_limit) |
        (df[col] > upper_limit)
    ]


    outlier_summary[col] = len(outliers)


    print("\nColumn:", col)

    print("Q1:", Q1)

    print("Q3:", Q3)

    print("IQR:", IQR)

    print("Lower Limit:", lower_limit)

    print("Upper Limit:", upper_limit)

    print("Number of Outliers:", len(outliers))


    if len(outliers) > 0:

        print("Outlier values:")

        print(outliers[["Student_ID", col]])


# ============================================================
# 20. OUTLIER SUMMARY GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    x=list(outlier_summary.keys()),
    y=list(outlier_summary.values())
)

plt.title("Number of Outliers by Variable")
plt.xlabel("Variable")
plt.ylabel("Number of Outliers")

plt.xticks(rotation=45)

plt.show()


# ============================================================
# 21. PAIRPLOT
# ============================================================

print("\nGenerating Pairplot...")

sns.pairplot(
    df,
    vars=[
        "Study_Hours",
        "Attendance",
        "Maths",
        "Science",
        "English"
    ],
    hue="Gender"
)

plt.show()


# ============================================================
# 22. TOP PERFORMING STUDENTS
# ============================================================

print("\n" + "=" * 60)
print("TOP PERFORMING STUDENTS")
print("=" * 60)


top_students = df.sort_values(
    "Average_Marks",
    ascending=False
)

print(
    top_students[
        [
            "Student_ID",
            "Study_Hours",
            "Attendance",
            "Average_Marks",
            "Grade"
        ]
    ].head()
)


# ============================================================
# 23. LOW PERFORMING STUDENTS
# ============================================================

print("\n" + "=" * 60)
print("LOW PERFORMING STUDENTS")
print("=" * 60)


low_students = df.sort_values(
    "Average_Marks",
    ascending=True
)

print(
    low_students[
        [
            "Student_ID",
            "Study_Hours",
            "Attendance",
            "Average_Marks",
            "Grade"
        ]
    ].head()
)


# ============================================================
# 24. AVERAGE PERFORMANCE BY GENDER
# ============================================================

print("\n" + "=" * 60)
print("AVERAGE PERFORMANCE BY GENDER")
print("=" * 60)

gender_performance = df.groupby(
    "Gender"
)["Average_Marks"].mean()

print(gender_performance)


# ============================================================
# 25. AVERAGE PERFORMANCE BY GRADE
# ============================================================

print("\n" + "=" * 60)
print("AVERAGE PERFORMANCE BY GRADE")
print("=" * 60)

grade_performance = df.groupby(
    "Grade"
)["Average_Marks"].mean()

print(grade_performance)


# ============================================================
# 26. FINAL INSIGHTS
# ============================================================

print("\n" + "=" * 60)
print("EDA FINAL SUMMARY")
print("=" * 60)

print("""
1. The dataset contains student demographic and academic information.

2. Missing-value analysis was performed and the dataset contains
   no missing values.

3. Duplicate-value analysis was performed.

4. Univariate analysis was performed using histograms,
   frequency charts and boxplots.

5. Skewness analysis was performed on numerical variables.

6. Bivariate analysis was performed to study relationships
   between study hours, attendance and academic marks.

7. Correlation analysis was performed using a correlation matrix
   and heatmap.

8. Outliers were detected using the IQR method.

9. Pairplot was used to visualize relationships among numerical
   variables.

10. Study hours and attendance can be examined as important
    factors associated with student performance.

11. Maths, Science and English marks can be compared to identify
    relationships among academic subjects.

12. The EDA provides useful patterns that can be used for
    further statistical analysis or machine learning.
""")


print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# ============================================================
# 1. CREATE DATASET
# ============================================================

data = {
    "Student_ID": [101,102,103,104,105,106,107,108,109,110,111,112,113,114,115],

    "Age": [18,19,18,20,19,18,21,20,19,18,22,20,19,21,18],

    "Study_Hours": [2,4,3,6,5,2,8,7,4,3,9,6,5,8,2],

    "Attendance": [75,85,80,92,88,70,95,90,82,78,97,91,86,94,68],

    "Maths": [65,78,72,90,85,55,95,88,76,68,98,91,84,93,50],

    "Science": [70,80,75,92,87,60,96,90,78,72,99,89,85,95,55],

    "English": [68,82,74,88,90,62,94,86,80,75,96,92,87,91,58],

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

# CREATE DATAFRAME
df = pd.DataFrame(data)

print("=" * 70)
print("COMPLETE EDA - STUDENT PERFORMANCE DATASET")
print("=" * 70)

print("\nDATASET:")
print(df)


# ============================================================
# 2. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nStatistical Summary:")
print(df.describe())


# ============================================================
# 3. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE ANALYSIS")
print("=" * 70)

print(df.isnull().sum())

print("\nTotal Missing Values:",
      df.isnull().sum().sum())


# ============================================================
# 4. DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE ANALYSIS")
print("=" * 70)

print("Duplicate Rows:", df.duplicated().sum())


# ============================================================
# 5. DEFINE COLUMNS
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


# ============================================================
# 6. UNIVARIATE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("UNIVARIATE ANALYSIS")
print("=" * 70)

for col in numeric_cols:

    print("\n", col)
    print(df[col].describe())

    # Histogram
    plt.figure(figsize=(7, 5))

    sns.histplot(
        df[col],
        kde=True
    )

    plt.title("Distribution of " + col)
    plt.xlabel(col)
    plt.ylabel("Frequency")

    plt.show()

    # Boxplot
    plt.figure(figsize=(7, 4))

    sns.boxplot(
        x=df[col]
    )

    plt.title("Boxplot of " + col)

    plt.show()


# ============================================================
# 7. CATEGORICAL ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CATEGORICAL ANALYSIS")
print("=" * 70)

for col in categorical_cols:

    print("\nFrequency of", col)

    print(df[col].value_counts())

    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=df,
        x=col
    )

    plt.title("Distribution of " + col)

    plt.show()


# ============================================================
# 8. SKEWNESS ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("SKEWNESS ANALYSIS")
print("=" * 70)

skewness = df[numeric_cols].skew()

print(skewness)

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
# 9. CREATE AVERAGE MARKS
# ============================================================

df["Average_Marks"] = (
    df["Maths"] +
    df["Science"] +
    df["English"]
) / 3

print("\n" + "=" * 70)
print("AVERAGE MARKS")
print("=" * 70)

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
# 10. BIVARIATE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("BIVARIATE ANALYSIS")
print("=" * 70)


# Study Hours vs Maths

plt.figure(figsize=(7, 5))

sns.scatterplot(
    data=df,
    x="Study_Hours",
    y="Maths"
)

plt.title("Study Hours vs Maths Marks")

plt.show()


# Study Hours vs Average Marks

plt.figure(figsize=(7, 5))

sns.scatterplot(
    data=df,
    x="Study_Hours",
    y="Average_Marks"
)

plt.title("Study Hours vs Average Marks")

plt.show()


# Attendance vs Average Marks

plt.figure(figsize=(7, 5))

sns.scatterplot(
    data=df,
    x="Attendance",
    y="Average_Marks"
)

plt.title("Attendance vs Average Marks")

plt.show()


# Age vs Average Marks

plt.figure(figsize=(7, 5))

sns.scatterplot(
    data=df,
    x="Age",
    y="Average_Marks"
)

plt.title("Age vs Average Marks")

plt.show()


# ============================================================
# 11. CATEGORICAL BIVARIATE ANALYSIS
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
# 12. CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

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
# 13. CORRELATION HEATMAP
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
# 14. STRONGEST CORRELATION
# ============================================================

temp_corr = correlation.copy()

np.fill_diagonal(
    temp_corr.values,
    np.nan
)

strongest_pair = (
    temp_corr.abs()
    .stack()
    .idxmax()
)

strongest_value = temp_corr.loc[
    strongest_pair[0],
    strongest_pair[1]
]

print("\n" + "=" * 70)
print("STRONGEST CORRELATION")
print("=" * 70)

print("Variables:", strongest_pair)
print("Correlation:", strongest_value)


# ============================================================
# 15. OUTLIER DETECTION
# ============================================================

print("\n" + "=" * 70)
print("OUTLIER ANALYSIS")
print("=" * 70)

outlier_summary = {}

for col in numeric_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower) |
        (df[col] > upper)
    ]

    outlier_summary[col] = len(outliers)

    print("\nColumn:", col)
    print("Q1:", Q1)
    print("Q3:", Q3)
    print("IQR:", IQR)
    print("Lower Limit:", lower)
    print("Upper Limit:", upper)
    print("Number of Outliers:", len(outliers))

    if len(outliers) > 0:
        print("Outliers:")
        print(outliers[["Student_ID", col]])


# ============================================================
# 16. PAIRPLOT
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
# 17. TOP PERFORMING STUDENTS
# ============================================================

print("\n" + "=" * 70)
print("TOP PERFORMING STUDENTS")
print("=" * 70)

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
# 18. LOW PERFORMING STUDENTS
# ============================================================

print("\n" + "=" * 70)
print("LOW PERFORMING STUDENTS")
print("=" * 70)

low_students = df.sort_values(
    "Average_Marks"
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
# 19. DETAILED INSIGHTS
# ============================================================

print("\n" + "=" * 70)
print("DETAILED EDA INSIGHTS")
print("=" * 70)


# Dataset insight

print("\n1. DATASET OVERVIEW")
print("-------------------")

print(
    f"The dataset contains {df.shape[0]} students "
    f"and {df.shape[1]} variables."
)

print(
    "The dataset contains demographic information, "
    "study habits, attendance and academic performance."
)


# Missing values

print("\n2. MISSING VALUES")
print("------------------")

if df.isnull().sum().sum() == 0:

    print(
        "There are no missing values in the dataset."
    )

else:

    print(
        "Missing values are present and require treatment."
    )


# Duplicate

print("\n3. DUPLICATES")
print("-------------")

if df.duplicated().sum() == 0:

    print(
        "No duplicate rows were found."
    )

else:

    print(
        f"{df.duplicated().sum()} duplicate rows were found."
    )


# Study hours

print("\n4. STUDY HOURS")
print("---------------")

print(
    f"Average study time: "
    f"{df['Study_Hours'].mean():.2f} hours."
)

study_corr = df[
    "Study_Hours"
].corr(
    df["Average_Marks"]
)

print(
    f"Correlation between study hours and average marks: "
    f"{study_corr:.2f}"
)

if study_corr > 0:

    print(
        "There is a positive relationship between study "
        "hours and academic performance."
    )


# Attendance

print("\n5. ATTENDANCE")
print("-------------")

print(
    f"Average attendance: "
    f"{df['Attendance'].mean():.2f}%."
)

attendance_corr = df[
    "Attendance"
].corr(
    df["Average_Marks"]
)

print(
    f"Correlation between attendance and average marks: "
    f"{attendance_corr:.2f}"
)

if attendance_corr > 0:

    print(
        "Higher attendance is generally associated with "
        "better academic performance."
    )


# Subject performance

print("\n6. SUBJECT PERFORMANCE")
print("----------------------")

subject_means = df[
    ["Maths", "Science", "English"]
].mean()

print(subject_means)

best_subject = subject_means.idxmax()

lowest_subject = subject_means.idxmin()

print(
    f"Highest average score: {best_subject}"
)

print(
    f"Lowest average score: {lowest_subject}"
)


# Grade

print("\n7. GRADE DISTRIBUTION")
print("---------------------")

grade_counts = df["Grade"].value_counts()

print(grade_counts)

print(
    f"Most common grade: "
    f"{grade_counts.idxmax()}"
)


# Top student

print("\n8. TOP PERFORMING STUDENT")
print("-------------------------")

top_student = df.loc[
    df["Average_Marks"].idxmax()
]

print(
    "Student ID:",
    top_student["Student_ID"]
)

print(
    "Average Marks:",
    round(top_student["Average_Marks"], 2)
)

print(
    "Study Hours:",
    top_student["Study_Hours"]
)

print(
    "Attendance:",
    top_student["Attendance"]
)

print(
    "Grade:",
    top_student["Grade"]
)


# Lowest student

print("\n9. LOWEST PERFORMING STUDENT")
print("----------------------------")

lowest_student = df.loc[
    df["Average_Marks"].idxmin()
]

print(
    "Student ID:",
    lowest_student["Student_ID"]
)

print(
    "Average Marks:",
    round(lowest_student["Average_Marks"], 2)
)

print(
    "Study Hours:",
    lowest_student["Study_Hours"]
)

print(
    "Attendance:",
    lowest_student["Attendance"]
)

print(
    "Grade:",
    lowest_student["Grade"]
)


# Gender

print("\n10. GENDER-WISE PERFORMANCE")
print("---------------------------")

gender_avg = df.groupby(
    "Gender"
)["Average_Marks"].mean()

print(gender_avg)

print(
    "This comparison is descriptive only because "
    "the dataset is small."
)


# Outliers

print("\n11. OUTLIER INSIGHTS")
print("--------------------")

total_outliers = sum(
    outlier_summary.values()
)

if total_outliers == 0:

    print(
        "No significant outliers were detected "
        "using the IQR method."
    )

else:

    print(
        f"A total of {total_outliers} outlier observations "
        "were detected across numerical variables."
    )


# Skewness

print("\n12. SKEWNESS INSIGHTS")
print("---------------------")

for col in numeric_cols:

    value = df[col].skew()

    if value > 1:

        result = "Highly right-skewed"

    elif value > 0.5:

        result = "Moderately right-skewed"

    elif value < -1:

        result = "Highly left-skewed"

    elif value < -0.5:

        result = "Moderately left-skewed"

    else:

        result = "Approximately symmetric"

    print(
        f"{col}: {value:.2f} → {result}"
    )


# ============================================================
# 20. FINAL CONCLUSION
# ============================================================

print("\n" + "=" * 70)
print("FINAL CONCLUSION")
print("=" * 70)

print("""
The Exploratory Data Analysis was performed on the Student
Performance dataset.

The analysis included dataset understanding, missing-value
checking, duplicate checking, univariate analysis, bivariate
analysis, skewness analysis, correlation analysis, visualization
and outlier detection.

The analysis indicates that study hours and attendance have
positive relationships with academic performance in this sample.

The subject scores also show relationships with one another,
indicating that students who perform well in one subject often
perform well in other subjects.

Histograms were used to understand distributions, boxplots were
used to identify possible outliers, scatterplots were used to
study relationships, and a correlation heatmap was used to
visualize numerical relationships.

Overall, the EDA helps identify important patterns in student
performance and provides a foundation for further statistical
analysis or machine learning.

Since this is a small dataset, the findings should be considered
descriptive rather than universally applicable.
""")


print("\n" + "=" * 70)
print("COMPLETE EDA FINISHED SUCCESSFULLY")
print("=" * 70)
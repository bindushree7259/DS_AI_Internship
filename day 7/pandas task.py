import pandas as pd

# Create a Pandas Series of student marks
marks = pd.Series(
    [85, 55, 72, 90, 45],
    index=["Maths", "Science", "English", "Computer", "Kannada"]
)

# Display the Series
print("Student Marks:")
print(marks)

# Access value using position
print("\nMark at position 0:", marks.iloc[0])

# Access value using label
print("Mark in Maths:", marks["Maths"])

# Print values
print("\nValues:")
print(marks.values)

# Print index
print("\nIndex:")
print(marks.index)

# Boolean masking - students/subjects scoring above 60
print("\nMarks above 60:")
print(marks[marks > 60])
# Function to calculate grade
def calculate_grade(avg):
    if avg >= 90:
        return "A+"
    elif avg >= 80:
        return "A"
    elif avg >= 70:
        return "B"
    elif avg >= 60:
        return "C"
    elif avg >= 50:
        return "D"
    else:
        return "F"

# Function to enter student details
def student_details():
    students = []

    while True:
        name = input("\nEnter Student Name (or 'done' to finish): ")

        if name.lower() == "done":
            break

        marks = []

        while True:
            mark = input("Enter Mark (or 'done' to finish marks): ")

            if mark.lower() == "done":
                break

            marks.append(float(mark))

        if len(marks) > 0:
            average = sum(marks) / len(marks)
            grade = calculate_grade(average)
        else:
            average = 0
            grade = "No Grade"

        students.append([name, marks, average, grade])

    print("\n------ Student Report ------")
    for student in students:
        print("Name    :", student[0])
        print("Marks   :", student[1])
        print("Average :", round(student[2], 2))
        print("Grade   :", student[3])
        print("----------------------------")

# Main function
student_details()
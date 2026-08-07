def get_grade(percentage):
    if percentage >= 90:
        return "A+"
    elif percentage >= 80:
        return "A"
    elif percentage >= 70:
        return "B"
    elif percentage >= 60:
        return "C"
    elif percentage >= 50:
        return "D"
    else:
        return "Fail"


def student_result():
    name = input("Enter Student Name: ")
    n = int(input("Enter Number of Subjects: "))

    marks = []
    total = 0

    for i in range(n):
        mark = int(input(f"Enter Marks of Subject {i+1} (Out of 100): "))
        marks.append(mark)
        total += mark

    total_marks = n * 100
    percentage = (total / total_marks) * 100
    grade = get_grade(percentage)

    print("\n===== STUDENT RESULT =====")
    print("Student Name :", name)
    print("Marks        :", marks)
    print("Total Marks  :", total, "/", total_marks)
    print("Percentage   :", round(percentage, 2), "%")
    print("Grade        :", grade)


# Function Call
student_result()